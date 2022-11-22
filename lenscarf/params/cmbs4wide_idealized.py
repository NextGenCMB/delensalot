"""Iterative reconstruction for masked polarization CMB data

    On idealized, full-sky maps, with homogeneous noise
    The difference to 'cmbs4wide_planckmask' is usage of a simpler filter instance, which works in harmonic space


FIXME's :
    plancklens independent QEs ?
    degrade method of _wl_ filters
    check of invertibility at very first step
    mf_resp for EB-like ?
"""
import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens
from plancklens import utils
from plancklens import qresp
from plancklens import qest, qecl
from plancklens.qcinv import cd_solve

from plancklens.sims import maps, phas, planck2018_sims
from plancklens.filt import filt_simple, filt_util

from lenscarf import remapping
from lenscarf import utils_scarf, utils_sims
from lenscarf.iterators import cs_iterator as scarf_iterator, steps

from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam, almxfl, alm_copy
from lenscarf.opfilt.opfilt_iso_ee_wl import alm_filter_nlev_wl

suffix = 'cmbs4_idealized2' # descriptor to distinguish this parfile from others...
TEMP =  opj(os.environ['SCRATCH'], 'lenscarfrecs', suffix)

lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (3000, 3000, 1., 1., np.sqrt(2.))

lmin_tlm, lmin_elm, lmin_blm = (30, 30, 30) # The fiducial transfer functions are set to zero below these lmins
# for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

lmax_qlm, mmax_qlm = (4000, 4000) # Lensing map is reconstructed down to this lmax and mmax
# NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
lmax_unl, mmax_unl = (4000, 4000) # Delensed CMB is reconstructed down to this lmax and mmax


#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
nside = 2048
zbounds     = (-1.,1.) # colatitude sky cuts for noise variance maps (We could exclude all rings which are completely masked)
ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(nside, zbounds=zbounds)

zbounds_len = (-1.,1.) # Outside of these bounds the reconstructed maps are assumed to be zero
pb_ctr, pb_extent = (0., 2 * np.pi) # Longitude cuts, if any, in the form (center of patch, patch extent)
lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(lmax_unl, 2, zbounds=zbounds_len)
lenjob_pbgeometry =utils_scarf.pbdGeometry(lenjob_geometry, utils_scarf.pbounds(pb_ctr, pb_extent))
lensres = 1.7  # Deflection operations will be performed at this resolution
Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted
stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
mc_sims_mf_it0 = np.array([]) # sims to use to build the very first iteration mean-field (QE mean-field) Here 0 since idealized


# Multigrid chain descriptor
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
#------------------

# Fiducial CMB spectra for QE and iterative reconstructions
# (here we use very lightly suboptimal lensed spectra QE weights)
cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))

# Fiducial model of the transfer function
transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
transf_d = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}
# Isotropic approximation to the filtering (used eg for response calculations)
ftl =  cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel =  cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl =  cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
ftl_unl =  cli(cls_unl['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel_unl =  cli(cls_unl['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl_unl =  cli(cls_unl['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# -------------------------
# ---- Input simulation libraries. Here we use the NERSC FFP10 CMBs with homogeneous noise and consistent transfer function
#       We define explictly the phase library such that we can use the same phases for for other purposes in the future as well if needed
#       I am putting here the phases in the home directory such that they dont get NERSC auto-purged
pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%nside), 3, (hp.nside2npix(nside),)) # T, Q, and U noise phases
#       actual data transfer function for the sim generation:
transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full FFP10 cmb's which are given to 4096)
sims      = maps.cmb_maps_nlev(planck2018_sims.cmb_len_ffp10(), transf_dat, nlev_t, nlev_p, nside, pix_lib_phas=pix_phas)

# Makes the simulation library consistent with the zbounds
sims_MAP  = utils_sims.ztrunc_sims(sims, nside, [zbounds])
# -------------------------

ivfs   = filt_simple.library_fullsky_sepTP(opj(TEMP, 'ivfs'), sims, nside, transf_d, cls_len, ftl, fel, fbl, cache=True)

# ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)
qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], nside, lmax_qlm=lmax_qlm)
qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)
# -------------------------
# This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
# This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.

# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                   np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], nside, lmax_qlm=lmax_qlm)

qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
# -------------------------


def get_itlib(k:str, simidx:int, version:str, cg_tol:float):
    """Return iterator instance for simulation idx and qe_key type k

        Args:
            k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            simidx: simulation index to build iterative lensing estimate on
            version: string to use to test variants of the iterator with otherwise the same parfile
                     (here if 'noMF' is in version, will not use any mean-fied at the very first step)
            cg_tol: tolerance of conjugate-gradient filter

    """
    assert k in ['p_eb', 'p_p'], k
    libdir_iterator = libdir_iterators(k, simidx, version)
    if not os.path.exists(libdir_iterator):
        os.makedirs(libdir_iterator)
    tr = int(os.environ.get('OMP_NUM_THREADS', 8))
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
    cpp[:Lmin] *= 0.

    # QE mean-field fed in as constant piece in the iteration steps:
    mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
    mf0 = qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        Nmf = len(mf_sims)
        mf0 = (mf0 - qlms_dd.get_sim_qlm(k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

    path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
    if not os.path.exists(path_plm0):
        # We now build the Wiener-filtered QE here since not done already
        plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
        plm0 -= mf0  # MF-subtracted unnormalized QE
        # Isotropic normalization of the QE
        R = qresp.get_response(k, lmax_ivf, 'p', cls_len, cls_len, {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        WF = cpp * utils.cli(cpp + utils.cli(R))
        plm0 = alm_copy(plm0,  None, lmax_qlm, mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
        almxfl(plm0, utils.cli(R), mmax_qlm, True) # Normalized QE
        almxfl(plm0, WF, mmax_qlm, True)           # Wiener-filter QE
        almxfl(plm0, cpp > 0, mmax_qlm, True)
        np.save(path_plm0, plm0)

    plm0 = np.load(path_plm0)
    R_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl_unl}, lmax_qlm=lmax_qlm)[0]
    if k in ['p_p']:
        mf_resp = qresp.get_mf_resp(k, cls_unl, {'ee': fel_unl, 'bb': fbl_unl}, lmax_ivf, lmax_qlm)[0]
    else:
        print('*** mf_resp not implemented for key ' + k, ', setting it to zero')
        mf_resp = np.zeros(lmax_qlm + 1, dtype=float)
    # Lensing deflection field instance (initiated here with zero deflection)
    ffi = remapping.deflection(lenjob_pbgeometry, lensres, np.zeros_like(plm0), mmax_qlm, tr, tr)
    if k in ['p_p', 'p_eb']:
        wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        assert np.all(transf_elm == transf_blm), 'This is not supported by the alm_filter_nlev_wl (but easy to fix)'
        # Here multipole cuts are set by the transfer function (those with 0 are not considered)
        filtr = alm_filter_nlev_wl(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                   wee=wee, transf_b=transf_blm, nlev_b=nlev_p)
        # dat maps must now be given in harmonic space in this idealized configuration
        sht_job = utils_scarf.scarfjob()
        sht_job.set_geometry(ninvjob_geometry)
        sht_job.set_triangular_alm_info(lmax_ivf,mmax_ivf)
        sht_job.set_nthreads(tr)
        datmaps = np.array(sht_job.map2alm_spin(sims_MAP.get_sim_pmap(int(simidx)), 2))
    else:
        assert 0
    k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    # Sets to zero all L-modes below Lmin in the iterations:
    iterator = scarf_iterator.iterator_pertmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
            plm0, mf_resp, R_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
            ,mf0=mf0)
    return iterator

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')


    args = parser.parse_args()
    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one

    # from lenscarf.core import mpi
    from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from lenscarf.iterators.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append(idx)

    if mpi.rank ==0:
        print("Caching things in " + TEMP)


    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, args.v, 1.)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

                itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
                itlib.soltn_cond   = soltn_cond(i)
                print("doing iter " + str(i))
                itlib.iterate(i, 'p')