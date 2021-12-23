"""Iterative reconstruction for masked polarization CMB data



FIXME's :
    plancklens independent QEs ?
    degrade method of _wl_ filters
    check of invertibility at very first step
    mf_resp for EB-like ?
~ cgtol 5 ~ 100 it for QE with planck chain
"""
import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens

from plancklens import utils, qresp, qest, qecl
from plancklens.qcinv import cd_solve
from plancklens.sims import maps, phas, planck2018_sims
from plancklens.filt import filt_cinv, filt_util

from lenscarf import remapping, utils_scarf, utils_sims
from lenscarf.iterators import cs_iterator as scarf_iterator, steps
from lenscarf.utils import cli, read_map
from lenscarf.utils_hp import gauss_beam, almxfl
from lenscarf.opfilt import opfilt_ee_wl

suffix = 'cmbs4_planckmask' # descriptor to distinguish this parfile from others...
TEMP =  opj(os.environ['SCRATCH'], 'lenscarfrecs', suffix)

lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (3000, 3000, 1., 1., np.sqrt(2.))

# The fiducial transfer functions are set to zero below these lmins
lmin_tlm, lmin_elm, lmin_blm = (2, 2, 2)  # for delensing useful to cut much more B

lmax_qlm, mmax_qlm, lmax_unl, mmax_unl = (4000, 4000, 4000, 4000)

#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
nside = 2048
zbounds     = (-1.,1.) # colatitude sky cuts for noise variance maps (We could exclude all rings which are completely masked)
ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(nside, zbounds=zbounds)

zbounds_len = (-1.,1.) # Outside of these bounds the reconstructed maps are assumed to be zero
pb_ctr, pb_extent = (0., 2 * np.pi) ## Longitude cuts, if any, in the form (center of patch, patch extent)
lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(lmax_unl, 2, zbounds=zbounds_len)
lenjob_pbgeometry =utils_scarf.pbdGeometry(lenjob_geometry, utils_scarf.pbounds(pb_ctr, pb_extent))
lensres = 1.7  # deflection operations will be performed at this resolution
Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted
stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
mc_sims_mf_it0 = np.arange(300) # sims to use to build the very first iteration mean-field (QE mean-field)


# Multigrid chain descriptor
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
#chain_descrs = lambda lmax_sol, cg_tol :  \
#            [[2, ["split(dense(" + opj(TEMP, 'cinv_p', 'dense.pk') + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,cd_solve.cache_mem()],
#             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
#             [0, ["split(stage(1), 1024, diag_cl)"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
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
pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%nside), 3, (hp.nside2npix(nside),))
#       actual data transfer function for the sim generation:
transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # taking here full FFP10 cmb's
sims      = maps.cmb_maps_nlev(planck2018_sims.cmb_len_ffp10(), transf_dat, nlev_t, nlev_p, nside, pix_lib_phas=pix_phas)

# Makes the simulation library consistent with the zbounds
sims_MAP  = utils_sims.ztrunc_sims(sims, nside, zbounds)
# -------------------------

# List of paths to masks that will be multiplied together to give the total mask
# here we use the same in Pol and T, though that would not be necessary
masks = ['/project/projectdirs/cmb/data/planck2018/pr3/Planck_L08_inputs/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz']


# List of the inverse noise pixel variance maps, all will be multiplied together
ninv_t = [np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_t ** 2])] + masks
cinv_t = filt_cinv.cinv_t(opj(TEMP, 'cinv_t'), lmax_ivf,nside, cls_len, transf_tlm, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_p ** 2])] + masks]
cinv_p = filt_cinv.cinv_p(opj(TEMP, 'cinv_p'), lmax_ivf, nside, cls_len, transf_elm, ninv_p,
            chain_descr=chain_descrs(lmax_ivf, 1e-5), transf_blm=transf_blm, marge_qmaps=(), marge_umaps=())

ivfs_raw    = filt_cinv.library_cinv_sepTP(opj(TEMP, 'ivfs'), sims, cinv_t, cinv_p, cls_len)
ftl_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
fel_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_elm)
fbl_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_blm)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl_rs, fel_rs, fbl_rs)

# -------------------------
# This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)
# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                   np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], nside, lmax_qlm=lmax_qlm)

qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)
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

    # QE mean-field fed in as constant piece in the iteration steps:
    mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
    mf0 = qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        mf0 = (mf0 - qlms_dd.get_sim_qlm(k, int(simidx)) / len(mf_sims)) / (len(mf_sims) - 1)

    path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
    if not os.path.exists(path_plm0):
        # We now build the Wiener-filtered QE here since not done already
        plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
        plm0 -= mf0  # MF-subtracted unnormalized QE
        # Isotropic normalization of the QE
        R = qresp.get_response(k, lmax_ivf, 'p', cls_len, cls_len, {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        WF = cpp * utils.cli(cpp + utils.cli(R))

        almxfl(plm0, utils.cli(R), mmax_qlm, True) # Normalized QE
        almxfl(plm0, WF, mmax_qlm, True)           # Wiener-filter QE
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
        tpl = None # for template projection, here set to None
        wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in ninv_p] # inverse pixel noise map on consistent geometry
        filtr = opfilt_ee_wl.alm_filter_ninv_wl(ninvjob_geometry, ninv, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf), tr, tpl,
                                                wee=wee, lmin_dotop=min(lmin_elm, lmin_blm), transf_blm=transf_blm)
        datmaps = np.array(sims_MAP.get_sim_pmap(int(simidx)))

    else:
        assert 0
    k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    # Sets to zero all L-modes below Lmin in the iterations:
    cpp[:Lmin] *= 0.
    almxfl(plm0, cpp > 0, mmax_qlm, True)
    iterator = scarf_iterator.iterator_pertmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
            plm0, mf_resp, R_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
            ,mf0=mf0, wflm0=ivfs.get_sim_emliklm)
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

    from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from lenscarf.iterators.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append(idx)

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