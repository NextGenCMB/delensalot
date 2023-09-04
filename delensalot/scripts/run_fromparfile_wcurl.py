"""Iterative reconstruction for masked polarization CMB data

    tests joint lensing gradient and curl potential reconstruction

    e.g. python ./run_fromparfile_wcurl.py -itmax 0 -v 'wcurlin'

    '' version is standard gradient reconstruction
    'wcurl' version reconstructs both gradient and curl on gradient-only input map
    'wcurlin' version reconstructs both gradient and curl on gradient and curl input map
              (starting point still the curl-free QE though)

"""
import os
from os.path import join as opj
import numpy as np
from multiprocessing import cpu_count

import plancklens
from plancklens import utils
from plancklens import qresp
from plancklens import qest, qecl
from plancklens.qcinv import cd_solve
from plancklens.sims import phas, maps
from plancklens.sims.cmbs import sims_cmb_unl
from plancklens.filt import filt_simple, filt_util

from lenspyx.remapping.deflection import deflection
from lenspyx.remapping.utils_geom import Geom, pbdGeometry, pbounds
from lenspyx.utils import cli
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy
from lenspyx import cachers
from lenspyx.sims import sims_cmb_len

from delensalot.utility import utils_steps
from delensalot.core.iterator import steps
from delensalot.core import mpi
from delensalot.core.opfilt.MAP_opfilt_iso_p import alm_filter_nlev_wl
from delensalot.core.iterator.cs_iterator import iterator_cstmf as iterator_cstmf
from delensalot.core.iterator.cs_iterator_multi import iterator_cstmf as iterator_cstmf_wcurl

suffix = 'delensalot_idealized_curly' # descriptor to distinguish this parfile from others...
TEMP =  opj(os.environ['SCRATCH'], suffix)
DATDIR = opj(os.environ['SCRATCH'], suffix, 'sims')
DATDIRwcurl = opj(os.environ['SCRATCH'],suffix, 'simswcurl')

if not os.path.exists(DATDIR):
    os.makedirs(DATDIR)
# harmonic space noise phas down to 4096
noise_phas = phas.lib_phas(opj(os.environ['HOME'], 'noisephas_lmax%s'%4096), 3, 4096) # T, E, and B noise phases
cmb_phas = phas.lib_phas(opj(os.environ['HOME'], 'cmbphas_lmax%s'%5120), 5, 5120) # unlensed T E B P O CMB phases

lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (4096, 4096, 1., 0.5 / np.sqrt(2), 0.5)
lmin_tlm, lmin_elm, lmin_blm = (1, 2, 2) # The fiducial transfer functions are set to zero below these lmins
# for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

lmax_qlm, mmax_qlm = (5120, 5120) # Lensing map is reconstructed down to this lmax and mmax
# NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
lmax_unl, mmax_unl = (5120, 5120) # Delensed CMB is reconstructed down to this lmax and mmax


#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
lenjob_geometry = Geom.get_thingauss_geometry(lmax_unl * 2, 2)
lenjob_pbgeometry = pbdGeometry(lenjob_geometry, pbounds(0., 2 * np.pi))
Lmin = 1 # The reconstruction of all lensing multipoles below that will not be attempted
mc_sims_mf_it0 = np.array([]) # sims to use to build the very first iteration mean-field (QE mean-field) Here 0 since idealized


# Multigrid chain descriptor
# The hard coded number nside 2048 here is irrelevant for diagonal preconditioner
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, 2048, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
#------------------

# Fiducial CMB spectra for QE and iterative reconstructions
# (here we use very lightly suboptimal lensed spectra QE weights)
cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_unl_wcurl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_unl_wcurl['oo'] = np.loadtxt(opj(cls_path, 'FFP10_fieldrotationCls.dat')) # lensing curl potential

gradcls = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))

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
#       actual data transfer function for the sim generation:
transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full FFP10 cmb's which are given to 4096)
cls_noise = {'t': np.full(4097, (nlev_t /180 / 60 *  np.pi) ** 2)  * (cls_len['tt'][:4097] > 0),
             'e': np.full(4097, (nlev_p / 180 / 60 * np.pi) ** 2)  * (cls_len['ee'][:4097] > 0),
             'b': np.full(4097, (nlev_p / 180 / 60 * np.pi) ** 2)  * (cls_len['bb'][:4097] > 0),}
cls_transf = {f: transf_dat for f in ['t', 'e', 'b']}
if mpi.rank ==0:
    # Problem of creating dir in parallel if does not exist
    cacher = cachers.cacher_npy(DATDIR)
    cacher_wcurl = cachers.cacher_npy(DATDIRwcurl)
mpi.barrier()

cacher = cachers.cacher_npy(DATDIR)
cacher_wcurl = cachers.cacher_npy(DATDIRwcurl)

cmb_unl = sims_cmb_unl(cls_unl, cmb_phas)
cmb_unl_wcurl = sims_cmb_unl(cls_unl_wcurl, cmb_phas)

cmb_len = sims_cmb_len(4096, cmb_unl, cache=cacher, epsilon=1e-7)
cmb_len_wcurl = sims_cmb_len(4096, cmb_unl_wcurl, cache=cacher_wcurl, epsilon=1e-7)

sims      = maps.cmb_maps_harmonicspace(cmb_len, cls_transf, cls_noise, noise_phas)
sims_wcurl = maps.cmb_maps_harmonicspace(cmb_len_wcurl, cls_transf, cls_noise, noise_phas)
# -------------------------

ivfs         = filt_simple.library_fullsky_alms_sepTP(opj(TEMP, 'ivfs'), sims, transf_d, cls_len, ftl, fel, fbl, cache=True)
ivfs_wcurl   = filt_simple.library_fullsky_alms_sepTP(opj(TEMP, 'ivfs_wcurl'), sims_wcurl, transf_d, cls_len, ftl, fel, fbl, cache=True)

# ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
mc_sims_bias = np.arange(0, dtype=int)
mc_sims_var  = np.arange(0, 60, dtype=int)
qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], 2048, lmax_qlm=lmax_qlm)
qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)


qlms_dd_wcurl = qest.library_sepTP(opj(TEMP, 'qlms_dd_wcurl'), ivfs_wcurl, ivfs_wcurl,   cls_len['te'], 2048, lmax_qlm=lmax_qlm)
qcls_dd_wcurl = qecl.library(opj(TEMP, 'qcls_dd_wcurl'), qlms_dd_wcurl, qlms_dd_wcurl, mc_sims_bias)

# -------------------------
# This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
# This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.

# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                   np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], 2048, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], 2048, lmax_qlm=lmax_qlm)

qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations

def get_n0_iter(k='p_p'):
    from plancklens import n0s
    fnN0s = 'N0siter' + k * (k != 'p_p')
    fndelcls = 'delcls'+ k * (k != 'p_p')
    cachecond = True
    if not cacher_wcurl.is_cached(fnN0s) or not cacher_wcurl.is_cached(fndelcls):
        _, N0sg, _, N0c, _, delcls = n0s.get_N0_iter(k, nlev_t, nlev_p, beam, cls_unl, {'t':lmin_tlm, 'e':lmin_elm, 'b':lmin_blm}, lmax_ivf,10, ret_delcls=True, ret_curl=True, lmax_qlm=lmax_qlm)
        if cachecond:
            cacher_wcurl.cache(fnN0s, np.array([N0sg, N0c]))
            cacher_wcurl.cache(fndelcls, np.array([delcls[-1][spec] for spec in ['ee', 'bb', 'pp']]))
        return np.array([N0sg, N0c]), delcls
    delcls =  cacher_wcurl.load(fndelcls)
    delclsdict = {'ee': delcls[0], 'bb':delcls[1], 'pp':delcls[2]}
    return cacher_wcurl.load(fnN0s), delclsdict

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
    tr = int(os.environ.get('OMP_NUM_THREADS', cpu_count() // 2))
    print("Using %s threads"%tr)
    cpp = np.copy(cls_unl_wcurl['pp'][:lmax_qlm + 1])
    cpp[:Lmin] *= 0.
    coo = np.copy(cls_unl_wcurl['oo'][:lmax_qlm + 1])
    coo[:Lmin] *= 0.
    # QE mean-field fed in as constant piece in the iteration steps:
    if 'wcurlin' in version:
        qlms_dd_QE = qlms_dd_wcurl
        sims_MAP = sims_wcurl
    else:
        qlms_dd_QE = qlms_dd
        sims_MAP = sims

    mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
    mf0_p = qlms_dd_QE.get_sim_qlm_mf('p' + k[1:], mf_sims)  # Mean-field to subtract on the first iteration:
    mf0_o = qlms_dd_QE.get_sim_qlm_mf('x' + k[1:], mf_sims)  # Mean-field to subtract on the first iteration:

    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        Nmf = len(mf_sims)
        mf0_p = (mf0_p - qlms_dd_QE.get_sim_qlm('p' + k[1:], int(simidx)) / Nmf) * (Nmf / (Nmf - 1))
        mf0_o = (mf0_o - qlms_dd_QE.get_sim_qlm('x' + k[1:], int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

    plm0 = qlms_dd_QE.get_sim_qlm('p' + k[1:], int(simidx)) - mf0_p  # Unormalized quadratic estimate:
    olm0 = qlms_dd_QE.get_sim_qlm('x' + k[1:], int(simidx)) - mf0_o  # Unormalized quadratic estimate:

    # Isotropic normalization of the QE
    Rpp, Roo = qresp.get_response(k, lmax_ivf, 'p', cls_len, cls_len, {'e': fel, 'b': fbl, 't': ftl},
                                  lmax_qlm=lmax_qlm)[0:2]
    # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
    WF_p = cpp * utils.cli(cpp + utils.cli(Rpp))
    WF_o = coo * utils.cli(coo + utils.cli(Roo))

    plm0 = alm_copy(plm0, None, lmax_qlm, mmax_qlm)  # Just in case the QE and MAP mmax'es were not consistent
    almxfl(plm0, utils.cli(Rpp), mmax_qlm, True)  # Normalized QE
    almxfl(plm0, WF_p, mmax_qlm, True)  # Wiener-filter QE
    almxfl(plm0, cpp > 0, mmax_qlm, True)

    olm0 = alm_copy(olm0, None, lmax_qlm, mmax_qlm)  # Just in case the QE and MAP mmax'es were not consistent
    almxfl(olm0, utils.cli(Roo), mmax_qlm, True)  # Normalized QE
    almxfl(olm0, WF_o, mmax_qlm, True)  # Wiener-filter QE assuming the curl signal is the expected one
    almxfl(olm0, coo > 0, mmax_qlm, True)

    Rpp_unl, Roo_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,
                                          {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0:2]
    # Lensing deflection field instance (initiated here with zero deflection)


    if k in ['p_p']:
        ffi = deflection(lenjob_geometry, np.zeros_like(plm0), mmax_qlm, numthreads=tr, epsilon=1e-7)
        # Here multipole cuts are set by the transfer function (those with 0 are not considered)
        filtr = alm_filter_nlev_wl(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    transf_b=transf_blm, nlev_b=nlev_p)
        # dat maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        datmaps = np.array([alm_copy(eblm[0], None, lmax_ivf, mmax_ivf), alm_copy(eblm[1], None, lmax_ivf, mmax_ivf) ])
        del eblm
    else:
        assert 0


    k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    if 'wcurl' in version:
        # Sets to zero all L-modes below Lmin in the iterations:
        assert 'wmf1' not in version
        stepper = utils_steps.harmonicbump(xa=400, xb=1500)
        iterator = iterator_cstmf_wcurl(libdir_iterator, 'p', [(lmax_qlm, mmax_qlm), (lmax_qlm, mmax_qlm)], datmaps,
                                  [plm0, olm0], [mf0_p, mf0_o], [Rpp_unl, Roo_unl], [cpp, coo], ('p', 'x'), cls_unl, filtr, k_geom,
                                  chain_descrs(lmax_unl, cg_tol), stepper,
                                 wflm0=lambda : alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl))

    else: # standard gradient only
        stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, xa=400, xb=1500)  # reduce the gradient by 0.5 for large scale and by 0.1 for small scales to improve convergence in regimes where the deflection field is not invertible
        iterator = iterator_cstmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
            plm0, plm0 * 0, Rpp_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
            , wflm0=lambda : alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl))
    return iterator

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=7., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=0, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')
    parser.add_argument('-p', dest='plot', action='store_true', help='make some plots on the fly')


    args = parser.parse_args()
    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one


    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from delensalot.core.iterator.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax or args.plot:
            jobs.append(idx)

    if mpi.rank ==0:
        print("Caching things in " + TEMP)


    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, args.v)
        print("iterator folder: " + lib_dir_iterator)

        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, args.v, 1.)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
                itlib.soltn_cond   = soltn_cond(i)
                print("doing iter " + str(i))
                itlib.iterate(i, 'p')

    if args.plot and mpi.rank == 0:
        import pylab as pl
        pl.ion()
        version = args.v
        input_sims = sims
        fig, axes = pl.subplots(1, 3, figsize=(15, 5))
        for idx in jobs[0:1]: # only first
            print("plots")
            lib_dir_iterator = libdir_iterators(args.k, idx, version)
            itrs = np.unique(np.linspace(0, args.itmax + 1, 5, dtype=int)) # plotting max 5 curves
            if args.itmax not in itrs:
                itrs = np.concatenate([itrs, [args.itmax]])
            plms = Rec.load_plms(lib_dir_iterator, itrs)
            plm_in = alm_copy(sims.sims_cmb_len.get_sim_plm(int(idx)), 5120, lmax_qlm, mmax_qlm)
            cpp_in = alm2cl(plm_in, plm_in, lmax_qlm, mmax_qlm, lmax_qlm)
            ls = np.arange(1, lmax_qlm + 1)
            wls = ls ** 2 * (ls + 1) ** 2 / (2 * np.pi)
            axes[0].set_title('auto-spectra')
            axes[1].set_title('cross-spectra')
            axes[2].set_title('cross-corr. coeff.')
            axes[0].loglog(ls, wls * cpp_in[ls], c='k')
            for itr, plms in zip(itrs, plms):
                plm = np.atleast_2d(plms)[0] # In the curly case there are two components
                cxx = alm2cl(plm, plm_in, lmax_qlm, mmax_qlm, lmax_qlm)
                cpp = alm2cl(plm, plm, lmax_qlm, mmax_qlm, lmax_qlm)
                axes[0].loglog(ls, wls * cpp[ls], label='itr ' + str(itr))
                axes[1].loglog(ls, wls * cxx[ls], label='itr ' + str(itr))
                axes[2].semilogx(ls, cxx[ls] / np.sqrt(cpp * cpp_in)[ls], label='itr ' + str(itr))

            for ax in axes:
                ax.legend()
        k = input("press close to exit")