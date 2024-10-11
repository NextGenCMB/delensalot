r"""Iterative reconstruction parameter-file script

    e.g.
        python ./scripts/run_fromparfile.py -itmax 1 -k ptt -tol 7 -p

"""
from __future__ import annotations
import os

from os.path import join as opj
import numpy as np
from multiprocessing import cpu_count


import plancklens
from plancklens import utils
from plancklens import qresp
from plancklens import qest, qecl
from plancklens.qcinv import cd_solve

from plancklens.sims import maps, phas
from plancklens.filt import filt_simple
from delensalot.core.iterator import cs_iterator as scarf_iterator_baseline, steps
from delensalot.core.iterator import cs_iterator_dev as scarf_iterator_baseline

from delensalot.core.iterator import cs_iterator_fast as scarf_iterator_fastwf

from plancklens.utils import cli
from delensalot.core.opfilt.MAP_opfilt_iso_p import alm_filter_nlev_wl as ee_filter
from delensalot.core.opfilt.MAP_opfilt_iso_tp import alm_filter_nlev_wl as gmv_filter
from delensalot.core.opfilt.MAP_opfilt_iso_t import alm_filter_nlev_wl as tt_filter
from delensalot.core.opfilt.MAP_opfilt_iso_e import alm_filter_nlev_wl as ee_nob_filter

from plancklens.helpers import cachers
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy
from plancklens.helpers import mpi
from plancklens.sims.cmbs import sims_cmb_unl
from lenspyx.remapping.deflection_029 import deflection
from lenspyx.remapping import utils_geom
from lenspyx.sims import sims_cmb_len
from lenspyx.utils import Drop

suffix = 'cmbs4pub_delensing_lenspyxed' # descriptor to distinguish this parfile from others...
TEMP = opj(os.environ['SCRATCH'], 'lenscarfrecs', suffix)
DATDIR = opj(os.environ['SCRATCH'],'lenspyxedFFP10cls_noaberration')
libdir_n1_dd = os.path.join(TEMP, 'n1_ffp10_l4096_L5120')

if not os.path.exists(DATDIR):
    os.makedirs(DATDIR)
# harmonic space noise phas down to 4096
noise_phas = phas.lib_phas(opj(os.environ['HOME'], 'noisephas_lmax%s'%4096), 3, 4096) # T, E, and B noise phases
cmb_phas = phas.lib_phas(opj(os.environ['HOME'], 'cmbphas_lmax%s'%5120), 4, 5120) # unlensed CMB phases

lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (4096, 4096, 1., 0.5 / np.sqrt(2), 0.5)
lmin_tlm, lmin_elm, lmin_blm = (1, 2, 200) # The fiducial transfer functions are set to zero below these lmins
# for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

lmax_qlm, mmax_qlm = (5120, 5120) # Lensing map is reconstructed down to this lmax and mmax
# NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
lmax_unl, mmax_unl = (5120, 5120) # Delensed CMB is reconstructed down to this lmax and mmax


#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
lenjob_geometry = utils_geom.Geom.get_thingauss_geometry(lmax_unl + 1024, 2)
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
gradcls = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))

# Fiducial model of the transfer function
transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
transf_d = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}
# Isotropic approximation to the filtering (used eg for response calculations)
ftl = cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel = cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl = cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
ftl_unl = cli(cls_unl['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel_unl = cli(cls_unl['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl_unl = cli(cls_unl['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# -------------------------
# ---- Input simulation libraries. Here we use the NERSC FFP10 CMBs with homogeneous noise and consistent transfer function
#       We define explictly the phase library such that we can use the same phases for for other purposes in the future as well if needed
#       I am putting here the phases in the home directory such that they dont get NERSC auto-purged
#       actual data transfer function for the sim generation:
transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full FFP10 cmb's which are given to 4096)
cls_noise = {'t': np.full(4097, (nlev_t / 180 / 60 * np.pi) ** 2) * (cls_len['tt'][:4097] > 0),
             'e': np.full(4097, (nlev_p / 180 / 60 * np.pi) ** 2) * (cls_len['ee'][:4097] > 0),
             'b': np.full(4097, (nlev_p / 180 / 60 * np.pi) ** 2) * (cls_len['bb'][:4097] > 0), }
cls_transf = {f: transf_dat for f in ['t', 'e', 'b']}
if mpi.rank ==0:
    # Problem of creating dir in parallel if does not exist
    cacher = cachers.cacher_npy(DATDIR)
mpi.barrier()
cacher = cachers.cacher_npy(DATDIR)


cmb_unl = sims_cmb_unl(cls_unl, cmb_phas)
cmb_len = sims_cmb_len(4096, cmb_unl, cache=cacher, epsilon=1e-7)

# ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)


def get_sims_ivfs_qlms(facnoise=1.):
    if facnoise == 1:
        sims = maps.cmb_maps_harmonicspace(cmb_len, cls_transf, cls_noise, noise_phas)
        ivfs   = filt_simple.library_fullsky_alms_sepTP(opj(TEMP, 'ivfs'), sims, transf_d, gradcls, ftl, fel, fbl, cache=True)
        qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs, gradcls['te'], 2048, lmax_qlm=lmax_qlm)
        qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)
        return sims, ivfs, qlms_dd, qcls_dd
    assert facnoise == 0
    _cls_noise = {k : cls_noise[k] * 0. for k in cls_noise.keys()}
    sims = maps.cmb_maps_harmonicspace(cmb_len, cls_transf, _cls_noise, noise_phas)
    ivfs = filt_simple.library_fullsky_alms_sepTP(opj(TEMP, 'ivfs_nfree'), sims, transf_d, gradcls, ftl, fel, fbl, cache=True)
    qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd_nfree'), ivfs, ivfs, gradcls['te'], 2048, lmax_qlm=lmax_qlm)
    qcls_dd = qecl.library(opj(TEMP, 'qcls_dd_nfree'), qlms_dd, qlms_dd, mc_sims_bias)
    return sims, ivfs, qlms_dd, qcls_dd



# -------------------------
def get_itlib(k:str, simidx:int, version:str, cg_tol:float, epsilon=1e-5, nbump=0, rscal=0, verbose=False, numthreads=0):
    """Return iterator instance for simulation idx and qe_key type k
        Args:
            k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            simidx: simulation index to build iterative lensing estimate on
            version: string to use to test variants of the iterator with otherwise the same parfile
                     (here if 'noMF' is in version, will not use any mean-fied at the very first step)
            cg_tol: tolerance of conjugate-gradient filter
    """
    libdir_iterator = libdir_iterators(k, simidx, version)
    if not os.path.exists(libdir_iterator):
        os.makedirs(libdir_iterator)
    if numthreads <= 0:
        numthreads = int(os.environ.get('OMP_NUM_THREADS', cpu_count()))
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
    cpp[:Lmin] *= 0.
    sims, ivfs, qlms_dd, qcls_dd = get_sims_ivfs_qlms(facnoise=0. if 'noisefree' in version else 1.)
    qlms_dd_QE = qlms_dd
    sims_MAP = sims

    mf_sims = np.unique(mc_sims_mf_it0)
    mf0_p = qlms_dd_QE.get_sim_qlm_mf('p' + k[1:], mf_sims)  # Mean-field to subtract on the first iteration:

    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        Nmf = len(mf_sims)
        mf0_p = (mf0_p - qlms_dd_QE.get_sim_qlm('p' + k[1:], int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

    plm0 = qlms_dd_QE.get_sim_qlm('p' + k[1:], int(simidx)) - mf0_p  # Unormalized quadratic estimate:

    # Isotropic normalization of the QE
    Rpp, Roo = qresp.get_response(k, lmax_ivf, 'p', gradcls, gradcls, {'e': fel, 'b': fbl, 't': ftl},
                                  lmax_qlm=lmax_qlm)[0:2]
    # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
    WF_p = cpp * utils.cli(cpp + utils.cli(Rpp))

    plm0 = alm_copy(plm0, None, lmax_qlm, mmax_qlm)  # Just in case the QE and MAP mmax'es were not consistent
    almxfl(plm0, utils.cli(Rpp), mmax_qlm, True)  # Normalized QE
    almxfl(plm0, WF_p, mmax_qlm, True)  # Wiener-filter QE
    almxfl(plm0, cpp > 0, mmax_qlm, True)

    Rpp_unl, Roo_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,
                                          {'e': fel_unl, 'b': fbl_unl, 't': ftl_unl}, lmax_qlm=lmax_qlm)[0:2]
    # Lensing deflection field instance (initiated here with zero deflection)
    ffi = deflection(lenjob_geometry, np.zeros_like(plm0), mmax_qlm,
                     numthreads=numthreads, verbosity=0, epsilon=epsilon)
    if nbump > 0:
        print("Setting up inverse noise drop of Dl = %s"%nbump)
        drop = Drop(a=lmax_ivf - nbump, b=lmax_ivf + 1).eval(np.arange(lmax_ivf + 1))
    else:
        drop = np.ones(lmax_ivf + 1, dtype=float)
    _filtr_ee, _filtr_tt = None, None
    if k in ['p_p']:

        # Here multipole cuts are set by the transfer function (those with 0 are not considered)
        filtr = ee_filter(nlev_p / drop, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    transf_b=transf_blm, nlev_b=nlev_p)
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        datmaps = np.array([alm_copy(eblm[0], None, lmax_ivf, mmax_ivf), alm_copy(eblm[1], None, lmax_ivf, mmax_ivf) ])
        del eblm
        wflm0 = lambda: alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl)
        if ffi.single_prec:
            datmaps = datmaps.astype(np.complex64)

    elif k in ['p_eb']:
        filtr = ee_filter(nlev_p / drop, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    transf_b=transf_blm, nlev_b=nlev_p)
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        datmaps = np.array([alm_copy(eblm[0], None, lmax_ivf, mmax_ivf), alm_copy(eblm[1], None, lmax_ivf, mmax_ivf) ])
        del eblm
        wflm0 = lambda: alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl)
        if ffi.single_prec:
            datmaps = datmaps.astype(np.complex64)
        _filtr_ee = ee_nob_filter(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
        # data maps must now be given in harmonic space in this idealized configuration
    elif k in ['p']:
        filtr = gmv_filter(nlev_t / drop, nlev_p / drop, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    transf_e=transf_elm, transf_b=transf_blm, nlev_b=nlev_p)
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        tlm = sims_MAP.get_sim_tmap(int(simidx))
        datmaps = np.array([alm_copy(tlm, None, lmax_ivf, mmax_ivf),
                            alm_copy(eblm[0], None, lmax_ivf, mmax_ivf),
                            alm_copy(eblm[1], None, lmax_ivf, mmax_ivf) ])
        del eblm, tlm
        wflm0 = lambda: np.array([alm_copy(ivfs.get_sim_tmliklm(simidx), None, lmax_unl, mmax_unl),
                                  alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl)])
    elif k in ['pmtt']:
        filtr = gmv_filter(nlev_t / drop, nlev_p / drop, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf),
                                    transf_e=transf_elm, transf_b=transf_blm, nlev_b=nlev_p)
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        tlm = sims_MAP.get_sim_tmap(int(simidx))
        datmaps = np.array([alm_copy(tlm, None, lmax_ivf, mmax_ivf),
                            alm_copy(eblm[0], None, lmax_ivf, mmax_ivf),
                            alm_copy(eblm[1], None, lmax_ivf, mmax_ivf) ])
        del eblm, tlm
        wflm0 = lambda: np.array([alm_copy(ivfs.get_sim_tmliklm(simidx), None, lmax_unl, mmax_unl),
                                  alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl)])
        _filtr_tt = tt_filter(nlev_t /drop, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
    elif k in ['ptt']:
        if rscal != 0:
            print("WF will be that of (L(L + 1))^(%s/2) T"%rscal)
            rescal = np.sqrt(np.arange(lmax_unl + 1) * np.arange(1, lmax_unl + 2, dtype=float)) ** abs(rscal)
            if rscal < 0:
                rescal = cli(rescal)
            wflm0 = lambda: almxfl(alm_copy(ivfs.get_sim_tmliklm(simidx), None, lmax_unl, mmax_unl), rescal, mmax_unl, False)
        else:
            rescal = None
            wflm0 = lambda: alm_copy(ivfs.get_sim_tmliklm(simidx), None, lmax_unl, mmax_unl)
        filtr = tt_filter(nlev_t / drop, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf), rescal=rescal)
        datmaps = sims_MAP.get_sim_tmap(int(simidx))
    elif k in ['pee']:
        filtr = ee_nob_filter(nlev_p, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        datmaps = alm_copy(eblm[0], None, lmax_ivf, mmax_ivf)
        del eblm
        wflm0 = lambda: alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl)
    elif k in ['pbb']:
        from delensalot.core.opfilt.opfilt_iso_b_wl import alm_filter_nlev_wl as bb_filter
        filtr = bb_filter(nlev_p, ffi, transf_blm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
        # data maps must now be given in harmonic space in this idealized configuration
        eblm = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        datmaps = alm_copy(eblm[1], None, lmax_ivf, mmax_ivf)
        del eblm
        wflm0 = None
    else:
        assert 0

    if verbose:
        filtr.verbose = True

    if ffi.single_prec:
        print("Sending in single precision data")
        datmaps = datmaps.astype(np.complex64)
        wflm0_ = lambda : wflm0().astype(np.complex64)
    else:
        wflm0_ = wflm0
        datmaps = datmaps.astype(complex if np.iscomplexobj(datmaps) else float, copy=False)

    filtr0 = filtr[0] if isinstance(filtr, list) else filtr
    k_geom = filtr0.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    # standard gradient only
    if 'hbump' in version:
        stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, xa=400, xb=1500)  # reduce the gradient by 0.5 for large scale and by 0.1 for small scales to improve convergence in regimes where the deflection field is not invertible
    elif 'stepval01' in version:
        stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, a=0.1, b=0.0999999, xa=400, xb=1500)  # no harmonic bump
    else:
        stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, a=0.5, b=0.4999, xa=400, xb=1500)  # no harmonic bump

    if 'fastwf' in version.lower():
        iterator = scarf_iterator_fastwf.iterator_cstmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
        plm0, plm0 * 0, Rpp_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
        ,wflm0=wflm0_)
    else:
        iterator = scarf_iterator.iterator_cstmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
        plm0, plm0 * 0, Rpp_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
        ,wflm0=wflm0_, _eefilter=_filtr_ee, _ttfilter=_filtr_tt)
    return iterator

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=0, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=0, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')
    parser.add_argument('-p', dest='plot', action='store_true', help='make some plots on the fly')
    parser.add_argument('-sol', dest='sol', type=str, default='', help='path to starting point for Wiener filtering (optional)')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7., help='-log10 of lensing accuracy')
    parser.add_argument('-nbump', dest='nbump', type=int, default=0, help='inflate the noise by a lot ''nbump'' multipoles before lmax transf')
    parser.add_argument('-r', dest='rescal', type=float, default=0., help='rescal temperature maps WF')
    parser.add_argument('-nt', dest='nthreads', type=int, default=0, help='number of threads to use (defaults to OMP_NUM_THREADS)')
    parser.add_argument('-dev', dest='dev', action='store_true', help='uses dev-version iterator')

    args = parser.parse_args()
    if args.dev:
        scarf_iterator = scarf_iterator_dev
        args.v += '_dev'
    else:
        scarf_iterator = scarf_iterator_baseline

    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one


    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    from delensalot.core.iterator.statics import rec as Rec
    version = args.v + ('tol%.1f'%args.tol) * (args.tol != 5.) + ('eps%.1f'%args.epsilon) * (args.epsilon != 5.)
    version += ('ndrop%s'%args.nbump) * (args.nbump != 0)
    version += ('rscal%s'%str(args.rescal)) * (args.rescal != 0)
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = libdir_iterators(args.k, idx, version)
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax or (args.plot and idx == 0):
            jobs.append(idx)

    if mpi.rank == 0:
        print("Caching things in " + TEMP)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(args.k, idx, version)
        print("iterator folder: " + lib_dir_iterator)

        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.k, idx, version, 1., epsilon=10 ** (- args.epsilon), nbump=args.nbump, rscal=args.rescal, numthreads=args.nthreads)
            if os.path.exists(args.sol):
                print('Setting WF starting point ' + args.sol)
                itlib.wflm0 = lambda: np.load(args.sol)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

                itlib.chain_descr = chain_descrs(lmax_unl, tol_iter(i))
                itlib.soltn_cond = soltn_cond(i)
                print("doing iter " + str(i))
                itlib.iterate(i, 'p')

    mpi.barrier()
    if args.plot and mpi.rank == 0:
        import pylab as pl
        pl.ion()
        sims, ivfs, qlms_dd, qcls_dd = get_sims_ivfs_qlms(facnoise=0. if 'noisefree' in version else 1.)
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
            for itr, plm in zip(itrs, plms):
                cxx = alm2cl(plm, plm_in, lmax_qlm, mmax_qlm, lmax_qlm)
                cpp = alm2cl(plm, plm, lmax_qlm, mmax_qlm, lmax_qlm)
                axes[0].loglog(ls, wls * cpp[ls], label='itr ' + str(itr))
                axes[1].loglog(ls, wls * cxx[ls], label='itr ' + str(itr))
                axes[2].semilogy(ls, cxx[ls] / np.sqrt(cpp * cpp_in)[ls], label='itr ' + str(itr))

            for ax in axes:
                ax.legend()
            k = input("press close to exit")
