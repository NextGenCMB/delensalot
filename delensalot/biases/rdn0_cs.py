"""This module calculates and caches generalized QE estimates for the MAP solution RDN0

Example for how to use:

python rdn0_cs.py -par cmbs4wide_idealized -k p_p -datidx 0


"""

# try:
    # import scarf
    # from delensalot.core.helper.utils_scarf import pbdGeometry, pbounds, scarfjob
# except:
#     print("Couldn't find scarf installation")

from lenspyx.remapping.utils_geom import Geom, pbdGeometry, pbounds

import numpy as np, os
import delensalot
from delensalot.core.iterator import cs_iterator
from delensalot.core import cachers

from plancklens.qcinv import multigrid
from plancklens.utils import stats, cli, mchash
from plancklens.sims import planck2018_sims
from os.path import join as opj
from plancklens.helpers import mpi
from plancklens.filt import filt_cinv, filt_util, filt_simple
from plancklens import utils, qest, qecl
from plancklens.sims import maps
from delensalot.sims import sims_ffp10
from delensalot.utility import utils_sims, utils_hp
import healpy as hp


output_dir = opj(os.path.dirname(os.path.dirname(delensalot.__file__)), 'outputs')
output_sim = lambda qe_key, suffix, version, datidx: opj(output_dir, suffix, version, '{}_sim_{:04d}'.format(qe_key, datidx))

fdir_dsss = lambda itr, rdn0tol : f'ds_ss_it{itr}' + f'_tol{rdn0tol}' * (rdn0tol !=5)
fn_cls_dsss = lambda itr, mcs, Nroll, rdn0tol : f'cls_dsss_it{itr}_Nroll{Nroll}'+ f'_rdn0tol{rdn0tol}' * (rdn0tol !=5) + f'_{mchash(mcs)}.dat'

fdir_mcn0 = lambda qe_key, suffix: opj(output_dir, suffix, f'{qe_key}_mcn0')
fdir_mcn1 = lambda qe_key, suffix: opj(output_dir, suffix, f'{qe_key}_mcn1')

fncl_ss = lambda mcs, Nroll : f'cls_ss_Nroll{Nroll}_{mchash(mcs)}.dat'

_ss_dict = lambda mcs, Nroll : {idx: int(Nroll * (idx // Nroll) + (idx + 1) % Nroll) for idx in mcs}

def export_dsss(itr:int, qe_key:str, libdir:str, suffix:str, datidx:int, version:str=None, ss_dict:dict=None, mcs=None, Nroll=None, rdn0tol=5.):
    if ss_dict is None:
        Nroll = 8
        Nsims = 96
        mcs = np.arange(0, Nsims)
        ss_dict =  _ss_dict(mcs, Nroll)
    itdir = opj(libdir, fdir_dsss(itr, rdn0tol))
    print('RDN0 libdir: ' + libdir)
    ds, ss = load_ss_ds(np.arange(len(ss_dict)), ss_dict, itdir, docov=True)
    print(ds.N, ss.N, ds.size)
    lmax = ds.size - 1
    pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)
    arr = np.array([ds.mean() * pp2kki, ss.mean() * pp2kki, ds.sigmas_on_mean()*  pp2kki,  ss.sigmas_on_mean()* pp2kki, np.ones_like(ds.mean()) * ds.N, np.ones_like(ss.mean()) * ss.N])
    fmt = ['%.7e'] * 4 + ['%3i'] * 2
    header = '1e7 kk2pp times  : ds    ss   ds_erroronmean, ss_erroronmean , number of ds sims, number of ss sims'
    header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^2 * 1e7 times this   (ds ss is response-like)'
    fn_dir = output_sim(qe_key, suffix, version, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    print(opj(fn_dir, fn_cls_dsss(itr, mcs, Nroll, rdn0tol)))   
    # fn_cldsss = 'cls_dsss_itr{}.dat'
    np.savetxt(opj(fn_dir, fn_cls_dsss(itr, mcs, Nroll, rdn0tol)), arr.transpose(), fmt=fmt, header=header)


def load_ss_ds(mcs, ss_dict, folder:str, docov=False):
    """Loads precomputed ds and ss stats.

    """
    print(ss_dict)
    fn_ds = lambda this_idx : folder + '/qcl_ds_%04d'%this_idx + '.dat'
    fn_ss = lambda i, j: folder + '/qcl_ss_%04d_%04d' % (min(i, j), max(i, j)) + '.dat'
    ds0 = np.loadtxt(fn_ds(mcs[0]))
    ds = stats(ds0.size, docov=docov)
    ds.add(ds0)
    for idx in mcs[1:]:
        if os.path.exists(fn_ds(idx)):
            ds.add(np.loadtxt(fn_ds(idx)))
    ss0 = np.loadtxt(fn_ss(mcs[0], ss_dict[mcs[0]]))
    ss = stats(ss0.size, docov=docov)
    ss.add(ss0)
    for idx in mcs[1:]:
        if os.path.exists(fn_ss(idx, ss_dict[idx])):
            ss.add(np.loadtxt(fn_ss(idx, ss_dict[idx])))
            print(fn_ss(idx, ss_dict[idx]))
    assert len(mcs) == ds.N and len(mcs) == ss.N + 1 , 'asked for %s sims but ds ss found %s and %s sims'%(len(mcs), ds.N, ss.N)
    print('ds ss found %s and %s sims'%(ds.N, ss.N))

    return ds, ss


def export_nhl(libdir:str, qe_key, parfile, datidx:int, recache=False):
    fn_dir = output_sim(qe_key, parfile.suffix, parfile.version, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    fn = os.path.join(fn_dir, 'QE_knhl.dat')
    if not os.path.exists(fn) or recache:
        print(fn)
        GG = semi_rdn0_qe(libdir, parfile, datidx)
        lmax = len(GG) - 1
        pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)
        header = 'semi-analytical kappa unnormalized rdn0'
        header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this   (ds ss is response-like)'
        np.savetxt(fn, GG * pp2kki, header=header)

def semi_rdn0_qe(libdir, parfile, datidx, qe_key='p_p'):
    # Semi-analytical RD-N0:
    from delensalot.utility.utils_hp import almxfl, alm2cl
    from plancklens.nhl import get_nhl

    lmax, mmax = parfile.lmax_ivf, parfile.mmax_ivf

    if qe_key == 'ptt':
        tlm_bar = utils_hp.alm_copy((hp.map2alm(parfile.sims_MAP.get_sim_tmap(datidx))), -1, lmax, mmax)
        almxfl(tlm_bar, cli(parfile.transf_tlm) * parfile.ftl, mmax, inplace=True)
        cls_ivfs = {'tt': alm2cl(tlm_bar, tlm_bar, lmax, mmax, lmax)}
    elif qe_key == 'p_p':
        eblm_bar = hp.map2alm_spin(parfile.sims_MAP.get_sim_pmap(datidx), 2)
        eblm_bar = np.array([utils_hp.alm_copy(x, -1, lmax, mmax) for x in eblm_bar])
        
        almxfl(eblm_bar[0], cli(parfile.transf_elm) * parfile.fel, mmax, inplace=True)
        almxfl(eblm_bar[1], cli(parfile.transf_blm) * parfile.fbl, mmax, inplace=True)
        cls_ivfs = {'ee': alm2cl(eblm_bar[0], eblm_bar[0], lmax, mmax, lmax),
                    'bb': alm2cl(eblm_bar[1], eblm_bar[1], lmax, mmax, lmax)}
        
    GG, CC, GC, CG = get_nhl(qe_key, qe_key, parfile.cls_weights, cls_ivfs, lmax, lmax, lmax_out=parfile.lmax_qlm)
    return GG

def get_mcn0_qe(param, qe_key, Ndatasims=40, Nmcsims=100, Nroll=10, version='', recache=False, ivfs=None, use_parfile=False):
    """
    Returns unormalised QE MC-N0 
        Args:
            param: parameter file instance
            qe_key: QE key
            Ndatasims: Total number of sims used as data (avoid overlap with sims used for MC)
            Nmcsims: Total number of sims used for MC corrections and RD estimate
            Nroll: MC sims are shuffled i -> i+1 by batch of Nroll
            version: Vesion of the estimator/ pipeline used
            use_parfile: Use the qlms_ss defined in the param file
    """ 
    mcs = np.arange(Ndatasims, Nmcsims+Ndatasims)
    # mcs_sims_less = np.delete(mcs, np.where(mcs==datidx))
    ss_dict = _ss_dict(mcs, Nroll)

    fn_dir = fdir_mcn0(qe_key, param.suffix)
    if not os.path.exists(fn_dir) and mpi.rank == 0:
        os.makedirs(fn_dir)
    mpi.barrier()
    fn = os.path.join(fn_dir, fncl_ss(mcs, Nroll))
    if use_parfile:
        fn = os.path.join(fn_dir, 'parfile qlms ss')
    
    if ivfs is None:
        ivfs = param.ivfs

    if not os.path.exists(fn) or recache:
        print(fn)
        ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)
        if use_parfile:
            qcls_ss = param.qcls_ss
        else:
            libdir = opj(param.TEMP, 'RD_sims_{}_{}_{}'.format(Ndatasims, Nmcsims, Nroll)) + version
            qlms_ss = qest.library_sepTP(opj(libdir, 'qlms_ss'), ivfs, ivfs_s, param.cls_len['te'], param.nside, lmax_qlm=param.lmax_qlm)
            qcls_ss = qecl.library(opj(libdir, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
        print('Computing Cl_ss')
        ss = qcls_ss.get_sim_stats_qcl(qe_key, mcs, k2=qe_key).mean()
        mcn0 = 2*ss
        lmax = len(mcn0) - 1
        pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)

        arr = np.array(mcn0*pp2kki)
        header = 'QE kappa unnormalized mcn0 (2*ss)'
        header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this   (ds ss is response-like)'
        np.savetxt(fn, arr.T, header=header)
    mcn0 = np.loadtxt(fn).T
    return mcn0

def get_rdn0_qe(param, datidx, qe_key, Ndatasims=40, Nmcsims=100, Nroll=10, version='', recache=False, ivfs=None):
    """Returns unormalised realization-dependent N0 lensing bias RDN0.

        Args:
            param: parameter file instance
            datidx: index of simulation to use as data
            qe_key: QE key
            Ndatasims: Total number of sims used as data (avoid overlap with sims used for MC)
            Nmcsims: Total number of sims used for MC corrections and RD estimate
            Nroll: MC sims are shuffled i -> i+1 by batch of Nroll
            version: Vesion of the estimator/ pipeline used

        Returns:
            rdn0: Unormalised RDN0
            ds: mean of QE with one leg the data and another leg the simulations
            ss: mean od QE with each leg a different simulation
    """
    mcs = np.arange(Ndatasims, Nmcsims+Ndatasims)
    # mcs_sims_less = np.delete(mcs, np.where(mcs==datidx))
    assert datidx not in mcs, "Conflict with the index used as data and indices of sims used for RD corrections"
    ss_dict = _ss_dict(mcs, Nroll)

    ds_dict = { k : datidx for k in np.arange(Ndatasims, Nmcsims + Ndatasims)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg
    assert ss_dict.keys() == ds_dict.keys(), "Make sure that all MC sim indices are affected to the data index"
    fn_dir = output_sim(qe_key, param.suffix, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)

    if ivfs is None:
        ivfs = param.ivfs

    ss = get_mcn0_qe(param, qe_key, Ndatasims=Ndatasims, Nmcsims=Nmcsims, Nroll=Nroll, version=version, recache=recache, ivfs=ivfs)
    lmax = len(ss)-1
    pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
    ss *= 1./2. * pp2kk

    fn = os.path.join(fn_dir, fn_cls_dsss(0, mcs, Nroll, 5.))
    if not os.path.exists(fn) or recache:
        print(fn)
        if datidx == -1:
            print("Using parameter file qlms_ds")
            qcls_ds = param.qcls_ds
        else:
            ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
            # ivfs_s = filt_util.library_shuffle(param.ivfs, ss_dict)
            libdir = opj(param.TEMP, 'RD_sims_{}_{}_{}'.format(Ndatasims, Nmcsims, Nroll)) + version
            qlms_ds = qest.library_sepTP(opj(libdir, 'qlms_ds' + '_{:04d}'.format(datidx)  * (datidx != -1) ), ivfs, ivfs_d, param.cls_len['te'], param.nside, lmax_qlm=param.lmax_qlm)
            # qlms_ss = qest.library_sepTP(opj(libdir, 'qlms_ss'), param.ivfs, ivfs_s, param.cls_len['te'], param.nside, lmax_qlm=param.lmax_qlm)

            qcls_ds = qecl.library(opj(libdir, 'qcls_ds' + '_{:04d}'.format(datidx) * (datidx != -1)), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
            # qcls_ss = qecl.library(opj(libdir, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations

        print('Computing Cl_ds')
        ds = qcls_ds.get_sim_stats_qcl(qe_key, mcs, k2=qe_key).mean()
        # print('Computing Cl_ss')
        # ss = qcls_ss.get_sim_stats_qcl(qe_key, mcs, k2=qe_key).mean()
        rdn0 = 4*ds - 2*ss
        lmax = len(rdn0) - 1
        pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)

        arr = np.array([rdn0*pp2kki, ds*pp2kki, ss*pp2kki])
        header = 'QE kappa unnormalized rdn0 (4ds - 2ss), ds, ss'
        header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this   (ds ss is response-like)'
        np.savetxt(fn, arr.T, header=header)
    
    rdn0, ds, ss = np.loadtxt(fn).T
    return rdn0, ds, ss


def get_mcn1_qe(param, qe_key,  Ndatasims=40, Nmcsims=100, Nroll=10, version=''):
    """Compute mc-N1, using pairs of sims with the same lensing potential phi"""

    mcs = np.arange(Ndatasims, Nmcsims+Ndatasims)
    # mcs_sims_less = np.delete(mcs, np.where(mcs==datidx))
    # assert datidx not in mcs, "Conflict with the index used as data and indices of sims used for RD corrections"
    dlm_dict = _ss_dict(mcs, Nroll)
    
    fn_dir = fdir_mcn1(qe_key, param.suffix)
    if not os.path.exists(fn_dir) and mpi.rank == 0:
        os.makedirs(fn_dir)
    mpi.barrier()
    fn = os.path.join(fn_dir, fncl_ss(mcs, Nroll))

    if not os.path.exists(fn):
        print(fn)
        libdir = param.DATDIR + f'_dlmshift_{Nroll}_{mchash(mcs)}'
        sims_shift      = maps.cmb_maps_nlev(sims_ffp10.cmb_len_ffp10_shuffle_dlm(dlm_dict, aberration=(0,0,0), cacher=cachers.cacher_npy(libdir), lmax_thingauss=2 * 4096, nbands=7, verbose=True), param.transf_dat, param.nlev_t, param.nlev_p, param.nside, pix_lib_phas=param.pix_phas)

        # Makes the simulation library consistent with the zbounds
        # sims_MAP  = utils_sims.ztrunc_sims(sims_shift, param.nside, [param.zbounds])

        if type(param.ivfs) == filt_util.library_ftl:
            ivfs_raw    = filt_cinv.library_cinv_sepTP(opj(param.TEMP, 'ivfs_mcn1'), sims_shift, param.cinv_t, param.cinv_p, param.cls_len)
            ftl_rs = np.ones(param.lmax_ivf + 1, dtype=float) * (np.arange(param.lmax_ivf + 1) >= param.lmin_tlm)
            fel_rs = np.ones(param.lmax_ivf + 1, dtype=float) * (np.arange(param.lmax_ivf + 1) >= param.lmin_elm)
            fbl_rs = np.ones(param.lmax_ivf + 1, dtype=float) * (np.arange(param.lmax_ivf + 1) >= param.lmin_blm)
            ivfs_len   = filt_util.library_ftl(ivfs_raw, param.lmax_ivf, ftl_rs, fel_rs, fbl_rs)

        elif type(param.ivfs) == filt_simple.library_fullsky_sepTP:
            print('full sky')
            ivfs_len   = filt_simple.library_fullsky_sepTP(opj(param.TEMP, 'ivfs_mcn1'), sims_shift, param.nside, param.transf_d, param.cls_len, param.ftl, param.fel, param.fbl, cache=True)

        ivfs_s = filt_util.library_shuffle(param.ivfs, dlm_dict)

        libdir = opj(param.TEMP, 'MCN1_sims_{}_{}_{}'.format(Ndatasims, Nmcsims, Nroll)) + version
        print(libdir)
        qlms_ss = qest.library_sepTP(opj(libdir, 'qlms_ss'), ivfs_s, ivfs_len, param.cls_len['te'], param.nside, lmax_qlm=param.lmax_qlm)
        qcls_ss = qecl.library(opj(libdir, 'qcls_ss'), qlms_ss, qlms_ss, np.array([])) 

        print('Computing Cl_ss')
        for idx in mcs[mpi.rank::mpi.size]:
            print(f'Running sim idx {idx}')
            qcls_ss.get_sim_qcl(qe_key, int(idx), k2=qe_key)
        
        mpi.barrier()
        if mpi.rank == 0:
            ss = qcls_ss.get_sim_stats_qcl(qe_key, mcs, k2=qe_key).mean()

            lmax = len(ss) - 1
            pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)

            arr = np.array([ss*pp2kki])
            header = 'QE kappa unnormalized mcn1'
            header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this   (ss is response-like)'
            np.savetxt(fn, arr.T, header=header)
        mpi.barrier()
        
    ss_n1 = np.loadtxt(fn).T
    return ss_n1


def export_cls(libdir:str, qe_key:str, itr:int,suffix:str, datidx:int):
    typ = 'QE' if itr == 0 else 'MAP_itr{}'.format(itr)
    plm = get_plm(itr, libdir)
    lmax = utils_hp.Alm.getlmax(plm.size, None)
    mmax = lmax
    plm_in = utils_hp.alm_copy(planck2018_sims.cmb_unl_ffp10.get_sim_plm(datidx), None, lmax, mmax)
    cl_in = utils_hp.alm2cl(plm_in, plm_in, None, None, None)
    cl_x = utils_hp.alm2cl(plm_in, plm, None, None, None)
    cl_auto = utils_hp.alm2cl(plm, plm, None, None, None)
    pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
    header ='1e7 kappa %s auto, in auto and %sxin for '%(typ, typ) + suffix
    fn_dir = output_sim(qe_key, suffix, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    np.savetxt(fn_dir + '/%s_cls.dat'%typ, np.array([cl_auto * pp2kk, cl_in * pp2kk, cl_x * pp2kk]).transpose(), header=header)
    print("Cached " + fn_dir + '/%s_cls.dat'%typ)

def get_plm(itr:int, itdir:str):
    plm = np.load(itdir + '/phi_plm_it000.npy')
    for i in range(itr):
        plm += np.load(itdir + '/hessian/rlm_sn_%s_p.npy'%(str(i)))
    return plm

def ss_ds_cross(mcs:np.ndarray, folder:str, itlibA:cs_iterator.qlm_iterator, itlibB:cs_iterator.qlm_iterator, ss_dict:dict):
    """Same as ss_ds but for cross-spectra. Here all maps must have been previously calculated


    """
    lmax_qlmA, mmax_qlmA = itlibA.mmax_qlm, itlibA.lmax_qlm
    lmax_qlmB, mmax_qlmB = itlibB.mmax_qlm, itlibB.lmax_qlm
    assert (lmax_qlmA, mmax_qlmA) == (lmax_qlmB, mmax_qlmB)
    lmax_qlm, mmax_qlm = lmax_qlmA, mmax_qlmA

    if mpi.rank == 0 and not os.path.exists(folder):
        os.makedirs(folder)
    mpi.barrier()
    assert os.path.exists(itlibA.lib_dir + '/ds_ss')
    assert os.path.exists(itlibB.lib_dir + '/ds_ss')

    ivfA_cacher = cachers.cacher_npy(itlibA.lib_dir + '/ds_ss')
    ivfB_cacher = cachers.cacher_npy(itlibB.lib_dir + '/ds_ss')

    fn_ds = lambda this_idx : folder + '/qcl_ds_%04d'%this_idx
    fn_wf = lambda this_idx : 'ebwf_filtersim_%04d'%this_idx # Wiener-filtered sim
    fn = lambda this_idx : 'eb_filtersim_%04d'%this_idx # full sims

    for i in np.unique(mcs)[mpi.rank::mpi.size]:
        q_geom = pbdGeometry(itlibA.k_geom, pbounds(0., 2 * np.pi)) # We assume they are consistent between A & B
        idx = int(i)
        if not os.path.exists(fn_ds(idx)+ '.dat'):
            ebwf_dat =  ivfA_cacher.load('ebwf_dat')
            qlmA  = 0.5 * itlibA.filter.get_qlms( itlibA.dat_maps, ebwf_dat,  q_geom, alm_wf_leg2=ivfA_cacher.load(fn_wf(idx)))[0]
            qlmA += 0.5 * itlibA.filter.get_qlms( ivfA_cacher.load(fn(idx)), ivfA_cacher.load(fn_wf(idx)),  q_geom, alm_wf_leg2=ebwf_dat)[0]

            ebwf_dat = ivfB_cacher.load('ebwf_dat')
            qlmB = 0.5 *  itlibB.filter.get_qlms(itlibB.dat_maps, ebwf_dat, q_geom, alm_wf_leg2=ivfB_cacher.load(fn_wf(idx)))[0]
            qlmB += 0.5 * itlibB.filter.get_qlms(ivfB_cacher.load(fn(idx)), ivfB_cacher.load(fn_wf(idx)), q_geom,alm_wf_leg2=ebwf_dat)[0]

            np.savetxt(fn_ds(idx) + '.dat', utils_hp.alm2cl(qlmA, qlmB, lmax_qlm, mmax_qlm, lmax_qlm))
            print("cached " + fn_ds(idx) + '.dat')

        i, j = int(idx), ss_dict[int(idx)]
        fn_ss = folder + '/qcl_ss_%04d_%04d' % (min(i, j), max(i, j)) + '.dat'
        if not os.path.exists(fn_ss):
            wf_i = ivfA_cacher.load(fn_wf(i))
            wf_j = ivfA_cacher.load(fn_wf(j))
            qlmA =   0.5 * itlibA.filter.get_qlms(ivfA_cacher.load(fn(i)), wf_i,  q_geom, alm_wf_leg2=wf_j)[0]
            qlmA +=  0.5 * itlibA.filter.get_qlms(ivfA_cacher.load(fn(j)), wf_j,  q_geom, alm_wf_leg2=wf_i)[0]

            wf_i = ivfB_cacher.load(fn_wf(i))
            wf_j = ivfB_cacher.load(fn_wf(j))
            qlmB =  0.5 * itlibB.filter.get_qlms(ivfB_cacher.load(fn(i)), wf_i, q_geom, alm_wf_leg2=wf_j)[0]
            qlmB += 0.5 * itlibB.filter.get_qlms(ivfB_cacher.load(fn(j)), wf_j, q_geom, alm_wf_leg2=wf_i)[0]

            np.savetxt(fn_ss, utils_hp.alm2cl(qlmA, qlmB, lmax_qlm, mmax_qlm, lmax_qlm))
            print("cached " + fn_ss)



def ss_ds(qe_key:str, itr:int, mcs:np.ndarray, itlib:cs_iterator.qlm_iterator, itlib_phases:cs_iterator.qlm_iterator, ss_dict:dict, rdn0tol=None,
          assert_phases_exist=False):
    """Builds ds and ss spectra

        Args:
            qe_key: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            itr: iteration index of MAP phi
            mcs: sim indices
            itlib: iterator instance to compute the ds ans ss for
            itlib_phases: iterator instance that generates the unlensed CMB phases for the sims (use this if want paired sims)
            assert_phases_exist: set this if you expect the phases to be already calculatex

    """
    
    assert itr > 0
    assert qe_key in ['p_p', 'ptt'], 'MAP_opfilt_aniso_tp.synalm not implemented'
    #TODO: implement the MV here and the synalm in MAP_opfilt_aniso_tp
    print('Start computing ss ds')

    assert hasattr(itlib.filter, 'synalm')
    # Setting up fitlering instance:
    dlm = itlib.get_hlm(itr - 1, 'p')
    itlib.hlm2dlm(dlm, True)
    ffi = itlib.filter.ffi.change_dlm([dlm, None], itlib.mmax_qlm, cachers.cacher_mem())
    itlib.filter.set_ffi(ffi)
    chain_descr = itlib.chain_descr
    if rdn0tol is not None:
        chain_descr[0][5] = 10**(-rdn0tol)
    mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
    
    if mpi.rank == 0:
        # Avoid file exists errors when creating the caching directory
        cachers.cacher_npy(opj(itlib.lib_dir, fdir_dsss(itr, rdn0tol)) )
        cachers.cacher_npy(opj(itlib_phases.lib_dir, fdir_dsss(itr, rdn0tol)))
    mpi.barrier()

    ivf_cacher = cachers.cacher_npy(opj(itlib.lib_dir, fdir_dsss(itr, rdn0tol)))
    ivf_phas_cacher = cachers.cacher_npy(opj(itlib_phases.lib_dir, fdir_dsss(itr, rdn0tol)))
    print('RDN0 libdir: ' + ivf_cacher.lib_dir)
    # data solution: as a check, should match closely the WF estimate in itlib folder
    fn_wf_dat = 'xwf_dat'
    if mpi.rank == 0:
        if not ivf_cacher.is_cached(fn_wf_dat):
            print(f'RDNO rank {mpi.rank} Getting the data WF')
            soltn = np.zeros(utils_hp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, itlib.dat_maps, dot_op=itlib.filter.dot_op())
            ivf_cacher.cache(fn_wf_dat, soltn)
    mpi.barrier()
    # if hasattr(itlib.filter, 'n_inv'):
    # The filtering is done on the QU maps
    fn_wf = lambda this_idx : 'dat_wf_filtersim_%04d'%this_idx # Wiener-filtered sim
    fn = lambda this_idx : 'dat_filtersim_%04d'%this_idx # full sims
    # else:
        # The filtering is performed on the Elm and Blm
        # fn_wf = lambda this_idx : 'ebwf_filtersim_%04d'%this_idx # Wiener-filtered sim
        # fn = lambda this_idx : 'eb_filtersim_%04d'%this_idx # full sims

    fn_unl = lambda this_idx : 'unllm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile
    # fn_eunl = lambda this_idx : 'unlelm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile

    
    def _sim_unl(itlib, lmax_sol, mmax_sol):
        if qe_key == 'p_p':
            assert np.all(itlib_phases.cls_filt['ee'][:lmax_sol+1] == itlib.cls_filt['ee'][:lmax_sol+1]), 'inconsistent inputs'
            return utils_hp.synalm(itlib_phases.cls_filt['ee'][:lmax_sol+1], lmax_sol, mmax_sol)
        elif qe_key == 'ptt':
            assert np.all(itlib_phases.cls_filt['tt'][:lmax_sol+1] == itlib.cls_filt['tt'][:lmax_sol+1]), 'inconsistent inputs'
            return utils_hp.synalm(itlib_phases.cls_filt['tt'][:lmax_sol+1], lmax_sol, mmax_sol)
        elif qe_key == 'p':
            #!FIXME should generate correlated TE
            return [utils_hp.synalm(itlib_phases.cls_filt[cl][:lmax_sol+1], lmax_sol, mmax_sol) for cl in ['tt', 'ee']]
        
    for i in np.unique(mcs)[mpi.rank::mpi.size]:
        idx = int(i)
        if not ivf_cacher.is_cached(fn_wf(idx)) or not ivf_cacher.is_cached(fn(idx)):
            print(f'RDNO: rank {mpi.rank} getting WF sim {idx}')
            if not ivf_phas_cacher.is_cached(fn_unl(idx)):
                assert (not assert_phases_exist)
                lmax_sol, mmax_sol = itlib_phases.filter.lmax_sol, itlib_phases.filter.mmax_sol
                assert (lmax_sol, mmax_sol) == (itlib.filter.lmax_sol, itlib.filter.mmax_sol), 'inconsistent inputs'
                xlm_unl = _sim_unl(itlib, lmax_sol, mmax_sol)
                ivf_phas_cacher.cache(fn_unl(idx), xlm_unl)
            xlm_unl = ivf_phas_cacher.load(fn_unl(idx))
            xlm_dat = itlib.filter.synalm(itlib.cls_filt, cmb_phas=xlm_unl)#, get_unlelm=True) # The unlensed CMB are the same but not the Phi map
            ivf_cacher.cache(fn(idx), xlm_dat)
            soltn = np.zeros(utils_hp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, ivf_cacher.load(fn(idx)), dot_op=itlib.filter.dot_op())
            ivf_cacher.cache(fn_wf(idx), soltn)
    mpi.barrier()

    if qe_key == 'p_p':
        get_qlms = itlib.filter.get_qlms_old
    elif qe_key == 'ptt':
        get_qlms = itlib.filter.get_qlms

    # builds qcl:
    fn_ds = lambda this_idx : ivf_cacher.lib_dir + '/qcl_ds_%04d'%this_idx

    # _q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    # print(_q_geom.geom)
    # print(_q_geom.pbound)

    q_geom = itlib.filter.ffi.pbgeom  
    #FIXME Seems like it is different from pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    # but should be the same, maybe only the memory adress is different so python see it as different in self.filtr._get_gpmap
    # print(q_geom.geom)
    # print(q_geom.pbound)

    for i in np.unique(mcs)[mpi.rank::mpi.size]:
        idx = int(i)
        if not os.path.exists(fn_ds(idx)+ '.dat'):
            xwf_dat =  ivf_cacher.load(fn_wf_dat)
            qlm  = 0.5 * get_qlms( itlib.dat_maps, xwf_dat,  q_geom, alm_wf_leg2=ivf_cacher.load(fn_wf(idx)))[0]
            qlm += 0.5 * get_qlms( ivf_cacher.load(fn(idx)), ivf_cacher.load(fn_wf(idx)),  q_geom, alm_wf_leg2=xwf_dat)[0]
            np.savetxt(fn_ds(idx) + '.dat', utils_hp.alm2cl(qlm, qlm, itlib.lmax_qlm, itlib.mmax_qlm, itlib.lmax_qlm))
            print("cached " + fn_ds(idx) + '.dat')

        i, j = int(idx), ss_dict[int(idx)]
        fn_ss = ivf_cacher.lib_dir + '/qcl_ss_%04d_%04d' % (min(i, j), max(i, j)) + '.dat'
        if not os.path.exists(fn_ss):
            wf_i = ivf_cacher.load(fn_wf(i))
            wf_j = ivf_cacher.load(fn_wf(j))
            qlm =   0.5 * get_qlms( ivf_cacher.load(fn(i)), wf_i,  q_geom, alm_wf_leg2=wf_j)[0]
            qlm +=  0.5 * get_qlms( ivf_cacher.load(fn(j)), wf_j,  q_geom, alm_wf_leg2=wf_i)[0]
            np.savetxt(fn_ss, utils_hp.alm2cl(qlm, qlm, itlib.lmax_qlm, itlib.mmax_qlm, itlib.lmax_qlm))
            print("cached " + fn_ss)




if __name__ == '__main__':
    import argparse
    import importlib
    

    parser = argparse.ArgumentParser(description='Compute RDN0 ')
    parser.add_argument('parfile', type=str, nargs=1)
    # parser.add_argument('-par', dest='par', type=str, required=True, help='parameter file which was used to estimate the lensing maps')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-datidx', dest='datidx', type=int, default=0, help='index of data map for which we compute RDN0')
    parser.add_argument('-Nsims', dest='Nsims', type=int, default=100, help='number of RDN0 simulations')
    parser.add_argument('-Nroll', dest='Nroll', type=int, default=10, help='Shift sims i->i+1 by batch of Nroll')
    parser.add_argument('-par_pha', dest='par_pha', type=str, default=None, help='parameter file that generates the unlensed CMB phases for the sims (use this if want paired sims)')
    parser.add_argument('-itmax', dest='itmax', type=int, default=15, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-rdn0tol', dest='rdn0tol', type=float, default=5., help='-log10 of cg tolerance default for RD sims WF')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7., help='-log10 of lensing accuracy')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')

    args = parser.parse_args()
    # args = parser.parse_args()
    # assert '.py' not in args.par[-3:], "Remove the .py from the param file"
    try:
        par = importlib.machinery.SourceFileLoader('parfile', args.parfile[0]).load_module()
        # par = importlib.import_module('delensalot.params.'+args.par)
    except ModuleNotFoundError:
        assert 0, "Could not find the parameter file {}".format(args.par)

    if args.par_pha is None: 
        args.par_pha = args.parfile

    try:
        par_pha = importlib.machinery.SourceFileLoader('parfile_pha', args.par_pha[0]).load_module()

        # par_pha = importlib.import_module('delensalot.params.'+args.par_pha)
    except ModuleNotFoundError:
        assert 0, "Could not find the parameter file {}".format(args.par_pha)


    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?

    print('Getting itpha:')
    itpha = par_pha.get_itlib(args.k, 0, args.v, tol_iter(0), epsilon=10 ** (- args.epsilon))
    
    print('Getting itlib:')
    itlib = par.get_itlib(args.k, args.datidx, args.v, tol_iter(0), epsilon=10 ** (- args.epsilon))
    # itlib = get_itlib(args.qe_key, idx, version, 1., epsilon=10 ** (- args.epsilon), numthreads=args.nthreads)

    itr = args.itmax 
    Nroll = args.Nroll
    Nsims = args.Nsims

    mcs = np.arange(0, Nsims) # Simulations used to get the RDN0
    # Note: These sims are not the ones generated by the get_itlib of the param files. 
    # They are new gaussian realisations generated with utils_hp.synalm in ss_ds() function. 
    # They will be lensed by the deflection field estimate of iter-1,
    #  and then Wiener filtered with the same iter-1 deflection etsimate, 
    # in order to estimate the phi MAP on it.

    # Shift the order of simulations by batches of Nroll
    ss_dict =  _ss_dict(mcs, Nroll)

    ss_ds(args.k, itr, mcs, itlib, itpha, ss_dict, rdn0tol=args.rdn0tol, assert_phases_exist=False)
    export_dsss(itr, args.k, itlib.lib_dir, par.suffix, args.datidx, args.v, ss_dict, mcs, Nroll, args.rdn0tol)
    # export_cls(itlib.lib_dir, args.k,  0,  par.suffix, args.datidx)
    # export_cls(itlib.lib_dir, args.k, itr,  par.suffix, args.datidx)
    # export_nhl(itlib.lib_dir, args.k, par, args.datidx)