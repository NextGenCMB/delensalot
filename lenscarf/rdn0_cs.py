"""This module calculates and caches generalized QE estimates for the MAP solution RDN0

Example for how to use:

python rdn0_cs.py -par cmbs4wide_idealized -k p_p -datidx 0


"""

import numpy as np, os
import lenscarf
from lenscarf.iterators import cs_iterator
from lenscarf import cachers, utils_hp
from lenscarf.utils_scarf import pbdGeometry, pbounds, scarfjob
from plancklens.qcinv import multigrid
from plancklens.utils import stats, cli, mchash
from plancklens.sims import planck2018_sims
from os.path import join as opj
from plancklens.helpers import mpi



output_dir = opj(os.path.dirname(os.path.dirname(lenscarf.__file__)), 'outputs')
output_sim = lambda suffix, datidx: opj(output_dir, suffix, 'sim_{:04d}'.format(datidx))

fdir_dsss = lambda itr : f'ds_ss_it{itr}'
fn_cls_dsss = lambda itr, ss_dict : f'cls_dsss_it{itr}_{mchash(ss_dict)}.dat'

def export_dsss(itr:int, libdir:str, suffix:str, datidx:int, ss_dict:dict=None):
    if ss_dict is None:
        Nroll = 10
        Nsims = 100
        ss_dict =  Nroll * (np.arange(Nsims) // Nroll) + (np.arange(Nsims) + 1) % Nroll

    itdir = opj(libdir, fdir_dsss(itr))
    ds, ss = load_ss_ds(np.arange(len(ss_dict)), ss_dict, itdir, docov=True)
    print(ds.N, ss.N, ds.size)
    lmax = ds.size - 1
    pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)
    arr = np.array([ds.mean() * pp2kki, ss.mean() * pp2kki, ds.sigmas_on_mean()*  pp2kki,  ss.sigmas_on_mean()* pp2kki, np.ones_like(ds.mean()) * ds.N, np.ones_like(ss.mean()) * ss.N])
    fmt = ['%.7e'] * 4 + ['%3i'] * 2
    header = '1e7 kk2pp times  : ds    ss   ds_erroronmean, ss_erroronmean , number of ds sims, number of ss sims'
    header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^2 * 1e7 times this   (ds ss is response-like)'
    fn_dir = output_sim(suffix, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    # fn_cldsss = 'cls_dsss_itr{}.dat'
    np.savetxt(opj(fn_dir, fn_cls_dsss(itr, ss_dict)), arr.transpose(), fmt=fmt, header=header)

def export_nhl(libdir:str, parfile, datidx:int):
    fn_dir = output_sim(parfile.suffix, datidx)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    fn = os.path.join(fn_dir, 'QE_knhl.dat')
    if not os.path.exists(fn):
        GG = ss_ds_QE(libdir, parfile, datidx)
        lmax = len(GG) - 1
        pp2kki = cli(0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7)
        header = 'semi-analytical kappa unnormalized rdn0'
        header += '\n' + 'Raw phi-based spec obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this   (ds ss is response-like)'
        np.savetxt(fn, GG * pp2kki, header=header)

def ss_ds_QE(libdir, parfile, datidx):
    # For the QE we use the semi-analytical N0:
    from lenscarf.utils_hp import almxfl, alm2cl
    from plancklens.nhl import get_nhl
    # TODO: Check if alm_bar should be the WF or the IVF alms 

    # if hasattr(itlib.filter, 'n_inv'):
    #     fn = lambda this_idx : 'qu_filtersim_%04d'%this_idx # full sims
    # else:
    #     fn = lambda this_idx : 'eb_filtersim_%04d'%this_idx # full sims
    # alm_bar = np.copy(np.load(opj(libdir, 'ds_ss', 'ebwf_dat.npy')))
    sht_job = scarfjob()
    sht_job.set_geometry(parfile.ninvjob_geometry)
    sht_job.set_triangular_alm_info(parfile.lmax_ivf,parfile.mmax_ivf)
    tr = int(os.environ.get('OMP_NUM_THREADS', 8))
    sht_job.set_nthreads(tr)
    alm_bar = np.copy(np.array(sht_job.map2alm_spin(parfile.sims_MAP.get_sim_pmap(datidx), 2)))
    lmax, mmax = parfile.lmax_ivf, parfile.mmax_ivf
    # print(alm_bar.shape)
    # print(parfile.transf_elm.shape)
    # print(parfile.fel.shape)
    almxfl(alm_bar[0], cli(parfile.transf_elm) * parfile.fel, mmax, inplace=True)
    almxfl(alm_bar[1], cli(parfile.transf_blm) * parfile.fbl, mmax, inplace=True)
    cls_ivfs = {'ee': alm2cl(alm_bar[0], alm_bar[0], lmax, mmax, lmax),
                'bb': alm2cl(alm_bar[1], alm_bar[1], lmax, mmax, lmax)}
    GG, CC, GC, CG = get_nhl('p_p', 'p_p', parfile.cls_len, cls_ivfs, lmax, lmax, lmax_out=parfile.lmax_qlm)
    return GG


def export_cls(libdir:str, itr:int,suffix:str, datidx:int):
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
    fn_dir = output_sim(suffix, datidx)
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



def ss_ds(itr:int, mcs:np.ndarray, itlib:cs_iterator.qlm_iterator, itlib_phases:cs_iterator.qlm_iterator, ss_dict:dict,
          assert_phases_exist=False):
    """Builds ds and ss spectra

        Args:
            itr: iteration index of MAP phi
            mcs: sim indices
            itlib: iterator instance to compute the ds ans ss for
            itlib_phases: iterator instance that generates the unlensed CMB phases for the sims (use this if want paired sims)
            assert_phases_exist: set this if you expect the phases to be already calculatex

    """
    
    assert itr > 0

    assert hasattr(itlib.filter, 'synalm')
    # Setting up fitlering instance:
    dlm = itlib.get_hlm(itr - 1, 'p')
    itlib.hlm2dlm(dlm, True)
    ffi = itlib.filter.ffi.change_dlm([dlm, None], itlib.mmax_qlm, cachers.cacher_mem())
    itlib.filter.set_ffi(ffi)
    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, itlib.filter)
    
    if mpi.rank == 0:
        # Avoid file exists errors when creating the caching directory
        cachers.cacher_npy(opj(itlib.lib_dir, fdir_dsss(itr)) )
        cachers.cacher_npy(opj(itlib_phases.lib_dir, fdir_dsss(itr)))
    mpi.barrier()

    ivf_cacher = cachers.cacher_npy(opj(itlib.lib_dir, fdir_dsss(itr)))
    ivf_phas_cacher = cachers.cacher_npy(opj(itlib_phases.lib_dir, fdir_dsss(itr)))

    # data solution: as a check, should match closely the WF estimate in itlib folder
    if mpi.rank == 0:
        if not ivf_cacher.is_cached('ebwf_dat'):
            soltn = np.zeros(utils_hp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, itlib.dat_maps, dot_op=itlib.filter.dot_op())
            ivf_cacher.cache('ebwf_dat', soltn)
    mpi.barrier()
    if hasattr(itlib.filter, 'n_inv'):
        # The filtering is done on the QU maps
        fn_wf = lambda this_idx : 'quwf_filtersim_%04d'%this_idx # Wiener-filtered sim
        fn = lambda this_idx : 'qu_filtersim_%04d'%this_idx # full sims
        fn_eunl = lambda this_idx : 'unlelm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile  
    else:
        # The filtering is performed on the Elm and Blm
        fn_wf = lambda this_idx : 'ebwf_filtersim_%04d'%this_idx # Wiener-filtered sim
        fn = lambda this_idx : 'eb_filtersim_%04d'%this_idx # full sims
        fn_eunl = lambda this_idx : 'unlelm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile
    for i in np.unique(mcs)[mpi.rank::mpi.size]:
        idx = int(i)
        if not ivf_cacher.is_cached(fn_wf(idx)) or not ivf_cacher.is_cached(fn(idx)):
            if not ivf_phas_cacher.is_cached(fn_eunl(idx)):
                assert (not assert_phases_exist)
                lmax_sol, mmax_sol = itlib_phases.filter.lmax_sol, itlib_phases.filter.mmax_sol
                assert (lmax_sol, mmax_sol) == (itlib.filter.lmax_sol, itlib.filter.mmax_sol), 'inconsistent inputs'
                assert np.all(itlib_phases.cls_filt['ee'][:lmax_sol+1] == itlib.cls_filt['ee'][:lmax_sol+1]), 'inconsistent inputs'
                elm_unl = utils_hp.synalm(itlib_phases.cls_filt['ee'][:lmax_sol+1], lmax_sol, mmax_sol)
                ivf_phas_cacher.cache(fn_eunl(idx), elm_unl)
            elm_unl = ivf_phas_cacher.load(fn_eunl(idx))
            elm_unl, eblm_dat = itlib.filter.synalm(itlib.cls_filt, cmb_phas=elm_unl, get_unlelm=True) # The unlensed CMB are the same but not the Phi map
            ivf_cacher.cache(fn(idx), eblm_dat)
            soltn = np.zeros(utils_hp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, ivf_cacher.load(fn(idx)), dot_op=itlib.filter.dot_op())
            ivf_cacher.cache(fn_wf(idx), soltn)
    mpi.barrier()
    # builds qcl:
    fn_ds = lambda this_idx : ivf_cacher.lib_dir + '/qcl_ds_%04d'%this_idx
    for i in np.unique(mcs)[mpi.rank::mpi.size]:
        q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
        idx = int(i)
        if not os.path.exists(fn_ds(idx)+ '.dat'):
            ebwf_dat =  ivf_cacher.load('ebwf_dat')
            qlm  = 0.5 * itlib.filter.get_qlms( itlib.dat_maps, ebwf_dat,  q_geom, alm_wf_leg2=ivf_cacher.load(fn_wf(idx)))[0]
            qlm += 0.5 * itlib.filter.get_qlms( ivf_cacher.load(fn(idx)), ivf_cacher.load(fn_wf(idx)),  q_geom, alm_wf_leg2=ebwf_dat)[0]
            np.savetxt(fn_ds(idx) + '.dat', utils_hp.alm2cl(qlm, qlm, itlib.lmax_qlm, itlib.mmax_qlm, itlib.lmax_qlm))
            print("cached " + fn_ds(idx) + '.dat')

        i, j = int(idx), ss_dict[int(idx)]
        fn_ss = ivf_cacher.lib_dir + '/qcl_ss_%04d_%04d' % (min(i, j), max(i, j)) + '.dat'
        if not os.path.exists(fn_ss):
            wf_i = ivf_cacher.load(fn_wf(i))
            wf_j = ivf_cacher.load(fn_wf(j))
            qlm =   0.5 * itlib.filter.get_qlms( ivf_cacher.load(fn(i)), wf_i,  q_geom, alm_wf_leg2=wf_j)[0]
            qlm +=  0.5 * itlib.filter.get_qlms( ivf_cacher.load(fn(j)), wf_j,  q_geom, alm_wf_leg2=wf_i)[0]
            np.savetxt(fn_ss, utils_hp.alm2cl(qlm, qlm, itlib.lmax_qlm, itlib.mmax_qlm, itlib.lmax_qlm))
            print("cached " + fn_ss)



def load_ss_ds(mcs, ss_dict, folder:str, docov=False):
    """Loads precomputed ds and ss stats.

    """
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
    print('ds ss found %s and %s sims'%(ds.N, ss.N))
    return ds, ss

if __name__ == '__main__':
    import argparse
    import importlib
    

    parser = argparse.ArgumentParser(description='Compute RDN0 ')
    parser.add_argument('-par', dest='par', type=str, required=True, help='parameter file which was used to estimate the lensing maps')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-datidx', dest='datidx', type=int, default=0, help='index of data map for which we compute RDN0')
    parser.add_argument('-Nsims', dest='Nsims', type=int, default=100, help='number of RDN0 simulations')
    parser.add_argument('-Nroll', dest='Nroll', type=int, default=10, help='Shift sims i->i+1 by batch of Nroll')
    parser.add_argument('-par_pha', dest='par_pha', type=str, default=None, help='parameter file that generates the unlensed CMB phases for the sims (use this if want paired sims)')
    parser.add_argument('-itmax', dest='itmax', type=int, default=15, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')

    args = parser.parse_args()
    assert '.py' not in args.par[-3:], "Remove the .py from the param file"
    try:
        par = importlib.import_module('lenscarf.params.'+args.par)
    except ModuleNotFoundError:
        assert 0, "Could not find the parameter file {}".format(args.par)

    if args.par_pha is None: 
        args.par_pha = args.par
    try:
        par_pha = importlib.import_module('lenscarf.params.'+args.par_pha)
    except ModuleNotFoundError:
        assert 0, "Could not find the parameter file {}".format(args.par_pha)


    tol_iter   = lambda it : 10 ** (- args.tol) # tolerance a fct of iterations ?

    itpha = par_pha.get_itlib(args.k, 0, args.v, tol_iter(0))
    
    itlib = par.get_itlib(args.k, args.datidx, args.v, tol_iter(0))

    # print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
    # print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

    # itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
    # itlib.soltn_cond   = soltn_cond(i)
    # print("doing iter " + str(i))
    # itlib.iterate(i, 'p')
    itr = args.itmax 
    Nroll = args.Nroll
    Nsims = args.Nsims

    print('titit')

    # TODO : Check if it removes index of data later 
    mcs = np.arange(0, Nsims) # Simulations used to get the RDN0

    # Shift the order of simulations by batches of Nroll
    ss_dict = {idx: Nroll * (idx // Nroll) + (idx + 1) % Nroll for idx in mcs}

    print('Start computing ss ds')
    ss_ds(itr, mcs, itlib, itpha, ss_dict, assert_phases_exist=False)

    export_dsss(itr, itlib.lib_dir, par.suffix, args.datidx, ss_dict)
    export_cls(itlib.lib_dir, 0,  par.suffix, args.datidx)
    export_cls(itlib.lib_dir, itr,  par.suffix, args.datidx)
    export_nhl(itlib.lib_dir, par, args.datidx)

    # mpi.barrier()
    # if mpi.rank == 0: