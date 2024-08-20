from termios import N_STRIP
import numpy as np
from os.path import join as opj
import os
import healpy as hp
from importlib.machinery import SourceFileLoader

from scipy.interpolate import UnivariateSpline as spline

from plancklens.sims import planck2018_sims
from plancklens.utils import mchash, cls_dot, stats
from plancklens import qresp, n0s, nhl
from plancklens.n1 import n1 as n1_lib

from delensalot import utils
from delensalot.utils import read_map, cli
from delensalot.biases import iterbiasesN0N1
from delensalot.biases import rdn0_cs
from delensalot.core import cachers
from delensalot.core.helper import utils_scarf
from delensalot.core.iterator import statics
from delensalot.utility import utils_hp as uhp
from delensalot.utility.utils_hp import alm_copy
from delensalot.utility.utils_plot import pp2kk, bnd

from lensitbiases import n1_fft

_write_alm = lambda fn, alm : hp.write_alm(fn, alm, overwrite=True)
 
class cpp_sims_lib:
    def __init__(self, k:str,  param_file:str, tol:int, eps:int,  version:str='', qe_version='', label:str='', n0n1_libdir:str or None=None, cache_in_home=False):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.

        Args:
            n0n1_libdir:N0 and N1, for QE and MAP will be loaded or stored there
            cache_in_home: The default cacher is on the scratch but can cache things in the home if True
        """
        
        self.k = k
        self.version = version
        self.qe_version = qe_version

        # self.iterator_version = iterator_version
        self.tol = tol 
        self.eps = eps
        self.label = label
        # Load the parameters defined in the param_file
        self.param_file = param_file
        self.param = SourceFileLoader(param_file, param_file +'.py').load_module()
        self.TEMP =  self.param.TEMP
        self.lmax_qlm = self.param.lmax_qlm
        self.mmax_qlm = self.param.mmax_qlm
        self.cache_in_home = cache_in_home

        if self.cache_in_home is False:
            self.cachedir = self.TEMP
        else:
            splt = self.TEMP.split('/')[-2:]
            self.cachedir = opj(os.environ['HOME'], *splt)

        self.cacher_param = cachers.cacher_npy(opj(self.cachedir, f'cpplib_tol{self.tol}_eps{self.eps}' + self.version))
        self.fsky = self.get_fsky() 

        # Cl weights used in the QE (either lensed Cls or grad Cls)
        try:
            self.cls_weights = self.param.ivfs.cl
        except AttributeError:
            self.cls_weights = self.param.ivfs.ivfs.cl

        # Grad cls used for CMB response
        self.cls_grad = self.param.cls_grad

        self.cpp_fid = self.param.cls_unl['pp']

        # self.config = (self.param.nlev_t, self.param.nlev_p, self.param.beam, 
        #                (self.param.lmin_tlm,  self.param.lmin_elm, self.param.lmin_blm), 
        #                self.param.lmax_ivf, self.param.lmax_qlm)
        
        if 'nocut' in qe_version:
            self.ivfs = self.param.ivfs_nocut
            self.qcls_ss = self.param.qcls_ss_nocut
            self.qcls_ds = self.param.qcls_ds_nocut
            self.qcls_dd = self.param.qcls_dd_nocut
            self.qlms_dd = self.param.qlms_dd_nocut
        elif 'qeinh' in qe_version:
            self.ivfs = self.param.ivfs_nocut_inh
            self.qcls_ss = self.param.qcls_ss_nocut_inh
            self.qcls_ds = self.param.qcls_ds_nocut_inh
            self.qcls_dd = self.param.qcls_dd_nocut_inh
            self.qlms_dd = self.param.qlms_dd_nocut_inh
        else:
            self.ivfs = self.param.ivfs
            self.qcls_ss = self.param.qcls_ss
            self.qcls_ds = self.param.qcls_ds
            self.qcls_dd = self.param.qcls_dd
            self.qlms_dd = self.param.qlms_dd
        try:
            self.nhllib = nhl.nhl_lib_simple(opj(self.cachedir, 'nhllib' + qe_version), self.ivfs, self.ivfs.ivfs.cl, self.param.lmax_qlm)
        except AttributeError:
            self.nhllib = nhl.nhl_lib_simple(opj(self.cachedir, 'nhllib' + qe_version), self.ivfs, self.ivfs.cl, self.param.lmax_qlm)

        self.n0n1_libdir = n0n1_libdir


    def libdir_sim(self, simidx, tol=None, eps=None):
        if tol is None: tol = self.tol 
        if eps is None: eps = self.eps
        # return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)
        return self.param.libdir_iterators(self.k, simidx, self.version, self.qe_version, tol, eps)

    def get_itlib_sim(self, simidx, tol=None, eps=None):
        if tol is None: tol = self.tol 
        if eps is None: eps = self.eps
        tol_iter  = 10 ** (- tol) 
        epsilon = 10**(-eps)
        return self.param.get_itlib(self.k, simidx, self.version, cg_tol=tol_iter, epsilon=epsilon)

    def cacher_sim(self, simidx, verbose=False):
        if self.cache_in_home is False:
            cacher = cachers.cacher_npy(opj(self.libdir_sim(simidx, self.tol, self.eps), 'cpplib'), verbose=verbose)
        else:
            splt = self.libdir_sim(simidx, self.tol, self.eps).split('/')[-3:]
            cacher = cachers.cacher_npy(opj(os.environ['HOME'], *splt, 'cpplib'), verbose=verbose)
        return cacher

    def get_plm(self, simidx, itr, use_cache=True):
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx, self.tol, self.eps))
            # print(self.libdir_sim(simidx))
            fn = f"phi_plm_it{itr:03.0f}"
            if not cacher.is_cached(fn):
                plm = statics.rec.load_plms(self.libdir_sim(simidx, self.tol, self.eps), [itr])[0]
                cacher.cache(fn, plm)
            plm = cacher.load(fn)
            return plm
        else:
            return statics.rec.load_plms(self.libdir_sim(simidx, self.tol, self.eps), [itr])[0]

    def get_plm_qe(self, simidx, use_cache=True, recache=False, verbose=False):

        if verbose:
            print('We get the QE qlms from ' + self.qlms_dd.lib_dir)
        
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            # print(self.libdir_sim(simidx))
            fn = f"phi_plm_qe" + self.qe_version
            # print(fn)
            if not cacher.is_cached(fn) or recache:                
                plm = self.qlms_dd.get_sim_qlm(self.k, int(simidx)) 
                cacher.cache(fn, plm)
            plm = cacher.load(fn)
            return plm
        else:
            return self.qlms_dd.get_sim_qlm(self.k, int(simidx)) 

    def get_sim_plm(self, idx):
        """Returns the input plm, depening if it is a sims_ffp10, a npipe sim or other sim"""
        if type(self.param.sims).__name__ == 'smicaNPIPE_wTpmask30amin':
            return self.param.sims.get_sim_plm(idx)
        elif type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            return planck2018_sims.cmb_unl_ffp10().get_sim_plm(idx)
        else:
            return self.param.sims.sims_cmb_len.get_sim_plm(idx)

    def get_plm_input(self, simidx, use_cache=True, recache=False):
        
        if not hasattr(self.param.sims, 'sims_cmb_len') or not hasattr(self.param.sims.sims_cmb_len, 'plm_shuffle') or self.param.sims.sims_cmb_len.plm_shuffle is None:
            shuffled_idx = simidx
        else:
            shuffled_idx = self.param.sims.sims_cmb_len.plm_shuffle(simidx)

        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_input"
            if not cacher.is_cached(fn) or recache:
                plm_in = alm_copy(self.get_sim_plm(shuffled_idx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm) # type: ignore
                cacher.cache(fn, plm_in)
            plm_in = cacher.load(fn)
            return plm_in
        else:
            return alm_copy(self.get_sim_plm(shuffled_idx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm) # type: ignore

    def get_eblm_dat(self, simidx, lmaxout=1024):
        QU_maps = self.param.sims_MAP.get_sim_pmap(simidx)
        tr = int(os.environ.get('OMP_NUM_THREADS', 8))
        #FIXME: Can we get read of scarf here?
        sht_job = utils_scarf.scarfjob()
        sht_job.set_geometry(self.param.ninvjob_geometry)
        sht_job.set_triangular_alm_info(self.param.lmax_ivf,self.param.mmax_ivf)
        sht_job.set_nthreads(tr)
        elm, blm = np.array(sht_job.map2alm_spin(QU_maps, 2))
        lmaxdat = hp.sphtfunc.Alm.getlmax(elm.size)
        elm = uhp.alm_copy(elm, mmaxin=lmaxdat, lmaxout=lmaxout, mmaxout=lmaxout)
        blm = uhp.alm_copy(blm, mmaxin=lmaxdat, lmaxout=lmaxout, mmaxout=lmaxout)
        return elm, blm

    def get_fsky(self):
        try:
            fn = 'fsky'
            if not self.cacher_param.is_cached(fn):
                mask = read_map(self.param.masks)
                fsky = np.sum(mask)/np.size(mask)
                print(fsky)
                self.cacher_param.cache(fn, fsky)
            return self.cacher_param.load(fn)
        except AttributeError:
            print('No masks defined in param file ' + self.param_file)
            return 1.

    def get_cl(self, alm, blm=None, lmax_out=None):
        if blm is None: blm = alm 
        return uhp.alm2cl(alm, blm, self.param.lmax_qlm, self.param.mmax_qlm, lmax_out)

    def almxfl(self, alm, cl):
        return uhp.almxfl(alm, cl, mmax = self.param.mmax_qlm, inplace=False)

    def get_cpp_input(self, simidx):
        fn_cpp_in = 'cpp_input'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_in):
            plmin = self.get_plm_input(simidx)
            cpp_in = self.get_cl(plmin)
            cacher.cache(fn_cpp_in, cpp_in)
        cpp_in = cacher.load(fn_cpp_in)
        return cpp_in

    def get_cpp_itXinput(self, simidx, itr,  recache=False):
        fn = 'cpp_in_x_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn) or recache:
            plmin = self.get_plm_input(simidx)
            # plmit = self.plms[simidx][itr]
            plmit = self.get_plm(simidx, itr)
            cpp_itXin = self.get_cl(plmit, plmin)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin
    
    def get_cpp_qeXinput(self, simidx, qe_version=None, recache=False, verbose=False):
        if qe_version == None:
            qe_version = self.qe_version
        fn = 'cpp_in_x_qe' + qe_version
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn) or recache:
            plmin = self.get_plm_input(simidx)
            plmqe = self.get_plm_qe(simidx, qe_version=qe_version, verbose=verbose)
            cpp_itXin = self.get_cl(plmqe, plmin)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin

    def get_cpp(self, simidx, itr, sub_mf=False, splitMF=True, mf_sims=None, recache=False):
        """Returns unnormalized Cpp MAP 
        
            Args:
                simidx: index of simulation 
                itr: Iteration of the phi MAP 
                sub_mf: if true will subtract an estimate of the MAP MF from self.get_mf
                mf_sims: provides the index of sims to estimate the MF

        """
        fn_cpp_it = 'cpp_it_{}'.format(itr) + sub_mf*"_submf" + sub_mf*splitMF*"_split_mf"
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it) or recache == True:
            plm = self.get_plm(simidx, itr)
            if sub_mf:
                assert mf_sims is not None, "Please provide mf_sims"
                if splitMF:
                    # Nmf = len(mf_sims)
                    mf_sims_1 = mf_sims[::2]
                    mf_sims_2 =  mf_sims[1::2]
                    mf1 = self.get_mf(itr, mf_sims_1, simidx=simidx, use_cache=True, verbose=False)
                    mf2 = self.get_mf(itr, mf_sims_2, simidx=simidx, use_cache=True, verbose=False)
                    cpp = self.get_cl(plm-mf1, plm-mf2)
                else:
                    mf = self.get_mf(itr, mf_sims, simidx=simidx, use_cache=True, verbose=False)
                    plm -= mf
                    cpp = self.get_cl(plm)
            else:
                cpp = self.get_cl(plm)
            cacher.cache(fn_cpp_it, cpp)
        cpp = cacher.load(fn_cpp_it)
        return cpp


    def get_cpp_qe(self, simidx, mc_sims_mf=None, qeresp=None, splitMF=True, recache=False):
        """Get nomalized Cpp QE

            Args: 
                simidx: index of sim to consider
                qeresp: Response of the QE, if None use the default one given by get_qe_resp
        """
        # if qe_version is None:
        #     qe_version = self.qe_version
        if qeresp is None:
            #!FIXME: Check the version here to have the ivfs nocut in response
            qeresp = self.get_qe_resp(recache=recache)
        cppqe = self.get_cpp_qe_raw(simidx, splitMF, recache, mc_sims_mf) * utils.cli(qeresp)**2
        return cppqe

    def get_cpp_qe_qcl(self, simidx):
        """Get the cached Cpp QE from run_qlms.py"""
        return self.qcls_dd.get_sim_qcl(self.k, simidx)

    def get_cpp_qe_raw(self, simidx, splitMF=True, recache=False, mc_sims_mf=None, fn_cpp_qe=None, verbose=False):
        """Returns unromalized Cpp QE"""
        
        if fn_cpp_qe is None:
            fn_cpp_qe = 'cpp_qe_raw' + splitMF*'_splitMF' + self.qe_version
            if mc_sims_mf is not None:
                fn_cpp_qe += ('_'+ mchash(mc_sims_mf))
        
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_qe) or recache:
            # plmqe  = _qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate
            plmqe = self.get_plm_qe(simidx, verbose=verbose, recache=recache)
            if mc_sims_mf is None:
                mc_sims_mf = self.param.mc_sims_mf_it0
            # QE mean-field
            if splitMF:
                # Nmf = len(self.param.mc_sims_mf_it0)
                mf_sims_1 =  np.unique(mc_sims_mf[::2])
                mf_sims_2 =  np.unique(mc_sims_mf[1::2])
                mf0_1 = self.get_mf0(simidx, mf_sims=mf_sims_1, verbose=verbose)
                mf0_2 = self.get_mf0(simidx, mf_sims=mf_sims_2, verbose=verbose)
                cppqe = self.get_cl(plmqe - mf0_1, plmqe - mf0_2)
            else:
                mf0 = self.get_mf0(simidx)
                plmqe -= mf0  # MF-subtracted unnormalized QE
            cacher.cache(fn_cpp_qe, cppqe)
        return cacher.load(fn_cpp_qe)

    def get_cpp_itmax(self, simidx, itmax):
        """Get Cpp at all iterations up to itmax"""
        cacher = self.cacher_sim(simidx)
        fn_cpp_it = lambda itr:  'cpp_it_{}'.format(itr)
        if np.any([not cacher.is_cached(fn_cpp_it(itr)) for itr in np.arange(itmax+1)]):
            plms = statics.rec.load_plms(self.libdir_sim(simidx), np.arange(itmax+1))
            for itr in np.arange(itmax+1):
                if not cacher.is_cached(fn_cpp_it(itr)):
                    _cpp = self.get_cl(plms[itr])
                    cacher.cache(fn_cpp_it(itr), _cpp)
        cpp = []
        for itr in np.arange(itmax+1):
            cpp.append(cacher.load(fn_cpp_it(itr)))
        return cpp   

    def get_mf0(self, simidx, mf_sims=None, verbose=False):
        """Get the QE mean-field"""

        _qlms_dd = self.qlms_dd
        if verbose:
            print(f'MF QE is computed from qlms in {_qlms_dd.lib_dir}')
        if mf_sims is None:
            mf_sims = np.unique(self.param.mc_sims_mf_it0 if not 'noMF' in self.version else np.array([]))
        else:
            mf_sims = np.unique(mf_sims)
        mf0 = _qlms_dd.get_sim_qlm_mf(self.k, mf_sims)  # Mean-field of the QE

        Nmf = len(mf_sims)
        for simid in np.atleast_1d(simidx):
            if simid in mf_sims:
                mf0 = (mf0 - _qlms_dd.get_sim_qlm(self.k, int(simid)) / Nmf) * (Nmf / (Nmf - 1))
                Nmf -= 1
        return mf0 

    def get_mf(self, itmax, mc_sims, simidx=None, use_cache=True, verbose=False, recache=False):
        """Get the mean field of the MAP, by averaging MAP estimates from a set simulations (caches the result).
            Adapted from plancklens.qest.get_sim_qlm_mf
            Args:
                itmax: Iteration of the MAP estimator
                mc_sims: simulation indices to use for the estimate.
                simidx: int or array containing indices of simulation to not take into account for the MF estimate

            Returns:
                plm_mf: Mean field plm 
        """
        this_mcs = np.unique(mc_sims)

        cacher = cachers.cacher_npy(self.cacher_param.lib_dir, verbose=verbose)
        fn =  f'simMF_itr{itmax}_k{self.k}_{mchash(mc_sims)}.fits'
        if not cacher.is_cached(fn) or recache:
            MF = np.zeros(hp.Alm.getsize(self.lmax_qlm), dtype=complex)
            if len(this_mcs) == 0: return MF
            for i, idx in utils.enumerate_progress(this_mcs, label='calculating MAP MF'):
                MF += self.get_plm(idx, itmax, use_cache=use_cache)
            MF = MF / len(this_mcs)
            cacher.cache(fn, MF)
        MF = cacher.load(fn)
        
        if simidx is not None:
            mc_sims_less = this_mcs
            for simid in np.atleast_1d(simidx):
                Nmf = len(mc_sims_less)
                if simid in this_mcs:
                    # We dont want to include the sim we consider in the mean-field.
                    if verbose:
                        print(f"Removing sim {simid} from MF estimate")
                        # print(Nmf)
                    mc_sims_less = np.delete(mc_sims_less, np.where(mc_sims_less==simid))
                    fn =  f'simMF_itr{itmax}_k_{mchash(mc_sims_less)}.fits'
                    if not cacher.is_cached(fn) or recache:
                        # MF = (MF - self.get_plm(simidx, itmax, use_cache=use_cache) / Nmf) * (Nmf / (Nmf - 1.))
                        MF = (Nmf * MF - self.get_plm(simid, itmax, use_cache=use_cache)) / (Nmf - 1.)
                        cacher.cache(fn, MF)
                    MF = cacher.load(fn)
        return MF
    
    def get_qe_resp(self, recache=False, resp_gradcls=True):
        #TODO: Implement the version to get the lmin_ivf=0 case

        fn_resp_qe = 'resp_qe_{}'.format(self.k) + self.qe_version
        if resp_gradcls: 
            fn_resp_qe += '_gradcls'
        cacher = self.cacher_param
        
        fals =  {'tt': self.ivfs.get_ftl(), 'ee':self.ivfs.get_fel(), 'bb':self.ivfs.get_fbl()}
        if not cacher.is_cached(fn_resp_qe) or recache:
            R = qresp.get_response(self.k, self.ivfs.lmax, 'p', self.cls_weights, self.cls_grad, fals, lmax_qlm=self.param.lmax_qlm)[0]
            cacher.cache(fn_resp_qe, R)
        R = cacher.load(fn_resp_qe)
        return R

    def get_map_resp(self, it=15, version=''):
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=it, version=version)
        return r_gg_fid

    def get_N0_N1_QE(self, normalize=True, resp_gradcls=True, n1fft=True, recache=False):
        """
        Get the QE N0 and N1 biases
        
        args:
            normalise: normalise by QE response 
            resp_gradcls: Use the grad lensed Cls in the response
            n1fft: estimate the N1 bias with the fast FFT algorithm
            recache: recompute the cached terms and overwrite them
            ivfs: provide the filtering of the CMB maps
            version: customisation parmaeters for filetrings
            
        returns:
            N0: (un) normalised N0 bias
            N1 (un) normalised N1 bias
        """
        # if ivfs is None:
        #     ivfs = self.ivfs
        # if qe_version is None and ivfs is None:
        
        version = self.qe_version 
        ivfs = self.ivfs

        fal_sepTP = {
            'tt': ivfs.get_ftl(),
            'ee': ivfs.get_fel(),
            'bb': ivfs.get_fbl()}
            # 'te': utils.cli(self.param.cls_len['te'])}
            # 'te': np.zeros_like(ivfs.get_fbl())}
            

        resp_qe = self.get_qe_resp(resp_gradcls)
        if self.n0n1_libdir is None:
            cacher = self.cacher_param
        else:
            cacher = cachers.cacher_npy(self.n0n1_libdir)
            
        fn_n0 = 'n0_qe_{}'.format(self.k) + version
        # print(fn_n0)
        if not cacher.is_cached(fn_n0) or recache:
            print(f'Computing N0 {fn_n0}')
            cls_dat = {spec: utils.cli(fal_sepTP[spec]) for spec in ['tt', 'ee', 'bb']}
            # Spectra of the inverse-variance filtered maps
            # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
            cls_ivfs_sepTP = cls_dot([fal_sepTP, cls_dat, fal_sepTP], ret_dict=True)
            #FIXME: For Sep TP, check if it is ok to put the TE to zero
            # for joint TP, will have to get the ivfs.get_fal here 
            cls_ivfs_sepTP['te'] = np.zeros_like(cls_ivfs_sepTP['tt'])
            NG, NC, NGC, NCG = nhl.get_nhl(self.k, self.k, self.cls_weights, cls_ivfs_sepTP, self.param.lmax_ivf, self.param.lmax_ivf,
                                    lmax_out=self.lmax_qlm)
            cacher.cache(fn_n0, NG)
        NG = cacher.load(fn_n0)
        
        if n1fft:
            dl = 50
            # n1_Ls = np.arange(50, (self.param.lmax_qlm // 50) * 50  + 50, 50)
            n1_Ls = np.arange(50, self.param.lmax_qlm , dl)
            fn_n1 = 'n1_fft_qe_dl_{}_{}'.format(dl, self.k) + version

            if not cacher.is_cached(fn_n1) or recache:
                # cls_noise_fid = {'tt': ( (self.param.nlev_t / 180 / 60 * np.pi) * utils.cli(self.param.transf_tlm) ) ** 2,
                #         'ee': ( (self.param.nlev_p / 180 / 60 * np.pi) * utils.cli(self.param.transf_elm) ) ** 2,
                #         'bb': ( (self.param.nlev_p / 180 / 60 * np.pi) * utils.cli(self.param.transf_blm) ) ** 2 }
                # fals, _, _, _ = iterbiasesN0N1.get_fals(self.k, self.param.cls_len, self.param.cls_len, cls_noise_fid, cls_noise_fid, self.param.lmin_ivf, self.param.lmax_ivf)

                lib = n1_fft.n1_fft(fal_sepTP, self.cls_weights, self.cls_weights, np.copy(self.param.cls_unl['pp']), lminbox=50, lmaxbox=5000, k2l=None)
                n1 = np.array([lib.get_n1(self.k, L, do_n1mat=False)  for L in n1_Ls])
                _n1 = spline(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / resp_qe[n1_Ls]**2, k=2,s=0, ext='zeros') (np.arange(len(NG)))
                ls = np.arange(self.param.lmax_qlm + 1)
                _n1 *= utils.cli(ls *1. * (ls*1.+1.)) ** 2 * resp_qe[ls]**2
                cacher.cache(fn_n1, _n1)
            _n1 = cacher.load(fn_n1)

        else:
            n1lib = n1_lib.library_n1(cacher.lib_dir + version, self.cls_weights['tt'], self.cls_weights['te'], self.cls_weights['ee'], self.lmax_qlm)

            _n1 = n1lib.get_n1(self.k, 'p',  self.param.cls_unl['pp'], fal_sepTP['tt'], fal_sepTP['ee'], fal_sepTP['bb'], Lmax=self.lmax_qlm)

        if normalize is False:
            return NG, _n1
        else:
            # resp_qe = self.get_qe_resp(resp_gradcls)
            return NG*utils.cli(resp_qe)**2, _n1*utils.cli(resp_qe)**2

    def get_N0_N1_iter(self, itermax=15, version='', recache=False, normalize=True):
        """
        Wrapper for the iterbias class and get_n0n1 function
        Returns the biased N0 N1, computed using the fiducial CMB Cls. 
        """
        #!FIXME: Get the N0 and N1 for lmin ivf no cut
        if self.n0n1_libdir is None:
            lib_dir = opj(self.TEMP, 'n0n1_iter'+version)
        else:
            lib_dir = self.n0n1_libdir
        
        if 'nocut' in version:
            lmin_ivf = 0 
        else:
            lmin_ivf = self.param.lmin_ivf
        itbias = iterbiasesN0N1.iterbiases(self.param.nlev_t, self.param.nlev_p, self.param.beam, lmin_ivf, self.param.lmax_ivf,
                                    self.param.lmax_qlm, self.param.cls_unl, None, lib_dir)
        N0_biased, N1_biased, r_gg_fid, r_gg_true = itbias.get_n0n1(self.k, itermax, None, None, version=version, recache=recache)
        if normalize is False:
            N0_biased *= r_gg_fid**2
            N1_biased *= r_gg_fid**2

        return N0_biased, N1_biased, r_gg_fid, r_gg_true


    def get_wf_fid(self, itermax=15, version=''):
        """Fiducial iterative Wiener filter.
        
        Normalisation of :math:`phi^{MAP}`
        :math:`\mathcal{W} = \frac{C_{\phi\phi, \mathrm{fid}}}{C_{\phi\phi, \mathrm{fid}} + 1/\mathcal{R}_L}`
 
        """
        if version == 'wN1_end':
            _, N1, resp_fid, _ = self.get_N0_N1_iter(itermax=itermax, version='')
            return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]) + N1[:self.lmax_qlm+1])

        else:
            _, N1, resp_fid, _ = self.get_N0_N1_iter(itermax=itermax, version=version)
            return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]))

    def get_wf_sim(self, simidx, itr, mf=False, mc_sims=None, recache=False):
        """Get the Wiener filter from the simulations.

        :math:`\hat \mathcal{W} = \frac{C_L{\phi^{\rm MAP} \phi{\rm in}}}{C_L{\phi^{\rm in} \phi{\rm in}}}`
        
        """
        fn = 'wf_sim_it{}'.format(itr) if mf is False else 'wf_sim_it{}_mfsub'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn) or recache:
            if mf is False:
                wf = self.get_cpp_itXinput(simidx, itr) * utils.cli(self.get_cpp_input(simidx)) / self.fsky
            else:
                plmin = self.get_plm_input(simidx)
                # plmit = self.plms[simidx][itr]
                mf = self.get_mf(itr, mc_sims, simidx, use_cache=True)
                plmit = self.get_plm(simidx, itr, use_cache=True) - mf
                cpp_itXin = self.get_cl(plmit, plmin)
                
                wf = cpp_itXin * utils.cli(self.get_cpp_input(simidx)) / self.fsky
            cacher.cache(fn, wf)
        return cacher.load(fn)

    def get_wf_eff(self, itmax_sims=15, itmax_fid=15, mf=False, mc_sims=None, version='', do_spline=True, lmin_interp=0, lmax_interp=None,  k=3, s=None, verbose=False, recache=False):
        """Effective Wiener filter averaged over several simulations
        We spline interpolate the ratio between the effective WF from simulations and the fiducial WF
        We take into account the sky fraction to get the simulated WFs
        
        Args:
            imin: Minimum index of simulations
            imax: Maximum index of simulations
            itmax_sims: Iteration of the MAP estimator
            itmax_fid: Iteration for the NO MAP theory computation

        Returns:
            wf_eff: Effective Wiener filter
            wfcorr_spl: Splined interpolation of the ratio between the effective and the fiducial Wiener filter
        
        """
        nsims = self.get_nsims_itmax(itmax_sims)
        fn_weff = f"wf_eff_{self.k}_itsim{itmax_sims}_itfid{itmax_fid}_nsims{nsims}_mf{mf}_v{version}" +f"_spl{do_spline}_{lmin_interp}_{lmax_interp}_{k}_{s}" * do_spline
        fn_wfspline = f"wf_spline_{self.k}_itsim{itmax_sims}_itfid{itmax_fid}_nsims{nsims}_mf{mf}_v{version}" f"_spl{do_spline}_{lmin_interp}_{lmax_interp}_{k}_{s}" * do_spline
        if np.any([not self.cacher_param.is_cached(fn) for fn in [fn_weff, fn_wfspline]]) or recache:
            wf_fid = self.get_wf_fid(itmax_fid, version=version)
            print(f'I use {nsims} sims to estimate the effective WF')
            print(fn_weff)
            wfsims_bias = np.zeros([nsims, len(wf_fid)])
            for isim in range(nsims):
                if verbose: print(f'wf eff {isim}/{nsims-1}')
                wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims, mf=mf, mc_sims=mc_sims, recache=recache) * utils.cli(wf_fid)
            wfcorr_mean = np.mean(wfsims_bias, axis=0)
            if do_spline:
                wfcorr_spl = np.zeros(len(wf_fid))
                if lmax_interp is None: lmax_interp=self.lmax_qlm
                ells = np.arange(lmin_interp, lmax_interp+1)
                wfcorr_spl[ells] = spline(ells, wfcorr_mean[ells], k=k, s=s)(ells)
            else:
                wfcorr_spl = wfcorr_mean
            wf_eff = wf_fid * wfcorr_spl
            self.cacher_param.cache(fn_weff, wf_eff)
            self.cacher_param.cache(fn_wfspline, wfcorr_spl)
        wf_eff = self.cacher_param.load(fn_weff)
        wfcorr_spl = self.cacher_param.load(fn_wfspline)
        return wf_eff[:self.lmax_qlm+1], wfcorr_spl


    def get_num_rdn0(self, itr=50, mcs=None, Nroll=None):
        "Retunr number of sims with RDN0 estimated"
        idx = 0 
        rdn0_computed = True
        while rdn0_computed:      
            outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
            fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
            # print(fn)
            # fn = opj(outputdir, 'cls_dsss.dat')
            try:
                np.loadtxt(fn)
                idx +=1
            except OSError:
                 rdn0_computed = False
        return idx

    
    def get_R_eff(self, rfid=None, dospline=True, lmin_interp=0, lmax_interp=None, k=3, s=None, itr=None, itmax_fid=15, rdsims=None, mcs=np.arange(0, 96), Nroll=8, version=''):
        """Correct the Response (normalisation) using the RDN0 estimates from several sims"""
        
        cacher = cachers.cacher_npy(self.cacher_param.lib_dir)
        
        if rdsims is None:
            nrdn0 = self.get_num_rdn0(itr, mcs, Nroll)
            rdsims = np.arange(0, nrdn0)
        print(f'I average {len(rdsims)} sims with rdn0 to get effective response')
        fn =  'rdn0_mean_{}_{}_Nroll{}_{}_{}'.format(self.k, itr, Nroll, mchash(rdsims), mchash(mcs))

        if not cacher.is_cached(fn):
            rdn0 = np.zeros(self.lmax_qlm+1)
            for idx in rdsims:
                rdn0 += self.load_rdn0_map(idx, itr, mcs, Nroll)[0]
            rdn0 /= len(rdsims)
            cacher.cache(fn, rdn0)
        rdn0 = cacher.load(fn)

        if rfid is None:
            r_gg_fid  = self.get_map_resp(itmax_fid, version=version)
        else:
            r_gg_fid = rfid
        Reff_Spline = np.ones(self.lmax_qlm+1) * r_gg_fid
        if lmax_interp is None: lmax_interp=self.lmax_qlm
        ells = np.arange(lmin_interp, lmax_interp+1)
        # Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        if dospline:
            Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] /self.fsky * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
            # Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells]  * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        else:
            Reff_Spline = rdn0 /self.fsky
        # kR_eff = np.zeros(4001)
        # R_eff  = r_gg_fid * Reff_Spline
        return Reff_Spline

    def get_qe_Reff(self, rfid=None, dospline=True, lmin_interp=0, lmax_interp=None, k=3, s=None,  Ndatasims=40, Nmcsims=100, Nroll=10):
        """Return teh QE effective response from using RDN0 estimates from a number of sims"""
        cacher = cachers.cacher_npy(self.cacher_param.lib_dir)

        fn =  'rdn0_mean_qe_{}_Nroll{}_Ndatasims{}_Nmcsims{}'.format(self.k, Nroll, Ndatasims, Nmcsims)
        print(f'I average {Ndatasims} sims with rdn0 to get effective QE response')

        if not cacher.is_cached(fn):
            rdn0 = np.zeros(self.lmax_qlm+1)
            for idx in np.arange(Ndatasims):
                rdn0 += self.get_rdn0_qe(idx, Ndatasims, Nmcsims, Nroll)[0]
            rdn0 /= Ndatasims
            cacher.cache(fn, rdn0)
        rdn0 = cacher.load(fn)

        if rfid is None:
            r_gg_fid  = self.get_qe_resp(resp_gradcls=True)
        else:
            r_gg_fid = rfid
        Reff_Spline = np.ones(self.lmax_qlm+1) * r_gg_fid
        if lmax_interp is None: lmax_interp=self.lmax_qlm
        ells = np.arange(lmin_interp, lmax_interp+1)
        if dospline:
            Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] /self.fsky * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        else:
            Reff_Spline = rdn0 /self.fsky
        return Reff_Spline

    def load_rdn0_map(self, idx, itr, mcs, Nroll, tol=5., rdn0tol=5., recache = False):
        """Load previously computed RDN0 estimate.
    
        See the file rdn0_cs.py to compute the RDN0 for a given paramfile and simidx.

        Args:
            idx: index of simulation

        Returns:
            rdn0: unormalized realisation dependent bias (phi-based spectum)
            pds: data x sim QE estimates 
            pss: sim x sim QE estimates 
        """
        outputdir = rdn0_cs.output_sim(self.k, self.param.suffix,  self.version, idx)
        fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll, rdn0tol))
        print(fn)
        if not os.path.exists(fn) or recache:
            # assert 0, f'Run the rdn0_cs script to get RDN0 for map idx {idx}' 
            itlibdir = self.param.libdir_iterators(self.k, idx, self.version, tol)
            ss_dict =  rdn0_cs._ss_dict(mcs, Nroll)
            rdn0_cs.export_dsss(itr, self.k, itlibdir, self.param.suffix, idx, self.version, ss_dict, mcs, Nroll)
        pds, pss, _, _, _, _ = np.loadtxt(fn).transpose()
        pds *= pp2kk(np.arange(len(pds))) * 1e7 
        pss *= pp2kk(np.arange(len(pss))) * 1e7 
        rdn0 = 4 * pds - 2 * pss
        return rdn0, pds, pss

    def get_rdn0_map(self, idx, itr, itmax_fid=15, normalize=True, rdsims=None, mcs=np.arange(0, 96), Nroll=8, rdn0tol=5., useReff=True,  Reff=None, lmin_interp=0, lmax_interp=None, k=3, s=None, version='', recache=True):
        """Get the normalized iterative RDN0 estimate        
        
        Args:
            idx: index of simulation
            itr: Iteration of teh MAP estimate
            itmax_fid: iteration for the MAP fiducial response

        Returns:
            RDN0: Normalized realisation dependent bias of Cpp MAP
        """
        print(rdn0tol)
        rdn0, _, _ = self.load_rdn0_map(idx, itr, mcs, Nroll, tol=5., rdn0tol=rdn0tol, recache=recache)
        # assert self.itmax == 15, "Need to check if the exported RDN0 correspond to the same iteration as the Cpp MAP" 
        #FIXME  maybe not relevant if everything is converged ?
        RDN0 = rdn0[:self.lmax_qlm+1]
        
        if normalize:
            if Reff is None and useReff:
                Reff = self.get_R_eff(rfid=None, lmin_interp=lmin_interp, lmax_interp=lmax_interp, k=k, s=s, itr=itr, rdsims=rdsims, mcs=mcs, Nroll=Nroll)
            if useReff:
                RDN0 *= utils.cli(Reff[:self.lmax_qlm+1])**2
            else:
                r_gg_fid = self.get_map_resp(itmax_fid, version=version)
                RDN0 *= utils.cli(r_gg_fid[:self.lmax_qlm+1])**2
        return RDN0 / self.fsky
        
    def get_semi_rdn0_qe(self, simidx):
        """Returns semi analytical realisation-dependent N0 lensing bias

            Args:
                simidx: index of simulation 
            Returns:
                Semi-analytical un-normalized RDN0 
        """
        
        return self.nhllib.get_sim_nhl(simidx, self.k,  self.k)    

    def get_mcn0_qe(self, Ndatasims=40, Nmcsims=100, Nroll=10, use_parfile=False, qcls_ss = None, use_old_files=False):
        """Returns unnormalised MC-N0 for the QE.
        Be careful to use sims with no overlap with the sims that are used as "data"
        i.e, we need to define the sims that are used for data, comprised bewteen idx=0 and idx=Ndatasims-1
        and the sims between idx=Ndatasims and idx=Ndatasims + Nmcsims -1 will be used to get the MC-N0 estimate.

        Args:
            Ndatasims: sims with index between 0 and Ndatasims-1 are not considered to get the rdn0
            Nmcsims: sims with index between Ndatasims and Ndatasims + Nmcsims -1 are used to get the rdno
            Nroll: the allocation of i, j sims is done with j = i+1, by batches of Nroll 
            use_parfile: Ignore custom settings and use the definition of the MC sims in the parameter file (see qlms_ss library)
        """
        # mcn0 = 2* self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, mcs, k2=self.k2).mean()

        if use_old_files:
            mcn0 = rdn0_cs.get_mcn0_qe(self.param, self.k, Ndatasims=Ndatasims, Nmcsims=Nmcsims, Nroll=Nroll, use_parfile=use_parfile)
            lmax = len(mcn0)-1
            pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
            return mcn0 * pp2kk
        if qcls_ss is None:
            qcls_ss = self.qcls_ss

        print(f'Using qcl library in {qcls_ss.lib_dir}')
        mcs = self.param.mc_sims_var
        ss = qcls_ss.get_sim_stats_qcl(self.k, mcs).mean()
        return 2*ss

    def get_rdn0_qe(self, datidx, Ndatasims=40, Nmcsims=100, Nroll=10):
        """Returns unnormalised realization-dependent N0 lensing bias RDN0.
        To get the RDN0, we use sims with no overlap with the sims that are used as "data"
        i.e, we need to define the sims that are used for data, comprised bewteen idx=0 and idx=Ndatasims-1
        and the sims between idx=Ndatasims and idx=Ndatasims + Nmcsims -1 will be used to get the RDN0 estimate.

        Args:
            datidx: index of simulation 
            Ndatasims: sims with index between 0 and Ndatasims-1 are not considered to get the rdn0
            Nmcsims: sims with index between Ndatasims and Ndatasims + Nmcsims -1 are used to get the rdno
            Nroll: the allocation of i, j sims is done with j = i+1, by batches of Nroll 

        """

        rdn0, ds, ss = rdn0_cs.get_rdn0_qe(self.param, datidx, self.k,  Ndatasims, Nmcsims, Nroll, version=self.version)
        lmax = len(rdn0)-1
        pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
        return rdn0 * pp2kk, ds*pp2kk, ss*pp2kk

    def get_mcn1_qe(self, Ndatasims=40, Nmcsims=100, Nroll=10):
        """Returns unnormalized estimates of the QE MC-N1
            MC-N1 = < 2 C_L(i, i') - 2 C_L(i, j)>
            With i, j two different sims and i and i' two sims with same lensing field but different CMB.

        """

        ss_n1 = rdn0_cs.get_mcn1_qe(self.param, self.k, Ndatasims, Nmcsims, Nroll, self.version)
        lmax = len(ss_n1)-1
        pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
        ss_n1 *= pp2kk
        _, _, ss = self.get_rdn0_qe(0, Ndatasims, Nmcsims, Nroll)
        return 2*ss_n1 - 2*ss

    def get_nsims_itmax(self, itmax):
        """Return the number of simulations reconstructed up to itmax"""
        nsim = 0
        while statics.rec.maxiterdone(self.libdir_sim(nsim)) >= itmax:
            nsim+=1
        return nsim

    def get_idx_sims_done(self, itmax=15):
        isdone = [False]*5000
        for i in range(5000):
            if self.maxiterdone(i) ==itmax:
                isdone[i] = True
        return isdone

    def maxiterdone(self, simidx):
        return statics.rec.maxiterdone(self.libdir_sim(simidx))

    def get_gauss_cov(self, version='', w=lambda ls : 1.,  edges=None, withN1=False, cosmicvar=True, QE_iter0=True):
        N0_map, N1_map, map_resp, _ = self.get_N0_N1_iter(15, version=version)
        if QE_iter0:
            # Takes the QE as the iteration 0 of iterative N0 and N1 (faster as N1 is from fft calc)
            # N0 is identical at 0.1 %, N1 at 10% compared to the Planck get_nhl and get_n1
            N0_qe, N1_qe, _, _= self.get_N0_N1_iter(0, version=version)
        else:
            N0_qe, N1_qe = self.get_N0_N1_QE(normalize=True)
        
        cov_qe =  1./(2.*np.arange(self.lmax_qlm+1) +1.)  / self.fsky * 2 * ((self.cpp_fid[:self.lmax_qlm+1]*cosmicvar + N0_qe[:self.lmax_qlm+1] + N1_qe[:self.lmax_qlm+1]*withN1) * w(np.arange(self.lmax_qlm+1) +1))**2 
        cov_map =1./(2.*np.arange(self.lmax_qlm+1) +1.) / self.fsky * 2 * ((self.cpp_fid[:self.lmax_qlm+1]*cosmicvar + N0_map[:self.lmax_qlm+1] + N1_map[:self.lmax_qlm+1]*withN1) * w(np.arange(self.lmax_qlm+1) +1))**2 
        
        if edges is not None:
            nbins = len(edges) - 1
            cov_qe_b = np.zeros([nbins, nbins])
            cov_map_b = np.zeros([nbins, nbins])
            
            for i in range(nbins):
                bins_l = edges[i]
                bins_u = edges[i+1]
                ells = np.arange(self.lmax_qlm+1)
                ii = np.where((ells >= bins_l) & (ells < bins_u))[0]
                cov_qe_b[i, i] = np.sum(cov_qe[ells[ii]]) / len(ii)**2
                cov_map_b[i, i] = np.sum(cov_map[ells[ii]])  / len(ii)**2
            return np.diag(cov_qe_b), np.diag(cov_map_b)

        else:
            return cov_qe, cov_map

    def get_dcpp_qe_map(self, cppsim_shuffle, wf_eff, qe_resp_fid, Nsims, mf_sims, itr:int, plm_shuffle= None, do_bin=False, nbin:int=None, ellb=None, do_spline=False, edges=None, lmin=None, lmax=None, k=3, s=None, recache=False):
        """Get the Cpp bias by taking the power spectrum of the lensing 
        reconstrutcion with same lensing field but different CMB.

        Args: 
            cppsim_shuffle: cpp_sims_lib with shuffled indices for the lensing potential field, but same CMB fields. 
        """

        if plm_shuffle is None:
            plm_shuffle = cppsim_shuffle.param.plm_shuffle
        Nmf = len(mf_sims)
        # mf_sims_1 = mf_sims[:int(Nmf/2)]
        # mf_sims_2 = mf_sims[int(Nmf/2):]

        mf_sims_1 =  np.unique(mf_sims[::2])
        mf_sims_2 =  np.unique(mf_sims[1::2])
        # print(mf_sims_1)
        # print(mf_sims_2)
        
        if lmax is None: lmax = self.lmax_qlm
        if lmin is None: lmin = 2
        
        if do_bin:
            dcpp_10 = stats(nbin, xcoord=ellb, docov=True)
            dcpp_10_qe = stats(nbin, xcoord=ellb, docov=True)
            
        else:  
            dcpp_10 = stats(lmax+1, xcoord=np.arange(lmax+1), docov=False)
            dcpp_10_qe = stats(lmax+1, xcoord=np.arange(lmax+1), docov=False)
        
        for idx in range(Nsims): 
            cacher = cppsim_shuffle.cacher_sim(idx)
            fn_cpp_map_cross = "cpp_map_shufle_cross"
            fn_cpp_qe_cross = "cpp_qe_shufle_cross"
            
            if np.any([not cacher.is_cached(fn) for fn in [fn_cpp_map_cross, fn_cpp_qe_cross]]) or recache:
                print(f'Sim idx {idx}, sim shuffle {plm_shuffle(idx)}')

                plm1 = cppsim_shuffle.get_plm(idx, itr)
                plm0 = self.get_plm(plm_shuffle(idx), itr)

                plm_mf1 = self.get_mf(itr, mf_sims_1, simidx=[idx, plm_shuffle(idx)], use_cache=True, verbose=False)
                plm_mf2 = self.get_mf(itr, mf_sims_2, simidx=[idx, plm_shuffle(idx)], use_cache=True, verbose=False)

                plmqe1 = cppsim_shuffle.get_plm_qe(idx)
                plmqe0 = self.get_plm_qe(plm_shuffle(idx))

                Nmf_qe = len(self.param.mc_sims_mf_it0)
                qe_mf_sims_1 = np.unique(self.param.mc_sims_mf_it0[:int(Nmf_qe/2)])
                qe_mf_sims_2 = np.unique(self.param.mc_sims_mf_it0[int(Nmf_qe/2):])
                
                plm_mf1_qe = self.get_mf0([idx, plm_shuffle(idx)], mf_sims=qe_mf_sims_1, version=self.version)
                plm_mf2_qe = self.get_mf0([idx, plm_shuffle(idx)], mf_sims=qe_mf_sims_2, version=self.version)

                _cpp10 = hp.alm2cl(plm1-plm_mf1, plm0-plm_mf2)/self.fsky
                _cpp10_qe = hp.alm2cl(plmqe1-plm_mf1_qe, plmqe0-plm_mf2_qe)/self.fsky
            
                cacher.cache(fn_cpp_map_cross, _cpp10)
                cacher.cache(fn_cpp_qe_cross, _cpp10_qe)
                
            _cpp10 = cacher.load(fn_cpp_map_cross)
            _cpp10_qe = cacher.load(fn_cpp_qe_cross)
            _cpp_in = cppsim_shuffle.get_cpp_input(idx)
            
            if do_bin:
                dcpp_10.add(bnd(_cpp10*cli(wf_eff**2)*cli(_cpp_in) - 1, lmin, lmax, edges)[1])  
                dcpp_10_qe.add(bnd(_cpp10_qe*cli(qe_resp_fid**2)*cli(_cpp_in) - 1, lmin, lmax, edges)[1])
            else:
                dcpp_10.add(_cpp10*cli(wf_eff**2)*cli(_cpp_in) - 1)
                dcpp_10_qe.add(_cpp10_qe*cli(qe_resp_fid**2)*cli(_cpp_in) - 1)
            
        if do_spline:
            ls = np.arange(lmin, lmax+1)
            dqe = spline(ls, dcpp_10_qe.mean()[ls], k=k, s=s)(np.arange(lmax+1))
            dmap = spline(ls, dcpp_10.mean()[ls], k=k, s=s)(np.arange(lmax+1))
            return dqe, dmap
        
        return dcpp_10_qe.mean(), dcpp_10.mean()

    def get_delta_cpp(self, itr:int, lmin:int, lmax:int, edges:np.ndarray, wf_it:np.ndarray=None, Resp:np.ndarray=None, n1:np.ndarray=None, resp_n1:np.ndarray=None,  mcs:np.ndarray=np.arange(0, 40), mf_sims:np.ndarray=np.arange(0, 40)):
        """Get the bias Delta Cpp / Cpp from a set of simulations
        We compute the bias (C_L^{\hat \phi, \hat \phi} - RDN0 - N1) / C_L^{\phi_{in}, \phi_{in}} - 1
        
        Args: 
            itr: iteration index of the MAP estimator
            wf_it: Wiener filter for the MAP normalisation
            lmin: minimal L for the binning
            lmax: maximal L for the binning
            edges: Bin edges
            Resp: Effective response
            n1: Unnormalised N1
            mcs: indexes of simulation to use for bias estimate
            mf_sims: indexes of simulations to use for the mean field estimate


        Returns:
            delta_cpp: a planclens.utils.stats objects containing mean and variance of the bias  
        """ 

        if Resp is None:
            if itr == 0:
                # QE normalisation
                Resp = self.get_qe_resp(resp_gradcls=True)
            else:
                # MAP normalisation
                _, _, Resp, _ = self.get_N0_N1_iter(15, version='')

        if resp_n1 is None:
                _, _, resp_n1, _ = self.get_N0_N1_iter(15, version='')

        if edges is None:
            delta_cpp = stats(self.lmax_qlm+1, xcoord=np.arange(self.lmax_qlm+1), docov=True)       
        else:
            nbin = len(edges)-1
            bl = edges[:-1];bu = edges[1:]
            ellb = 0.5 * bl + 0.5 * bu
            delta_cpp = stats(nbin, xcoord=ellb, docov=True)
            
        for idx in mcs:
            cpp_input = self.get_cpp_input(idx)

            if itr == 0:
                cpp_qe = self.get_cpp_qe_raw(idx, splitMF=True, recache=False)          
                rdn0_qe = self.get_semi_rdn0_qe(idx, normalize=False)
                dcpp = ((cpp_qe/self.fsky - rdn0_qe - n1) *utils.cli(Resp) **2 )*utils.cli(cpp_input) -1
            
            else:
                cpp_map = self.get_cpp(idx, itr, sub_mf=True, mf_sims=mf_sims, splitMF=True)
                RDN0_map  = self.get_rdn0_map(idx, itr =itr, Reff=Resp,  useReff=True)
                dcpp = (cpp_map*utils.cli(wf_it)**2 /self.fsky - RDN0_map - n1*utils.cli(resp_n1**2)) * utils.cli(cpp_input) - 1

            if edges is None:
                delta_cpp.add(dcpp)
            else:
                delta_cpp.add(bnd(dcpp,  lmin, lmax, edges)[1])

        return delta_cpp


    def get_sim_cov(self, mcs, edges=None, w= lambda ls: 1, use_rdn0=True):
        """Get Clpp covariance matrix from a set of simulations
        
        Args: 
            mcs: indexes of simulation to use for covariance estimate
            edges: Bin edges
            w: Weighting of the Cls
            use_rdn0: Debias using rdn0 instead of analytical rdn0
            
        """

        return 0
