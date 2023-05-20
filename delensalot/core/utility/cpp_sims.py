from termios import N_STRIP
import numpy as np
from delensalot import utils
from delensalot.utils import cli
from delensalot.iterators import statics
from plancklens.sims import planck2018_sims
from delensalot.core import cachers
from delensalot import utils_scarf
from plancklens.utils import mchash, cls_dot, stats
from plancklens import qresp, n0s, nhl
from plancklens.n1 import n1

from scipy.interpolate import UnivariateSpline as spline

from os.path import join as opj
import os
import importlib
from delensalot.core.utility.utils_hp import alm_copy
from delensalot.utils import read_map
from delensalot.core.utility.utils_plot import pp2kk, bnd
from delensalot.biases import rdn0_cs
from delensalot.core.utility import utils_hp as uhp
from delensalot.biases import n0n1_iterative


import healpy as hp


_write_alm = lambda fn, alm : hp.write_alm(fn, alm, overwrite=True)
 
class cpp_sims_lib:
    def __init__(self, k, v='', param_file='cmbs4wide_planckmask', label='', module='delensalot'):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.
        """
        
        # Load the parameters defined in the param_file
        self.param_file = param_file
        if 'n32' in module:
            import n32
            self.param = importlib.import_module('n32.params.'+param_file)
        else:
            self.param = importlib.import_module('delensalot.params.'+param_file)

        self.k = k
        # self.itmax = itmax
        # self.tol = tol
        # self.tol_iter  = 10 ** (- self.tol) 
        # self.imin = imin
        # self.imax = imax
        self.version = v
        # self.iters = np.arange(itmax+1)
        self.label = label
        self.TEMP =  self.param.TEMP
        self.lmax_qlm = self.param.lmax_qlm
        self.mmax_qlm = self.param.mmax_qlm

        # self.plms = [[None]*(itmax+1)]*(imax+1)
        self.cacher_param = cachers.cacher_npy(opj(self.TEMP, 'cpplib'))
        self.fsky = self.get_fsky() 

        # Cl wieghts used in the QE (either lensed Cls or grad Cls)
        try:
            self.cls_weights = self.param.ivfs.cl
        except AttributeError:
            self.cls_weights = self.param.ivfs.ivfs.cl

        # Grad cls used for CMB response
        self.cls_grad = self.param.cls_grad

        self.cpp_fid = self.param.cls_unl['pp']

        self.config = (self.param.nlev_t, self.param.nlev_p, self.param.beam, 
                       (self.param.lmin_tlm,  self.param.lmin_elm, self.param.lmin_blm), 
                       self.param.lmax_ivf, self.param.lmax_qlm)


        # if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
        #     self.sims_unl = planck2018_sims.cmb_unl_ffp10() 
        # elif type(self.param.sims.sims_cmb_len).__name__ == 'sims_postborn':
        #     self.sims_unl = self.param.sims.sims_cmb_len
        # else:
        #     assert 0, "I do not know what are the unlensed sims"

    def libdir_sim(self, simidx):
        return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)


    def get_itlib_sim(self, simidx, tol):
        tol_iter  = 10 ** (- tol) 
        return self.param.get_itlib(self.k, simidx, self.version, tol_iter)


    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)


    def get_plm(self, simidx, itr, use_cache=True):
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_it{itr:03.0f}"
            if not cacher.is_cached(fn):
                plm = statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
                cacher.cache(fn, plm)
            plm = cacher.load(fn)
            return plm
        else:
            return statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]

    def get_plm_qe(self, simidx, use_cache=True):
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_qe"
            if not cacher.is_cached(fn):                
                plm = self.param.qlms_dd.get_sim_qlm(self.k, int(simidx)) 
                cacher.cache(fn, plm)
            plm = cacher.load(fn)
            return plm
        else:
            return self.param.qlms_dd.get_sim_qlm(self.k, int(simidx)) 

    def get_sim_plm(self, idx):
        """Returns the simulated plm, depening if it is a sims_ffp10 or a new sim"""
        if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            return planck2018_sims.cmb_unl_ffp10().get_sim_plm(idx)
        else:
            return self.param.sims.sims_cmb_len.get_sim_plm(idx)

    def get_plm_input(self, simidx, use_cache=True, recache=False):
    
        if (not hasattr(self.param.sims.sims_cmb_len, 'plm_shuffle')) or self.param.sims.sims_cmb_len.plm_shuffle is None:
            shuffled_idx = simidx
        else:
            shuffled_idx = self.param.sims.sims_cmb_len.plm_shuffle(simidx)

        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_input"
            if not cacher.is_cached(fn) or recache:
                plm_in = alm_copy(self.get_sim_plm(shuffled_idx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)
                cacher.cache(fn, plm_in)
            plm_in = cacher.load(fn)
            return plm_in
        else:
            return alm_copy(self.get_sim_plm(shuffled_idx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)


    def get_eblm_dat(self, simidx, lmaxout=1024):
        QU_maps = self.param.sims_MAP.get_sim_pmap(simidx)
        tr = int(os.environ.get('OMP_NUM_THREADS', 8))
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

    def get_cl(self, alm, blm=None):
        if blm is None: blm = alm 
        return uhp.alm2cl(alm, blm, self.param.lmax_qlm, self.param.mmax_qlm, None)

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


    def abcef(self, simidx, itr):
        fn = 'cpp_in_x_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            plmin = self.get_plm_input(simidx)
            # plmit = self.plms[simidx][itr]
            plmit = self.get_plm(simidx, itr)
            cpp_itXin = self.get_cl(plmit, plmin)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin
    

    def get_cpp_qeXinput(self, simidx):
        fn = 'cpp_in_x_qe'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            plmin = self.get_plm_input(simidx)
            plmqe = self.get_plm_qe(simidx)
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
        # if sub_mf:
        #     fn_cpp_it = 'cpp_submf_it_{}'.format(itr)
        # else:
        #     fn_cpp_it = 'cpp_it_{}'.format(itr)
        fn_cpp_it = 'cpp_it_{}'.format(itr) + sub_mf*"_submf" + sub_mf*splitMF*"_split_mf"
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it) or recache == True:
            plm = self.get_plm(simidx, itr)
            if sub_mf:
                assert mf_sims is not None, "Please provide mf_sims"
                if splitMF:
                    Nmf = len(mf_sims)
                    mf_sims_1 = mf_sims[:int(Nmf/2)]
                    mf_sims_2 =  mf_sims[int(Nmf/2):]
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


    def get_cpp_qe(self, simidx, qeresp=None, splitMF=True, recache=False):
        """Get nomalized Cpp QE nomalized

            Args: 
                simidx: index of sim to consider
                qeresp: Response of the QE, if None use the default one given by get_qe_resp
        """
        if qeresp is None:
            qeresp = self.get_qe_resp(recache=recache)
        cppqe = self.get_cpp_qe_raw(simidx, splitMF, recache) * utils.cli(qeresp)**2
        return cppqe

    def get_cpp_qe_raw(self, simidx, splitMF=True, recache=False):
        """Returns unromalized Cpp QE"""

        fn_cpp_qe = 'cpp_qe_raw' + splitMF*'_splitMF'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_qe) or recache:
            plmqe  = self.param.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate

            # QE mean-field
            if splitMF:
                Nmf = len(self.param.mc_sims_mf_it0)
                mf_sims_1 =  np.unique(self.param.mc_sims_mf_it0[:int(Nmf/2)])
                mf_sims_2 =  np.unique(self.param.mc_sims_mf_it0[int(Nmf/2):])
                mf0_1 = self.get_mf0(simidx, mf_sims=mf_sims_1)
                mf0_2 = self.get_mf0(simidx, mf_sims=mf_sims_2)
                cppqe = self.get_cl(plmqe - mf0_1, plmqe - mf0_2)
            else:
                mf0 = self.get_mf0(simidx)
                plmqe -= mf0  # MF-subtracted unnormalized QE
            cacher.cache(fn_cpp_qe, cppqe)
        return cacher.load(fn_cpp_qe)


    # def get_cpp_qe(self, simidx, recache=False):
    #     fn_cpp_qe = 'cpp_qe'
    #     cacher = self.cacher_sim(simidx)
    #     R = self.get_qe_resp()
    #     if not cacher.is_cached(fn_cpp_qe) or recache:
    #         cpp = self.param.cls_unl['pp'][:self.lmax_qlm + 1]
    #         WF = cpp * utils.cli(cpp + utils.cli(R))
    #         cpp_qe_wf = self.get_cpp(simidx, 0)
    #         cpp_qe = cpp_qe_wf * utils.cli(WF)**2
    #         cacher.cache(fn_cpp_qe, cpp_qe)
    #     cpp_qe = cacher.load(fn_cpp_qe)
    #     return cpp_qe, R


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


    def get_mf0(self, simidx, mf_sims=None):
        """Get the QE mean-field"""
        # return np.load(opj(self.libdir_sim(simidx), 'mf.npy'))
        if mf_sims is None:
            mf_sims = np.unique(self.param.mc_sims_mf_it0 if not 'noMF' in self.version else np.array([]))
        else:
            mf_sims = np.unique(mf_sims)
        mf0 = self.param.qlms_dd.get_sim_qlm_mf(self.k, mf_sims)  # Mean-field of the QE
        # print(len(mf_sims))
        
        
        # mc_sims_less = mf_sims

        Nmf = len(mf_sims)
        for simid in np.atleast_1d(simidx):
            if simid in mf_sims:
                mf0 = (mf0 - self.param.qlms_dd.get_sim_qlm(self.k, int(simid)) / Nmf) * (Nmf / (Nmf - 1))
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
        fn =  f'simMF_itr{itmax}_k_{mchash(mc_sims)}.fits'
        if not cacher.is_cached(fn) or recache:
            MF = np.zeros(hp.Alm.getsize(self.lmax_qlm), dtype=complex)
            if len(this_mcs) == 0: return MF
            for i, idx in utils.enumerate_progress(this_mcs, label='calculating MF'):
                MF += self.get_plm(idx, itmax, use_cache=use_cache)
            MF = MF / len(this_mcs)
            # _write_alm(fname, MF)
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
        fn_resp_qe = 'resp_qe_{}'.format(self.k) + self.version
        if resp_gradcls: 
            fn_resp_qe += '_gradcls'
        cacher = self.cacher_param
        if not cacher.is_cached(fn_resp_qe):
            R = qresp.get_response(self.k, self.param.lmax_ivf, 'p', self.cls_weights, self.cls_grad, {'e': self.param.fel, 'b': self.param.fbl, 't':self.param.ftl}, lmax_qlm=self.param.lmax_qlm)[0]
            cacher.cache(fn_resp_qe, R)
        R = cacher.load(fn_resp_qe)
        # iterbiases = n0n1_iterative.polMAPbiases(self.config, fidcls_unl=self.param.cls_unl, itrmax = 0, cacher=self.cacher_param)
        # N0_biased, N1_biased_spl, R, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=self.version)  
        return R

    def get_map_resp(self, it=15, version=''):
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=it, version=version)
        return r_gg_fid

    def get_N0_N1_QE(self, normalize=True, resp_gradcls=True):
        # iterbiases = n0n1_iterative.polMAPbiases(self.config, fidcls_unl=self.param.cls_unl, itrmax = 0, cacher=self.cacher_param)
        # N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=version)
        fn_n0 = 'n0_qe_{}'.format(self.k)


        resp_qe = self.get_qe_resp(resp_gradcls)
        # N0_qe = n0s.get_N0(self.parm.beam, self.param.nlev_t, self.param.nlev_p, self.param.lmax_ivf, self.param.lmin_elm, self.param.lmax_qlm, cls_resp, self.cls_weights, joint_TP=False, ksource='p')

        # Isotropic approximation of the filtering, 1/ (C+ N)
        fal_sepTP = {
            'tt': self.param.ivfs.get_ftl(),
            'ee': self.param.ivfs.get_fel(),
            'bb': self.param.ivfs.get_fbl()}
            # 'te': np.copy(self.cls_len['te'][:self.lmax_ivf + 1])}

        cls_dat = {spec: utils.cli(fal_sepTP[spec]) for spec in ['tt', 'ee', 'bb']}
        # Spectra of the inverse-variance filtered maps
        # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
        cls_ivfs_sepTP = cls_dot([fal_sepTP, cls_dat, fal_sepTP], ret_dict=True)

        if not self.cacher_param.is_cached(fn_n0):
            NG, NC, NGC, NCG = nhl.get_nhl(self.k, self.k, self.cls_weights, cls_ivfs_sepTP, self.param.lmax_ivf, self.param.lmax_ivf,
                                    lmax_out=self.lmax_qlm)
            self.cacher_param.cache(fn_n0, NG)
        NG = self.cacher_param.load(fn_n0)
        
        n1lib = n1.library_n1(self.cacher_param.lib_dir, self.cls_weights['tt'], self.cls_weights['te'], self.cls_weights['ee'], self.lmax_qlm)

        _n1 = n1lib.get_n1(self.k, 'p',  self.param.cls_unl['pp'], fal_sepTP['tt'], fal_sepTP['ee'], fal_sepTP['bb'], Lmax=self.lmax_qlm)

        if normalize is False:
            return NG, _n1
        else:
            resp_qe = self.get_qe_resp(resp_gradcls)
            return NG*utils.cli(resp_qe)**2, _n1*utils.cli(resp_qe)**2

        # return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true

    def get_N0_N1_iter(self, itermax=15, version='', recache=False):
        assert self.k =='p_p', 'Iterative biases not implemented fot MV and TT estimators'       
        iterbiases = n0n1_iterative.polMAPbiases(self.config, fidcls_unl=self.param.cls_unl, itrmax = itermax, cacher=self.cacher_param)
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=version, recache=recache)
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true


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
        """Get the Wiener from the simulations.

        :math:`\hat \mathcal{W} = \frac{C_L{\phi^{\rm MAP} \phi{\rm in}}}{C_L{\phi^{\rm in} \phi{\rm in}}}`
        
        """
        fn = 'wf_sim_it{}'.format(itr) if mf is False else 'wf_sim_it{}_mfsub'.format(itr)
        # print(fn)
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
        # if itmax_sims is None: itmax_sims = self.itmax
        fn_weff = f"wf_eff_{itmax_sims}_{itmax_fid}_{mf}_{version}_{do_spline}_{lmin_interp}_{lmax_interp}_{k}_{s}"
        fn_wfspline = f"wf_spline_{itmax_sims}_{itmax_fid}_{mf}_{version}_{do_spline}_{lmin_interp}_{lmax_interp}_{k}_{s}"
        if np.any([not self.cacher_param.is_cached(fn) for fn in [fn_weff, fn_wfspline]]) or recache:
            wf_fid = self.get_wf_fid(itmax_fid, version=version)
            nsims = self.get_nsims_itmax(itmax_sims)
            # sims_idx = self.get_idx_sims_done(itmax=15)
            print(f'I use {nsims} sims to estimate the effective WF')
            print(fn_weff)
            wfsims_bias = np.zeros([nsims, len(wf_fid)])
        #     for i, f in enumerate(dat_files):
        #         _, ckk_in[f], ckk_cross[f] = np.loadtxt(os.path.join(DIR, f, 'MAP_cls.dat')).transpose()
        #         wfcorr_full[i] =  ckk_cross[f] *cli(ckk_in[f] * wfpred) 
            for isim in range(nsims):
                if verbose: print(f'wf eff {isim}/{nsims}')
                wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims, mf=mf, mc_sims=mc_sims, recache=recache) * utils.cli(wf_fid)
                # wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims) * utils.cli(wf_fid)
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
        fn =  'rdn0_mean_{}_Nroll{}_{}_{}'.format(itr, Nroll, mchash(rdsims), mchash(mcs))
        # fn =  f'Reff_{itr}_Nroll{Nroll}_{mchash(rdsims)}_{mchash(mcs)}_{}'

        if not cacher.is_cached(fn):
            # rdn0s = []
            # idx = 0 
            # rdn0_computed = True
            # while rdn0_computed:      
            #     # outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
            #     # fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
            #     try:
            #         # kdsfid, kssfid, _, _, _, _ = np.loadtxt(fn).transpose()
            #         # rdn0s.append(4 * kdsfid - 2 * kssfid)
            #         rdn0, pds, pss = self.load_rdn0_map(idx, itr, mcs, Nroll)
            #         rdn0s.append(rdn0)
            #         idx +=1
            #     except OSError:
            #          rdn0_computed = False
            # print(f'I average {idx} sims with rdn0 to get effective response')
            # rdn0s = np.array(rdn0s)
            # rdn0_mean = np.mean(rdn0s, axis=0)
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
        # return 1.
        cacher = cachers.cacher_npy(self.cacher_param.lib_dir)

        fn =  'rdn0_mean_qe_Nroll{}_Ndatasims{}_Nmcsims{}'.format(Nroll, Ndatasims, Nmcsims)
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


    def load_rdn0_map(self, idx, itr, mcs, Nroll):
        """Load previously computed RDN0 estimate.
    
        See the file rdn0_cs.py to compute the RDN0 for a given paramfile and simidx.

        Args:
            idx: index of simulation

        Returns:
            rdn0: unormalized realisation dependent bias (phi-based spectum)
            pds: data x sim QE estimates 
            pss: sim x sim QE estimates 
        """
        outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
        fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
        if not os.path.exists(fn):
            itlibdir = self.param.libdir_iterators(self.k, idx, self.version)
            ss_dict =  rdn0_cs._ss_dict(mcs, Nroll)
            rdn0_cs.export_dsss(itr, self.k, itlibdir, self.param.suffix, idx, ss_dict, mcs, Nroll)
        pds, pss, _, _, _, _ = np.loadtxt(fn).transpose()
        pds *= pp2kk(np.arange(len(pds))) * 1e7 
        pss *= pp2kk(np.arange(len(pss))) * 1e7 
        rdn0 = 4 * pds - 2 * pss
        return rdn0, pds, pss

    def get_rdn0_map(self, idx, itr, itmax_fid=15,  rdsims=None, mcs=np.arange(0, 96), Nroll=8, useReff=True,  Reff=None, lmin_interp=0, lmax_interp=None, k=3, s=None, version=''):
        """Get the normalized iterative RDN0 estimate        
        
        Args:
            idx: index of simulation
            itr: Iteration of teh MAP estimate
            itmax_fid: iteration for the MAP fiducial response

        Returns:
            RDN0: Normalized realisation dependent bias of Cpp MAP
        """
        rdn0, _, _ = self.load_rdn0_map(idx, itr, mcs, Nroll)
        # assert self.itmax == 15, "Need to check if the exported RDN0 correspond to the same iteration as the Cpp MAP" 
        # Fixme  maybe not relevant if everything is converged ?
        RDN0 = rdn0[:self.lmax_qlm+1]
        
        if Reff is None and useReff:
            Reff = self.get_R_eff(rfid=None, lmin_interp=lmin_interp, lmax_interp=lmax_interp, k=k, s=s, itr=itr, rdsims=rdsims, mcs=mcs, Nroll=Nroll)
        if useReff:
            RDN0 *= utils.cli(Reff[:self.lmax_qlm+1])**2
        else:
            r_gg_fid = self.get_map_resp(itmax_fid, version=version)
            RDN0 *= utils.cli(r_gg_fid[:self.lmax_qlm+1])**2
        return RDN0 / self.fsky
        
    def get_semi_rdn0_qe(self, datidx, normalize=True, resp_gradcls=True):
        """Returns semi analytical realisation-dependent N0 lensing bias

            Args:
                datidx: index of simulation 
                resp: response to normallise the rdn0
                normalize: If False returns unormalized rdn0

        """
        fn_dir = rdn0_cs.output_sim(self.k, self.param.suffix, datidx)
        fn = os.path.join(fn_dir, 'QE_knhl.dat')
        if not os.path.exists(fn):
            print(fn)
            rdn0_cs.export_nhl(self.libdir_sim(datidx), self.k, self.param, datidx)
        GG = np.loadtxt(fn)
        GG *=  pp2kk(np.arange(len(GG))) * 1e7 
        if normalize is True:
            resp_qe = self.get_qe_resp(resp_gradcls=resp_gradcls)
            GG *= utils.cli(resp_qe[:self.lmax_qlm+1])**2
        return GG

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
        assert datidx < Ndatasims or datidx > Ndatasims+Nmcsims-1, "Do not estimate the RDN0 for a simulation inside the set of sims used for the RDN0"
        fn_dir = rdn0_cs.output_sim(self.k, self.param.suffix, datidx)
        mcs = np.arange(Ndatasims, Nmcsims+Ndatasims)
        fn = os.path.join(fn_dir, rdn0_cs.fn_cls_dsss(0, mcs, Nroll))
        # print(fn)
        if not os.path.exists(fn):
            print("Running RDN0 estimate for datidx {} with {} {} {}".format(datidx, Ndatasims, Nmcsims, Nroll))
            rdn0_cs.get_rdn0_qe(self.param, datidx, self.k,  Ndatasims, Nmcsims, Nroll, version=self.version)
            
        rdn0, ds, ss = np.loadtxt(fn).transpose()
        # rdn0 = np.loadtxt(fn).transpose()
        lmax = len(rdn0)-1
        pp2kk = 0.25 * np.arange(lmax + 1)** 2 * (np.arange(1, lmax + 2) ** 2) * 1e7
        return rdn0 * pp2kk, ds*pp2kk, ss*pp2kk
        # return rdn0 * pp2kk

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



    def get_gauss_cov(self, version='', w=lambda ls : 1.,  edges=None, withN1=False, cosmicvar=True):
        N0_map, N1_map, map_resp, _ = self.get_N0_N1_iter(15, version=version)
        N0_qe, N1_qe= self.get_N0_N1_QE(normalize=True)
        
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

    def get_dcpp_qe_map(self, cppsim_shuffle, wf_eff, qe_resp_fid, Nsims, mf_sims, itr:int, do_bin=False, nbin:int=None, ellb=None, do_spline=False, edges=None, lmin=None, lmax=None, k=3, s=None, recache=False):
        """Get the Cpp bias by taking the power spectrum of the lensing 
        reconstrutcion with same lensing field but different CMB.

        Args: 
            cppsim_shuffle: cpp_sims_lib with shuffled indices for the lensing potential field, but same CMB fields. 
        """

        plm_shuffle = cppsim_shuffle.param.plm_shuffle
        Nmf = len(mf_sims)
        mf_sims_1 = mf_sims[:int(Nmf/2)]
        mf_sims_2 = mf_sims[int(Nmf/2):]
        print(mf_sims_1)
        print(mf_sims_2)
        # N1_qe = n1_qe * utils.cli(qe_resp)**2
        
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
                print(idx, plm_shuffle(idx))

                plm1 = cppsim_shuffle.get_plm(idx, itr)
                plm0 = self.get_plm(plm_shuffle(idx), itr)
                # plm_in1 = cppsim_shuffle.get_plm_input(idx)

                # plm_mf1 = cppsim_shuffle.get_mf(itr, mf_sims_1, simidx=idx, use_cache=True, verbose=False)
                plm_mf1 = self.get_mf(itr, mf_sims_1, simidx=[idx, plm_shuffle(idx)], use_cache=True, verbose=False)
                plm_mf2 = self.get_mf(itr, mf_sims_2, simidx=[idx, plm_shuffle(idx)], use_cache=True, verbose=False)

                plmqe1 = cppsim_shuffle.get_plm_qe(idx)
                plmqe0 = self.get_plm_qe(plm_shuffle(idx))

                Nmf_qe = len(self.param.mc_sims_mf_it0)
                qe_mf_sims_1 = np.unique(self.param.mc_sims_mf_it0[:int(Nmf_qe/2)])
                qe_mf_sims_2 = np.unique(self.param.mc_sims_mf_it0[int(Nmf_qe/2):])
                
                plm_mf1_qe = self.get_mf0([idx, plm_shuffle(idx)], mf_sims=qe_mf_sims_1)
                plm_mf2_qe = self.get_mf0([idx, plm_shuffle(idx)], mf_sims=qe_mf_sims_2)
                # cppqe = self.get_cl(plmqe - mf0_1, plmqe - mf0_2)

                # plm_mf1_qe = self.get_mf0(plm_shuffle(idx), mf_sims=mf_sims_1)
                # plm_mf2_qe = self.get_mf0(plm_shuffle(idx), mf_sims=mf_sims_2)

                _cpp10 = hp.alm2cl(plm1-plm_mf1, plm0-plm_mf2)/self.fsky
                _cpp10_qe = hp.alm2cl(plmqe1-plm_mf1_qe, plmqe0-plm_mf2_qe)/self.fsky
            
                cacher.cache(fn_cpp_map_cross, _cpp10)
                cacher.cache(fn_cpp_qe_cross, _cpp10_qe)
                
            _cpp10 = cacher.load(fn_cpp_map_cross)
            _cpp10_qe = cacher.load(fn_cpp_qe_cross)
            _cpp_in = cppsim_shuffle.get_cpp_input(idx)
            # _cpp_in = cppsim.cpp_fid[:len(_cpp10)]
            
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
                # if self == idealpp_cstMF:
                rdn0_qe = self.get_semi_rdn0_qe(idx, normalize=False)
                # else:
                #     rdn0_qe = self.get_rdn0_qe(i, 40, 100, 10)
                dcpp = ((cpp_qe/self.fsky - rdn0_qe - n1) *utils.cli(Resp) **2 )*utils.cli(cpp_input) -1
            
            else:
                cpp_map = self.get_cpp(idx, itr, sub_mf=True, mf_sims=mf_sims, splitMF=True)
                RDN0_map  = self.get_rdn0_map(idx, itr =itr, Reff=Resp,  useReff=True)
                # dcpp = (cpp_map*utils.cli(wf_it)**2 /self.fsky - RDN0_map - n1*utils.cli(Resp**2)) * utils.cli(cpp_input) - 1
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

    # def get_mf_it(self, simidx, itr, tol, ret_alm=False, verbose=False):
    #     tol_iter  = 10 ** (- tol) 
    #     cacher = self.cacher_sim(simidx)
    #     fn_mf1 = 'mf1_it{}'.format(itr)
    #     fn_mf2 = 'mf2_it{}'.format(itr)
    #     if not cacher.is_cached(fn_mf1) or not cacher.is_cached(fn_mf2):
    #         if verbose:
    #             print('Starting to estimate MF for it {} from sim {}'.format(itr, simidx))
    #         itlib = self.get_itlib_sim(simidx)
    #         filtr = itlib.filter
    #         filtr.set_ffi(itlib._get_ffi(itr)) # load here the phi map you want 
    #         chain_descr = self.param.chain_descrs(self.param.lmax_unl, tol_iter)
    #         mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
    #         MF1 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
    #         MF2 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
    #         cacher.cache(fn_mf1, MF1)
    #         cacher.cache(fn_mf2, MF2)
    #     MF1 = cacher.load(fn_mf1)
    #     MF2 = cacher.load(fn_mf2)
    #     fn_mf = 'cpp_mf_it{}'.format(itr)
    #     fn_mf_mc = 'cpp_mf_mcnoise_it{}'.format(itr)
    #     if not cacher.is_cached(fn_mf) or not cacher.is_cached(fn_mf_mc):
    #         norm = self.param.qresp.get_response('p_p', self.param.lmax_ivf, 'p', self.param.cls_unl, self.param.cls_unl,  {'e': self.param.fel_unl, 'b': self.param.fbl_unl, 't':self.param.ftl_unl}, lmax_qlm=self.param.lmax_qlm)[0]
    #         cpp_mf = alm2cl(MF1[0], MF2[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
    #         cpp_mf_mc = alm2cl(MF1[0], MF1[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
    #         cacher.cache(fn_mf, cpp_mf)
    #         cacher.cache(fn_mf_mc, cpp_mf_mc)
    #     cpp_mf = cacher.load(fn_mf)
    #     cpp_mf_mc = cacher.load(fn_mf_mc)
    #     return (cpp_mf, cpp_mf_mc) if not ret_alm else (cpp_mf, cpp_mf_mc,MF1, MF2)