import numpy as np
from lenscarf import utils
from lenscarf.iterators import statics
from plancklens.sims import planck2018_sims
from plancklens import qresp
from lenscarf import cachers
from plancklens.qcinv import multigrid
from plancklens import nhl 

from os.path import join as opj
import os
import importlib
from lenscarf.utils_hp import alm2cl, alm_copy
from lenscarf.utils import read_map
from lenscarf.rdn0_cs import load_ss_ds


class cpp_sims_lib:
    def __init__(self, k, itmax, tol, v='', param_file='cmbs4wide_planckmask', label=''):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.
        """
        
        # Load the parameters defined in the param_file
        self.param_file = param_file
        self.param = importlib.import_module('lenscarf.params.'+param_file)

        self.k = k
        self.itmax = itmax
        self.tol = tol
        self.tol_iter  = 10 ** (- self.tol) 
        # self.imin = imin
        # self.imax = imax
        self.version = v
        self.iters = np.arange(itmax+1)
        self.label = label
        self.TEMP =  self.param.TEMP
        self.lmax_qlm = self.param.lmax_qlm
        self.mmax_qlm = self.param.mmax_qlm

        # self.plms = [[None]*(itmax+1)]*(imax+1)
        self.cacher_param = cachers.cacher_npy(opj(self.TEMP, 'cpplib'))
        self.fsky = self.get_fsky() 

        self.cpp_fid = self.param.cls_unl['pp']


        if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            self.sims_unl = planck2018_sims.cmb_unl_ffp10() 
    
    def libdir_sim(self, simidx):
        return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)

    def get_itlib_sim(self, simidx):
        return self.param.get_itlib(self.k, simidx, self.version, self.tol_iter)

    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)

    def get_plm(self, simidx, itr):
        # if self.plms[simidx][itr] is None:
        #     self.plms[simidx][itr] = statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
        return statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
    
    def get_plm_input(self, simidx):
        return alm_copy(self.sims_unl.get_sim_plm(simidx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)
    
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


    def get_cpp_input(self, simidx):
        fn_cpp_in = 'cpp_input'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_in):
            plmin = self.get_plm_input(simidx)
            cpp_in = alm2cl(plmin, plmin, self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn_cpp_in, cpp_in)
        cpp_in = cacher.load(fn_cpp_in)
        return cpp_in


    def get_cpp_itXinput(self, simidx, itr):
        fn = 'cpp_in_x_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            plmin = self.get_plm_input(simidx)
            # plmit = self.plms[simidx][itr]
            plmit = self.get_plm(simidx, itr)
            cpp_itXin = alm2cl(plmit, plmin, self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin
    


    def get_cpp(self, simidx, itr):
        fn_cpp_it = 'cpp_it_{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it):
            cpp = alm2cl(self.get_plm(simidx, itr), self.get_plm(simidx, itr), self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn_cpp_it, cpp)
        cpp = cacher.load(fn_cpp_it)
        return cpp
        
    # def get_wf(self):

    
    def get_cpp_qe(self, simidx):
        fn_cpp_qe = 'cpp_qe'
        fn_resp_qe = 'resp_qe'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_resp_qe):
            R = qresp.get_response(self.k, self.param.lmax_ivf, 'p', self.param.cls_len, self.param.cls_len, {'e': self.param.fel, 'b': self.param.fbl, 't':self.param.ftl}, lmax_qlm=self.param.lmax_qlm)[0]
            cacher.cache(fn_resp_qe, R)
        R = cacher.load(fn_resp_qe)
        if not cacher.is_cached(fn_cpp_qe):
            cpp = self.param.cls_unl['pp'][:self.lmax_qlm + 1]
            WF = cpp * utils.cli(cpp + utils.cli(R))
            cpp_qe_wf = self.get_cpp(simidx, 0)
            cpp_qe = cpp_qe_wf * utils.cli(WF)**2
            cacher.cache(fn_cpp_qe, cpp_qe)
        cpp_qe = cacher.load(fn_cpp_qe)
        return cpp_qe, R

    def get_mf0(self, simidx):
        return np.load(opj(self.libdir_sim(simidx), 'mf.npy'))
    

    def get_mf(self, simidx, itr, ret_alm=False, verbose=False):
        cacher = self.cacher_sim(simidx)
        fn_mf1 = 'mf1_it{}'.format(itr)
        fn_mf2 = 'mf2_it{}'.format(itr)
        if not cacher.is_cached(fn_mf1) or not cacher.is_cached(fn_mf2):
            if verbose:
                print('Starting to estimate MF for it {} from sim {}'.format(itr, simidx))
            itlib = self.get_itlib_sim(simidx)
            filtr = itlib.filter
            filtr.set_ffi(itlib._get_ffi(itr)) # load here the phi map you want 
            chain_descr = self.param.chain_descrs(self.param.lmax_unl, self.tol_iter)
            mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
            MF1 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
            MF2 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
            cacher.cache(fn_mf1, MF1)
            cacher.cache(fn_mf2, MF2)
        MF1 = cacher.load(fn_mf1)
        MF2 = cacher.load(fn_mf2)
        fn_mf = 'cpp_mf_it{}'.format(itr)
        fn_mf_mc = 'cpp_mf_mcnoise_it{}'.format(itr)
        if not cacher.is_cached(fn_mf) or not cacher.is_cached(fn_mf_mc):
            norm = self.param.qresp.get_response('p_p', self.param.lmax_ivf, 'p', self.param.cls_unl, self.param.cls_unl,  {'e': self.param.fel_unl, 'b': self.param.fbl_unl, 't':self.param.ftl_unl}, lmax_qlm=self.param.lmax_qlm)[0]
            cpp_mf = alm2cl(MF1[0], MF2[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
            cpp_mf_mc = alm2cl(MF1[0], MF1[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
            cacher.cache(fn_mf, cpp_mf)
            cacher.cache(fn_mf_mc, cpp_mf_mc)
        cpp_mf = cacher.load(fn_mf)
        cpp_mf_mc = cacher.load(fn_mf_mc)
        return (cpp_mf, cpp_mf_mc) if not ret_alm else (cpp_mf, cpp_mf_mc,MF1, MF2)

        # # plot MFs:
        # ls = np.arange(2, 2048)
        # w = ls ** 2 * (ls + 1.) ** 2 * 1e7 /2./np.pi
        # norm = self.param.qresp.get_response('p_p', self.param.lmax_ivf, 'p', self.param.cls_unl, self.param.cls_unl,  {'e': self.param.fel_unl, 'b': self.param.fbl_unl, 't':self.param.ftl_unl}, lmax_qlm=self.param.lmax_qlm)[0]
        # pl.loglog(ls, w * hp.alm2cl(MF1[0], MF2[0])[ls] / norm[ls] ** 2, label='MF spec')
        # pl.loglog(ls, w * hp.alm2cl(MF1[0], MF1[0])[ls] / norm[ls] ** 2, label='MF spec + MC noise')


    def get_n0_iter(self, itermax=15):
        lmin_ivf = 2 # TODO Not sure if this is the correct lmin in all case ?
        cacher = self.cacher_param
        fn_n0s = 'N0s_biased_itmax{}'.format(itermax)
        fn_n0s_unb = 'N0s_unbiased_itmax{}'.format(itermax)
        fn_resp_fid = 'resp_fid_itmax{}'.format(itermax)
        fn_resp_true = 'resp_true_itmax{}'.format(itermax)

        if any (not cacher.is_cached(fn) for fn in [fn_n0s, fn_n0s_unb, fn_resp_fid, fn_resp_true]):
            N0s_biased, N0s_unbiased, r_gg_fid, r_gg_true = nhl.get_N0_iter(
                self.k, self.param.nlev_t, self.param.nlev_p, self.param.beam, self.param.cls_unl, lmin_ivf, self.param.lmax_ivf, itermax, ret_delcls=False, ret_resp=True)
            cacher.cache(fn_n0s, N0s_biased)
            cacher.cache(fn_n0s_unb, N0s_unbiased)
            cacher.cache(fn_resp_fid, r_gg_fid)
            cacher.cache(fn_resp_true, r_gg_true)
        N0s_biased = cacher.load(fn_n0s)
        N0s_unbiased = cacher.load(fn_n0s_unb)
        rgg_fid  = cacher.load(fn_resp_fid)     
        rgg_true = cacher.load(fn_resp_true)     

        return N0s_biased, N0s_unbiased, rgg_fid, rgg_true


    def get_wf_fid(self, itermax=15):
        _, _, resp_fid, _ = self.get_n0_iter(itermax=itermax)
        return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]))

    def get_wf_sim(self, simidx, itr):
        fn = 'wf_sim_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            wf = self.get_cpp_itXinput(simidx, itr) * utils.cli(self.get_cpp_input(simidx))
            cacher.cache(fn, wf)
        return cacher.load(fn)
