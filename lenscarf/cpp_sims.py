import numpy as np
from lenscarf import utils
from lenscarf.iterators import statics
from plancklens.sims import planck2018_sims
from plancklens import qresp
from lenscarf import cachers

from os.path import join as opj
import os
import importlib
from lenscarf.utils_hp import alm2cl, alm_copy

class cpp_sims_lib:
    def __init__(self, k, itmax, tol, imin, imax, v='', param_file='cmbs4wide_planckmask', label=''):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.
        """
        
        # Load the parameters defined in the param_file
        self.param = importlib.import_module('lenscarf.params.'+param_file)

        self.k = k
        self.itmax = itmax
        self.tol = tol
        self.imin = imin
        self.imax = imax
        self.version = v
        self.iters = np.arange(itmax+1)
        self.label = label
        self.TEMP =  self.param.TEMP
        self.lmax_qlm = self.param.lmax_qlm
        self.mmax_qlm = self.param.mmax_qlm
        # Load required plms
        self.plms = []
        for isim in np.arange(imin, imax+1):
            self.plms.append(self.get_plm(isim, self.iters))
        
        if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            self.sims_unl = planck2018_sims.cmb_unl_ffp10() 
     
    def libdir_sim(self, simidx):
        return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)

    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)

    def get_plm(self, simidx, iters):
        return statics.rec.load_plms(self.libdir_sim(simidx), iters)
    
    def get_plm_input(self, simidx):
        return alm_copy(self.sims_unl.get_sim_plm(simidx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)
    

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
            plmit = self.plms[simidx][itr]
            cpp_init = alm2cl(plmit, plmin, self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn, cpp_init)
        cpp_init = cacher.load(fn)
        return cpp_init
    

    def get_cpp(self, simidx, itr):
        fn_cpp_it = 'cpp_it_{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it):
            cpp = alm2cl(self.plms[simidx][itr], self.plms[simidx][itr], self.lmax_qlm, self.mmax_qlm, None)
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
            cpp = self.param.cls_unl['pp'][:self.param.lmax_qlm + 1]
            WF = cpp * utils.cli(cpp + utils.cli(R))
            cpp_qe_wf = self.get_cpp(simidx, 0)
            cpp_qe = cpp_qe_wf * utils.cli(WF)**2
            cacher.cache(fn_cpp_qe, cpp_qe)
        cpp_qe = cacher.load(fn_cpp_qe)
        return cpp_qe

    def get_mf(self, simidx):
        return np.load(opj(self.libdir_sim(simidx), 'mf.npy'))
    

    def get_n0n1_iter(self):
        # get_N0_iter and response ?
        # Allow to get iterative WF
        return 
