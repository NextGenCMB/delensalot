#!/usr/bin/env python

"""iteration_handler.py: This module is a passthrough to Dlensalot.cs_iterator. In the future, it will serve as a template module, which helps
setting up an iterator, (e.g. permf or constmf), and decide which object on iteration level will be used, (e.g. cg, bfgs, filter).
    
"""

import os, sys

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp

from lenspyx.remapping import utils_geom
from delensalot.core.iterator import cs_iterator, cs_iterator_fast

class base_iterator():

    def __init__(self, job_model, simidx:int, delensalot_model):
        """Iterator instance for simulation idx and qe_key type k
            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter
        """
        self.simidx = simidx
        self.__dict__.update(job_model.__dict__)
        self.iterator_config = delensalot_model
        
        self.libdir_iterator = self.libdir_MAP(self.k, simidx, self.version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)

        self.tr = self.iterator_config.tr 
        self.wflm0 = self.qe.get_wflm(self.simidx)
        self.R_unl0 = self.qe.R_unl()
        self.mf0 = self.qe.get_meanfield(self.simidx) if self.QE_subtract_meanfield else np.zeros(shape=hp.Alm.getsize(self.lm_max_qlm[0]))
        self.plm0 = self.qe.get_plm(self.simidx, self.QE_subtract_meanfield)
        # TODO not sure why this happens here. Could be done much earlier
        self.it_chain_descr = self.iterator_config.it_chain_descr(self.iterator_config.lm_max_unl[0], self.iterator_config.it_cg_tol)
        

    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        # assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        if self.it_filter_directional == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.nivjob_geomlib.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))[0]
            elif self.k in ['ptt']:
                return self.nivjob_geomlib.map2alm(self.sims_MAP.get_sim_tmap(int(self.simidx)), *self.lm_max_ivf, nthreads=self.tr)
            elif self.k in ['p']:
                QUobs = np.array(self.nivjob_geomlib.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))
                Tobs = self.nivjob_geomlib.map2alm(self.sims_MAP.get_sim_tmap(int(self.simidx)), *self.lm_max_ivf, nthreads=self.tr)
                return np.array([Tobs, QUobs])
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))
            else:
                assert 0, 'implement if needed'
        

class iterator_transformer(base_iterator):

    def __init__(self, qe, simidx, job_model):
        super(iterator_transformer, self).__init__(qe, simidx, job_model)


    def build_constmf_iterator(self, cf):

        def extract():
            return {
                'lib_dir': self.libdir_iterator,
                'h': cf.k[0],
                'lm_max_dlm': cf.lm_max_qlm,
                'dat_maps': self.get_datmaps(),
                'plm0': self.plm0,
                'mf0': self.mf0,
                'pp_h0': self.R_unl0,
                'cpp_prior': cf.cpp,
                'cls_filt': cf.cls_unl,
                'ninv_filt': cf.filter,
                'k_geom': cf.filter.ffi.geom,
                'chain_descr': self.it_chain_descr,
                'stepper': cf.stepper,
                'wflm0': self.wflm0,
            }

        return cs_iterator.iterator_cstmf(**extract())

    
    def build_pertmf_iterator(self, cf):

        def extract():
            return {
                'lib_dir': self.libdir_iterator,
                'h': cf.k[0],
                'lm_max_dlm': cf.lm_max_qlm,
                'dat_maps': self.get_datmaps(),
                'plm0': self.plm0,
                'mf_resp': self.qe.get_response_meanfield(),
                'pp_h0': self.R_unl0,
                'cpp_prior': cf.cpp,
                'cls_filt': cf.cls_unl,
                'ninv_filt': cf.filter,
                'k_geom': cf.filter.ffi.geom,
                'chain_descr': self.it_chain_descr,
                'stepper': cf.stepper,
                'mf0': self.mf0,
                'wflm0': self.wflm0,
            }
        return cs_iterator.iterator_pertmf(**extract())
    

    def build_fastwf_iterator(self, cf):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        def extract():
            return {
                'lib_dir': self.libdir_iterator,
                'h': cf.k[0],
                'lm_max_dlm': cf.lm_max_qlm,
                'dat_maps': self.get_datmaps(),
                'plm0': self.plm0,
                'mf0': self.mf0,
                'pp_h0': self.R_unl0,
                'cpp_prior': cf.cpp,
                'cls_filt': cf.cls_unl,
                'ninv_filt': cf.filter,
                'k_geom': cf.filter.ffi.geom,
                'chain_descr': self.it_chain_descr,
                'stepper': cf.stepper,
                'wflm0': self.wflm0,
            }
        return cs_iterator_fast.iterator_cstmf(**extract())