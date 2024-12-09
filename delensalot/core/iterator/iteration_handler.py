#!/usr/bin/env python

"""iteration_handler.py: This module is a passthrough to delensalot.cs_iterator. In the future, it will serve as a template module, which helps
setting up an iterator, (e.g. permf or constmf, lognormal, ..), and decide which object on iteration level will be used, (e.g. cg, bfgs, filter).
    
"""

import os, sys
from os.path import join as opj

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp

from delensalot.utility.utils_hp import alm_copy, alm2cl
from delensalot.utils import cli
from delensalot.core import mpi
from delensalot.core import cachers
from delensalot.core.cg_simple import multigrid
from delensalot.core.iterator import cs_iterator, cs_iterator_fast
from delensalot.core.iterator import bfgs

from delensalot.utils import cli

from delensalot.utility.utils_hp import Alm, almxfl


def _p2h(h, lmax):
    if h == 'p':
        return np.ones(lmax + 1, dtype=float)
    elif h == 'k':
        return 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
    elif h == 'd':
        return np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2), dtype=float)
    else:
        assert 0, h + ' not implemented'

def _h2p(h, lmax): return cli(_p2h(h, lmax))

class base_iterator():

    def __init__(self, job_model, simidx:int, delensalot_model):
        self.simidx = simidx
        self.__dict__.update(job_model.__dict__)
        self.iterator_config = delensalot_model
        
        self.libdir_iterator = self.libdir_MAP(self.k, simidx, self.version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)

        self.tr = self.iterator_config.tr
        if self.qe.qe_filter_directional == 'anisotropic':
            mpi.disable()
            self.qe.init_aniso_filter()
            mpi.enable()
        self.wflm0 = self.qe.get_wflm(self.simidx)


        self.R_unl0 = self.qe.R_unl()
        chh = self.cpp[:self.lm_max_qlm[0]+1] * _p2h(self.k[0], self.lm_max_qlm[0]) ** 2
        h0 = cli(self.R_unl0[:self.lm_max_qlm[0] + 1] * _h2p(self.k[0], self.lm_max_qlm[0]) ** 2 + cli(chh))  #~ (1/Cpp + 1/N0)^-1
        h0 *= (chh > 0)
        apply_H0k = lambda rlm, kr: almxfl(rlm, h0, self.lm_max_qlm[0], False)
        apply_B0k = lambda rlm, kr: almxfl(rlm, cli(h0), self.lm_max_qlm[0], False)
        lp1 = 2 * np.arange(self.lm_max_qlm[0] + 1) + 1
        dot_op = lambda rlm1, rlm2: np.sum(lp1 * alm2cl(rlm1, rlm2, self.m_max_qlm[0], self.m_max_qlm[1], self.m_max_qlm[0]))
        self.hess_cacher = cachers.cacher_npy(opj(self.libdir_iterator, 'hessian'))
        self.BFGS_lib = bfgs.BFGS_Hessian(h0=h0, apply_H0k=apply_H0k, apply_B0k=apply_B0k, cacher=self.hess_cacher, dot_op=dot_op)

        self.mf0 = self.qe.get_meanfield(self.simidx) if self.QE_subtract_meanfield else np.zeros(shape=hp.Alm.getsize(self.lm_max_qlm[0]))
        self.plm0 = self.qe.get_plm(self.simidx, self.QE_subtract_meanfield)
        self.it_chain_descr = self.iterator_config.it_chain_descr(self.iterator_config.lm_max_unl[0], self.iterator_config.it_cg_tol)

        opfilt = sys.modules[self.filter.__module__]
        self.mchain = multigrid.multigrid_chain(opfilt, self.it_chain_descr, self.cls_filt)
        

    @log_on_start(logging.DEBUG, "get_datmaps() started")
    @log_on_end(logging.DEBUG, "get_datmaps() finished")
    def get_datmaps(self):
        if self.it_filter_directional == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_ivf)
            if self.k in ['pee']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_ivf)[0]
            elif self.k in ['ptt']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *self.lm_max_ivf)
            elif self.k in ['p']:
                EBobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_ivf)
                Tobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *self.lm_max_ivf)         
                ret = np.array([Tobs, *EBobs])
                return ret
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.simidx), dtype=float)
            else:
                assert 0, 'implement if needed'
        

class iterator_transformer(base_iterator):

    def __init__(self, qe, simidx, job_model):
        super(iterator_transformer, self).__init__(qe, simidx, job_model)


    def build_glm_constmf_iterator(self, cf):

        def extract():
            return {
                'data': self.get_datmaps(),
                'ninv_filt': cf.filter,
                'lib_dir': self.libdir_iterator,
                'h': cf.k[0],
                'lm_max_qlm': cf.lm_max_qlm,
                'plm0': self.plm0,
                'mf0': self.mf0,
                'h0': self.h0,
                'mchain': self.mchain,
                'cpp_prior': cf.cpp,
                'wflm0': self.wflm0,
                'BGFS_lib': self.BFGS_lib,
            }

        return cs_iterator.glm_iterator(**extract())
    
    def build_joint_constmf_iterator(self, cf):

        def extract():
            return {
                'lib_dir': self.libdir_iterator,
                'h': cf.k[0],
                'lm_max_dlm': cf.lm_max_qlm,
                'dat_maps': self.get_datmaps(),
                'plm0': self.plm0,
                'mf0': self.mf0,
                'h0': self.h0,
                'mchain': self.mchain,
                'cpp_prior': cf.cpp,
                'ninv_filt': cf.filter,
                'stepper': cf.stepper,
                'wflm0': self.wflm0,
                'BGFS_lib': self.BFGS_lib,
            }

        return cs_iterator.gclm_iterator(**extract())


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