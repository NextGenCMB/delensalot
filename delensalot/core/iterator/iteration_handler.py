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

from delensalot.config.visitor import transform
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept
from delensalot.core.opfilt.opfilt_handler import MAP_transformer 
from delensalot.core.iterator import cs_iterator, cs_iterator_fast


class base_iterator():

    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, iterator_config):
        """Iterator instance for simulation idx and qe_key type k
            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter
        """
        self.k = k
        self.simidx = simidx
        self.version = version
        self.__dict__.update(iterator_config.__dict__)
        self.iterator_config = iterator_config
        self.sims_MAP = sims_MAP
        
        self.libdir_iterator = libdir_iterators(k, simidx, version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)

        self.tr = iterator_config.tr 
        self.qe = qe
        self.wflm0 = qe.get_wflm(self.simidx)
        self.R_unl0 = qe.R_unl()
        self.mf0 = qe.get_meanfield(self.simidx) if self.QE_subtract_meanfield else np.zeros(shape=hp.Alm.getsize(self.lm_max_qlm[0]))
        self.plm0 = self.qe.get_plm(self.simidx, self.QE_subtract_meanfield)
        self.filter = self.get_filter()
        # TODO not sure why this happens here. Could be done much earlier
        self.it_chain_descr = iterator_config.it_chain_descr(iterator_config.lm_max_unl[0], iterator_config.it_cg_tol)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        # TODO change naming convention. Should align with map/alm params for ivfs and simdata
        if self.it_filter_directional == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration
            job = utils_geom.Geom.get_healpix_geometry(self.sims_nside)
            thtbounds = (np.arccos(self.zbounds[1]), np.arccos(self.zbounds[0]))
            job = job.restrict(*thtbounds, northsouth_sym=False)
            if self.k in ['pee']:
                return np.array(job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))[0]
            else:
                return np.array(job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))
        else:
            return np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))
        

    @log_on_start(logging.INFO, "get_filter() started")
    @log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self): 
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)

        filter_MAP = transform(self.iterator_config, MAP_transformer())
        filter = transform(self.iterator_config, filter_MAP())
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
        
        return filter
        

class iterator_transformer(base_iterator):

    def __init__(self, qe, k, simidx, version, sims_MAP, libdir_iterators, iterator_config):
        super(iterator_transformer, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, iterator_config)

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
                'ninv_filt': self.filter,
                'k_geom': self.filter.ffi.geom,
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
                'ninv_filt': self.filter,
                'k_geom': self.filter.ffi.geom,
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
                'ninv_filt': self.filter,
                'k_geom': self.filter.ffi.geom,
                'chain_descr': self.it_chain_descr,
                'stepper': cf.stepper,
                'wflm0': self.wflm0,
            }
        return cs_iterator_fast.iterator_cstmf(**extract())


@transform.case(DLENSALOT_Concept, iterator_transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    if expr.iterator_typ in ['constmf']:
        return transformer.build_constmf_iterator(expr)
    elif expr.iterator_typ in ['pertmf']:
        return transformer.build_pertmf_iterator(expr)
    elif expr.iterator_typ in ['fastWF']:
        return transformer.build_fastwf_iterator(expr)