#!/usr/bin/env python

"""iteration_handler.py: This module is a passthrough to Dlensalot.cs_iterator. In the future, it will serve as a template module, which helps
setting up an iterator, (e.g. permf or constmf), and decide which object on iteration level will be used, (e.g. cg, bfgs, filter).
At this level, can possibly also choose likelihood, which would be needed for crosscorrelating to external tracers.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os, sys

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from lenscarf import remapping
from lenscarf import utils_sims
from lenscarf.iterators import cs_iterator
from lenscarf.utils import read_map
from lenscarf.opfilt import opfilt_ee_wl


class scarf_iterator_pertmf():
    def __init__(self, qe, k:str, simidx:int, version:str, libdir_iterators, lensing_config):
        """Return iterator instance for simulation idx and qe_key type k
            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter
        """
        self.__dict__.update(lensing_config.__dict__)
        self.simidx = simidx
        self.lensing_config = lensing_config
        
        self.libdir_iterator = libdir_iterators(k, simidx, version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)
        if lensing_config.tpl is not None:
            self.tpl = lensing_config.tpl(**lensing_config.tpl_kwargs)
        else:
            self.tpl = lensing_config.tpl
        self.tr = lensing_config.tr

        self.qe = qe
        self.mf_resp0 = qe.get_response_meanfield()
        self.wflm0 = qe.get_wflm(self.simidx)
        self.R_unl0 = qe.R_unl()
        self.mf0 = self.qe.get_meanfield(self.simidx)
        self.plm0 = self.qe.get_plm(self.simidx)

        self.ffi = remapping.deflection(self.lenjob_pbgeometry, self.lensres, np.zeros_like(self.plm0),
            self.mmax_qlm, self.tr, self.tr)
        self.datmaps = self.get_datmaps()
        self.filter = self.get_filter(self.sims_MAP, self.ffi, self.tpl)
        # TODO not sure why this happens here. Could be done much earlier
        self.chain_descr = lensing_config.chain_descr(lensing_config.lmax_unl, lensing_config.cg_tol)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        self.sims_MAP  = utils_sims.ztrunc_sims(self.sims, self.nside, [self.zbounds])
        datmaps = np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))

        return datmaps


    @log_on_start(logging.INFO, "get_filter() started")
    @log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self, sims_MAP=None, ffi=None, tpl=None):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        if sims_MAP == None:
            sims_MAP = self.sims_MAP
        if ffi == None:
            ffi = self.ffi
        if tpl == None:
            tpl = self.tpl
        wee = self.k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in self.ninv_p] # inverse pixel noise map on consistent geometry
        filter = opfilt_ee_wl.alm_filter_ninv_wl(self.ninvjob_geometry, ninv, ffi, self.transf_elm, (self.lmax_unl, self.mmax_unl), (self.lmax_ivf, self.mmax_ivf), self.tr, tpl,
                                                wee=wee, lmin_dotop=min(self.lmin_elm, self.lmin_blm), transf_blm=self.transf_blm)
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc

        return filter


    # TODO choose iterator via visitor pattern. perhaps already in p2lensrec
    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.
        Returns:
            _type_: _description_
        """
        iterator = cs_iterator.iterator_pertmf(
            self.libdir_iterator, 'p', (self.lmax_qlm, self.mmax_qlm), self.datmaps, self.plm0, self.mf_resp0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.chain_descr,
            self.stepper, mf0=self.mf0, wflm0=self.wflm0)
        
        return iterator


class scarf_iterator_constmf():
    def __init__(self, qe, k:str, simidx:int, version:str, libdir_iterators, lensing_config):
        """Return constmf iterator instance for simulation idx and qe_key type k

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
        self.__dict__.update(lensing_config.__dict__)
        
        self.libdir_iterator = libdir_iterators(k, simidx, version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)
        self.tpl = lensing_config.tpl(**lensing_config.tpl_kwargs)
        self.tr = lensing_config.tr 
        self.qe = qe
        self.mf_resp0 = qe.get_response_meanfield()
        self.wflm0 = qe.get_wflm(self.simidx)
        self.R_unl0 = qe.R_unl()
        self.mf0 = self.qe.get_meanfield(self.simidx)
        self.plm0 = self.qe.get_plm(self.simidx)

        self.ffi = remapping.deflection(self.lenjob_pbgeometry, self.lensres, np.zeros_like(self.plm0),
            self.mmax_qlm, self.tr, self.tr)
        self.datmaps = self.get_datmaps()
        self.filter = self.get_filter(self.sims_MAP, self.ffi, self.tpl)
        # TODO not sure why this happens here. Could be done much earlier
        self.chain_descr = lensing_config.chain_descr(lensing_config.lmax_unl, lensing_config.cg_tol)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        self.sims_MAP  = utils_sims.ztrunc_sims(self.sims, self.nside, [self.zbounds])
        datmaps = np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))

        return datmaps


    @log_on_start(logging.INFO, "get_filter() started")
    @log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self, sims_MAP=None, ffi=None, tpl=None):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        if sims_MAP == None:
            sims_MAP = self.sims_MAP
        if ffi == None:
            ffi = self.ffi
        if tpl == None:
            tpl = self.tpl
        wee = self.k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in self.ninv_p] # inverse pixel noise map on consistent geometry
        filter = opfilt_ee_wl.alm_filter_ninv_wl(self.ninvjob_geometry, ninv, ffi, self.transf_elm, (self.lmax_unl, self.mmax_unl), (self.lmax_ivf, self.mmax_ivf), self.tr, tpl,
                                                wee=wee, lmin_dotop=min(self.lmin_elm, self.lmin_blm), transf_blm=self.transf_blm)
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc

        return filter


    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        iterator = cs_iterator.iterator_cstmf(
            self.libdir_iterator, 'p', (self.lmax_qlm, self.mmax_qlm), self.datmaps, self.plm0, self.mf0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.chain_descr,
            self.stepper, wflm0=self.wflm0)
        
        return iterator


# TODO Change into a proper visitor pattern
def transformer(descr):
    if descr == 'pertmf':
        return scarf_iterator_pertmf
    elif descr == 'constmf':
        return scarf_iterator_constmf
    else:
        assert 0, "Not yet implemented"
