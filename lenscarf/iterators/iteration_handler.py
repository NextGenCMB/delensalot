#!/usr/bin/env python

"""iteration_handler.py: This module receives input from lerepi, handles delensing jobs and runs them.
    Two main classes define the delensing; the plancklens QE delenser, and the D.lensalot MAP delensing.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os
import logging
from logdecorator import log_on_start, log_on_end

import numpy as np

from lenscarf import remapping
from lenscarf import utils_sims

from lenscarf.iterators import cs_iterator
from lenscarf.utils import read_map
from lenscarf.opfilt import opfilt_ee_wl

# TODO not sure if I need something like this at the moment. Implement visitor pattern if needed.
# from lenscarf.metamodel.iterator_pert import IT_PERT
# from lenscarf.core.visitor import transform


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
        self.k = k
        self.simidx = simidx
        self.version = version
        self.lensing_config = lensing_config
        
        self.libdir_iterator = libdir_iterators(k, simidx, version)
        self.tpl = lensing_config.tpl
        self.tr = lensing_config.tr

        self.mf0 = qe.get_meanfield_it0(simidx)
        self.plm0 = qe.get_plm_it0(simidx)
        self.mf_resp = qe.get_meanfield_response_it0()
        self.wflm0 = qe.get_wflm0
        self.R_unl = qe.R_unl

        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)

        # TODO this should be done earlier. Perhaps in delensing_interface. Lambda this to add simidx parameter
        self.datmaps = self.get_datmaps() 

        # TODO this should be done earlier. Lambda this to add simidx parameter, if needed
        self.ffi = remapping.deflection(lensing_config.lenjob_pbgeometry, lensing_config.lensres, np.zeros_like(self.plm0), lensing_config.mmax_qlm, self.tr, self.tr)
        
        # TODO this should be done earlier. Lambda this to add simidx parameter
        self.filter = self.get_filter(self.sims_MAP, self.ffi, self.tpl)

        # TODO not sure why this happens here. Could be done much earlier
        self.chain_descr = lensing_config.chain_descr(lensing_config.lmax_unl, lensing_config.cg_tol)


    @log_on_start(logging.INFO, "Start of get_datmaps()")
    @log_on_end(logging.INFO, "Finished get_datmaps()")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        self.sims_MAP  = utils_sims.ztrunc_sims(self.lensing_config.sims, self.lensing_config.nside, [self.lensing_config.zbounds])
        datmaps = np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))

        return datmaps


    @log_on_start(logging.INFO, "Start of get_filter()")
    @log_on_end(logging.INFO, "Finished get_filter()")
    def get_filter(self, sims_MAP=None, ffi=None, tpl=None):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        if sims_MAP == None:
            sims_MAP = self.sims_MAP
        if ffi == None:
            ffi = self.ffi
        if tpl == None:
            tpl = self.tpl
        wee = self.k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in self.lensing_config.ninv_p] # inverse pixel noise map on consistent geometry
        # TODO Add a typechecker to make sure we are passing the right objects to the filter
        filter = opfilt_ee_wl.alm_filter_ninv_wl(self.lensing_config.ninvjob_geometry, ninv, ffi, self.lensing_config.transf_elm, (self.lensing_config.lmax_unl, self.lensing_config.mmax_unl), (self.lensing_config.lmax_ivf, self.lensing_config.mmax_ivf), self.tr, tpl,
                                                wee=wee, lmin_dotop=min(self.lensing_config.lmin_elm, self.lensing_config.lmin_blm), transf_blm=self.lensing_config.transf_blm)
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc

        return filter


    # TODO this should somewhat be transformed into a visitor pattern
    @log_on_start(logging.INFO, "Start of get_iterator()")
    @log_on_end(logging.INFO, "Finished get_iterator()")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        iterator = cs_iterator.iterator_pertmf(
            self.libdir_iterator, 'p', (self.lensing_config.lmax_qlm, self.lensing_config.mmax_qlm), self.datmaps, self.plm0, self.mf_resp,
            self.R_unl, self.lensing_config.cpp, self.lensing_config.cls_unl, self.filter, self.k_geom, self.chain_descr,
            self.lensing_config.stepper, mf0=self.mf0, wflm0=self.wflm0)
        
        return iterator


def transformer(descr):
    if descr == 'pertmf':
        return scarf_iterator_pertmf
    else:
        assert 0, "Not yet implemented"

# TODO Above could be changed into a proper visitor pattern if needed at some point. But
# this iteration_handle would have to become a transformer module with transformer class
# @transform.case(pertmf, ITERATOR_TRANSFORMER)
# def f1(expr, transformer):
#     return transformer.scarf_iterator_pertmf(expr)