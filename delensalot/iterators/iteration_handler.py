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

from lenspyx.remapping import deflection
from delensalot import utils_sims, utils_scarf
from delensalot.utils_hp import alm_copy, almxfl
from delensalot.iterators import cs_iterator, cs_iterator_fast
from delensalot.utils import read_map
from delensalot.iterators import cs_iterator, cs_iterator_fast
from delensalot.opfilt.opfilt_ee_wl import alm_filter_ninv_wl
from delensalot.opfilt.opfilt_iso_ee_wl import alm_filter_nlev_wl

class base_iterator():

    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, lensing_config):
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
        self.__dict__.update(lensing_config.__dict__)
        self.sims_MAP = sims_MAP
        
        self.libdir_iterator = libdir_iterators(k, simidx, version)
        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)

        self.tr = lensing_config.tr 
        self.qe = qe
        self.wflm0 = qe.get_wflm(self.simidx)
        self.R_unl0 = qe.R_unl()
        if self.QE_subtract_meanfield:
            self.mf0 = self.qe.get_meanfield(self.simidx)
        else:
            self.mf0 = np.zeros(shape=hp.Alm.getsize(self.lm_max_qlm[0]))
        self.plm0 = self.qe.get_plm(self.simidx, self.QE_subtract_meanfield)
        self.ffi = deflection(self.lenjob_geometry, np.zeros_like(self.plm0), self.lm_max_qlm[1], numthreads=self.tr, verbosity=self.verbose)
        # self.ffi = lenspyx.remapping.deflection.deflection(self.lenjob_pbgeometry, self.lensres, np.zeros_like(self.plm0),
            # self.lm_max_qlm[1], self.tr, self.tr)
        self.datmaps = self.get_datmaps()
        self.filter = self.get_filter()
        # TODO not sure why this happens here. Could be done much earlier
        self.it_chain_descr = lensing_config.it_chain_descr(lensing_config.lm_max_unl[0], lensing_config.it_cg_tol)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        # TODO change naming convention. Should align with map/alm params for ivfs and simdata
        if self.it_filter_directional == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration
            sht_job = utils_scarf.scarfjob()
            ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(self.sims_nside, zbounds=self.zbounds)
            sht_job.set_geometry(ninvjob_geometry)
            sht_job.set_triangular_alm_info(*self.lm_max_ivf)
            sht_job.set_nthreads(self.tr)
            return np.array(sht_job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2))
        else:
            return np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))
        

    @log_on_start(logging.INFO, "get_filter() started")
    @log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self):
        def get_filter_aniso():
            wee = self.k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
            ninv = [self.sims_MAP.ztruncify(read_map(ni)) for ni in self.ninvp_desc] # inverse pixel noise map on consistent geometry

            ## TODO use old scarf geom for now, but switch to lenspyx soon. filter currently does not support lenspyx geom, as no `scarfjob` objects exist anymore
            ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(self.sims_nside, zbounds=self.zbounds)
            filter = alm_filter_ninv_wl(ninvjob_geometry, ninv, self.ffi, self.ttebl['e'], self.lm_max_unl, self.lm_max_ivf, self.tr, self.tpl,
                                                    wee=wee, lmin_dotop=min(self.lmin_teb[1], self.lmin_teb[2]), transf_blm=self.ttebl['b'])
            self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc

            return filter

        def get_filter_iso():
            wee = self.k == 'p_p'
            filter = alm_filter_nlev_wl(self.nlev_p, self.ffi, self.ttebl['e'], self.lm_max_unl, self.lm_max_ivf,
                    wee=wee, transf_b=self.ttebl['b'], nlev_b=self.nlev_p)
            self.k_geom = filter.ffi.geom
            
            return filter
        
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        filter = get_filter_iso() if self.it_filter_directional == 'isotropic' else get_filter_aniso()
        
        return filter
        

class iterator_pertmf(base_iterator):

    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, lensing_config):
        super(iterator_pertmf, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, lensing_config)
        self.mf_resp0 = qe.get_response_meanfield()


    # TODO choose iterator via visitor pattern. perhaps already in p2lensrec
    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.
        Returns:
            _type_: _description_
        """
        iterator = cs_iterator.iterator_pertmf(
            self.libdir_iterator, 'p', self.lm_max_qlm, self.datmaps, self.plm0, self.mf_resp0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.it_chain_descr,
            self.stepper, mf0=self.mf0, wflm0=self.wflm0)
        
        return iterator


class iterator_constmf(base_iterator):
    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, lensing_config):
        """Return constmf iterator instance for simulation idx and qe_key type k

            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter

        """ 
        super(iterator_constmf, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, lensing_config)


    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        iterator = cs_iterator.iterator_cstmf(
            self.libdir_iterator, 'p', self.lm_max_qlm, self.datmaps, self.plm0, self.mf0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.it_chain_descr,
            self.stepper, wflm0=self.wflm0)
        
        return iterator


class iterator_fastWF(base_iterator):
    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, lensing_config):
        """Return constmf iterator instance for simulation idx and qe_key type k, fast WF for idealized fullsky case.

            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                cg_tol: tolerance of conjugate-gradient filter

        """
        super(iterator_fastWF, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, lensing_config)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        # TODO these are supposedly alms for fastWF.. how can this be alms in the most efficient way? 
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        # self.sims_MAP = self._sims

        # dat maps must now be given in harmonic space in this idealized configuration
        sht_job = utils_scarf.scarfjob()
        ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(self.sims_nside, zbounds=self.zbounds)
        sht_job.set_geometry(ninvjob_geometry)
        sht_job.set_triangular_alm_info(*self.lm_max_ivf)
        sht_job.set_nthreads(self.tr)
        return np.array(sht_job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2))
        

    @log_on_start(logging.INFO, "get_filter() started")
    @log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self):
        wee = self.k == 'p_p'
        filter = alm_filter_nlev_wl(self.nlev_p, self.ffi, self.ttebl['b'], self.lm_max_unl, self.lm_max_ivf,
                wee=wee, transf_b=self.ttebl['b'], nlev_b=self.nlev_p)
        self.k_geom = filter.ffi.geom

        return filter


    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        iterator = cs_iterator_fast.iterator_cstmf(
            self.libdir_iterator, self.k[0], self.lm_max_qlm, self.datmaps, self.plm0, self.mf0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.it_chain_descr,
            self.stepper, wflm0=self.wflm0)
        
        return iterator


# TODO Change into a proper visitor pattern
def transformer(descr):
    if descr == 'pertmf':
        return iterator_pertmf
    elif descr == 'constmf':
        return iterator_constmf
    elif descr == 'fastWF':
        return iterator_fastWF
    else:
        assert 0, "Not yet implemented"
