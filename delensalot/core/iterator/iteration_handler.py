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
from delensalot.core.opfilt.opfilt_handler import QE_transfomer, MAP_transfomer 
from delensalot.core.iterator import cs_iterator, cs_iterator_fast


class base_iterator():

    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, iterator_config, isQE=False):
        """Iterator instance for simulation idx and qe_key type k
            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter
        """
        self.k = k

        # TODO This is a nightmare.. due to get_template_blm being inside cs_iterator, iteration_handler needs to be faked..
        # but inside ith the filter is initialized.. so for QE, it will set up a MAP filter.... and why are we using MAP filter for QE lensing template?
        self.isQE = isQE
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
        if self.QE_subtract_meanfield:
            self.mf0 = self.qe.get_meanfield(self.simidx)
        else:
            self.mf0 = np.zeros(shape=hp.Alm.getsize(self.lm_max_qlm[0]))
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

        # if self.isQE:
        #     filter_QE = transform(self.iterator_config, QE_transfomer())
        #     filter = transform(self.iterator_config, filter_QE())
        #     self.k_geom = None
        # else:
        filter_MAP = transform(self.iterator_config, MAP_transfomer())
        filter = transform(self.iterator_config, filter_MAP())
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
        
        return filter
        

class iterator_pertmf(base_iterator):
    """Return perturbative-mean-field iterator instance for simulation idx and qe_key type k
    """

    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, iterator_config, isQE=False):
        super(iterator_pertmf, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, iterator_config, isQE)
        self.mf_resp0 = qe.get_response_meanfield()


    # TODO choose iterator via visitor pattern. perhaps already in p2lensrec
    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.
        Returns:
            _type_: _description_
        """
        self.datmaps = self.get_datmaps()
        iterator = cs_iterator.iterator_pertmf(
            self.libdir_iterator, 'p', self.lm_max_qlm, self.datmaps, self.plm0, self.mf_resp0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.it_chain_descr,
            self.stepper, mf0=self.mf0, wflm0=self.wflm0)
        
        return iterator


class iterator_constmf(base_iterator):
    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, iterator_config, isQE=False):
        """Return constmf iterator instance for simulation idx and qe_key type k

            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter

        """ 
        super(iterator_constmf, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, iterator_config, isQE)


    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        self.datmaps = self.get_datmaps()
        iterator = cs_iterator.iterator_cstmf(
            self.libdir_iterator, 'p', self.lm_max_qlm, self.datmaps, self.plm0, self.mf0,
            self.R_unl0, self.cpp, self.cls_unl, self.filter, self.k_geom, self.it_chain_descr,
            self.stepper, wflm0=self.wflm0)
        
        return iterator


class iterator_fastWF(base_iterator):
    def __init__(self, qe, k:str, simidx:int, version:str, sims_MAP, libdir_iterators, iterator_config, isQE=False):
        """Return fast Wiener-filtering constmf iterator instance for simulation idx and qe_key type k, fast WF for idealized fullsky case.

            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                cg_tol: tolerance of conjugate-gradient filter

        """
        super(iterator_fastWF, self).__init__(qe, k, simidx, version, sims_MAP, libdir_iterators, iterator_config, isQE)


    @log_on_start(logging.INFO, "get_datmaps() started")
    @log_on_end(logging.INFO, "get_datmaps() finished")
    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        # self.sims_MAP = self._sims

        job = utils_geom.Geom.get_healpix_geometry(self.sims_nside)
        thtbounds = (np.arccos(self.zbounds[1]), np.arccos(self.zbounds[0]))
        job = job.restrict(*thtbounds, northsouth_sym=False)
        
        if self.k in ['pee']:
            return np.array(job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))[0]
        else:
            return np.array(job.map2alm_spin(self.sims_MAP.get_sim_pmap(int(self.simidx)), 2, *self.lm_max_ivf, nthreads=self.tr))


    @log_on_start(logging.INFO, "get_iterator() started")
    @log_on_end(logging.INFO, "get_iterator() finished")
    def get_iterator(self):
        """iterator_pertmf needs a whole lot of parameters, which are calculated when initialising this class.

        Returns:
            _type_: _description_
        """
        self.datmaps = self.get_datmaps()
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
        log.info(descr)
        assert 0, "Not yet implemented"
