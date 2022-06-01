import os
from os.path import join as opj

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

        self.mf0 = qe.mf0
        self.plm0 = qe.plm0
        self.mf_resp = qe.mf_resp
        self.wflm0 = qe.wflm0(simidx)
        self.R_unl = qe.R_unl

        if not os.path.exists(self.libdir_iterator):
            os.makedirs(self.libdir_iterator)
        print('Starting get itlib for {}'.format(self.libdir_iterator))

        self.datmaps = self.get_datmaps() #TODO this should be done earlier. Perhaps in delensing_interface
        self.ffi = remapping.deflection(lensing_config.lenjob_pbgeometry, lensing_config.lensres, np.zeros_like(self.plm0), lensing_config.mmax_qlm, self.tr, self.tr)
        self.filter = self.get_filter(self.sims_MAP, self.ffi, self.tpl)
        self.chain_descr = lensing_config.chain_descr(lensing_config.lmax_unl, lensing_config.cg_tol)


    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        self.sims_MAP  = utils_sims.ztrunc_sims(self.lensing_config.sims, self.lensing_config.nside, [self.lensing_config.zbounds])
        datmaps = np.array(self.sims_MAP.get_sim_pmap(int(self.simidx)))

        return datmaps


    def get_filter(self, sims_MAP, ffi, tpl):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        wee = self.k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in self.lensing_config.ninv_p] # inverse pixel noise map on consistent geometry
        filter = opfilt_ee_wl.alm_filter_ninv_wl(self.lensing_config.ninvjob_geometry, ninv, ffi, self.lensing_config.transf_elm, (self.lensing_config.lmax_unl, self.lensing_config.mmax_unl), (self.lensing_config.lmax_ivf, self.lensing_config.mmax_ivf), self.tr, tpl,
                                                wee=wee, lmin_dotop=min(self.lensing_config.lmin_elm, self.lensing_config.lmin_blm), transf_blm=self.lensing_config.transf_blm)
        self.k_geom = filter.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc

        return filter


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

# TODO Above could be changed into a proper visitor pattern if needed at some point
# @transform.case(IBN_Model, IBN2Pomegranate_Transformer)
# def f1(expr, transformer): # pylint: disable=missing-function-docstring
#     return transformer.pom_ibn(expr)