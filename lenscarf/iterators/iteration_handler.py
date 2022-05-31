import os
from os.path import join as opj

import numpy as np

from plancklens import utils, qresp

from lenscarf import remapping
from lenscarf import utils_sims
from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators import cs_iterator
from lenscarf.utils import read_map
from lenscarf.opfilt import opfilt_ee_wl


class scarf_iterator_pertmf():
    def __init__(self, k:str, simidx:int, version:str, libdir_iterators, lensing_config, survey_config, run_config):
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
        libdir_iterator = libdir_iterators(k, simidx, version)
        self.lensing_config = lensing_config
        self.survey_config = survey_config
        self.libdir_iterator = libdir_iterator
        self.tpl = lensing_config.tpl
        self.tr = lensing_config.tr
        if not os.path.exists(libdir_iterator):
            os.makedirs(libdir_iterator)
        print('Starting get itlib for {}'.format(libdir_iterator))

        self.wflm0 = lambda : alm_copy(lensing_config.ivfs.get_sim_emliklm(simidx), None, lensing_config.lmax_unl, lensing_config.mmax_unl)
        self.R_unl = qresp.get_response(self.k, self.lensing_config.lmax_ivf, 'p', self.lensing_config.cls_unl, self.lensing_config.cls_unl,  {'e': self.lensing_config.fel_unl, 'b': self.lensing_config.fbl_unl, 't':self.lensing_config.ftl_unl}, lmax_qlm=self.lensing_config.lmax_qlm)[0]
        self.mf0 = self.get_meanfield_it0()
        self.plm0 = self.get_plm_it0()
        self.mf_resp = self.get_meanfield_response_it0()
        self.datmaps = self.get_datmaps()
        self.ffi = remapping.deflection(lensing_config.lenjob_pbgeometry, lensing_config.lensres, np.zeros_like(self.plm0), lensing_config.mmax_qlm, self.tr, self.tr)

        self.filter = self.get_filter(self.sims_MAP, self.ffi, self.tpl)


        self.chain_descr = lensing_config.chain_descr(lensing_config.lmax_unl, run_config.cg_tol)


    def get_meanfield_it0(self):
        # QE mean-field fed in as constant piece in the iteration steps:
        mf_sims = np.unique(self.lensing_config.mc_sims_mf_it0 if not 'noMF' in self.version else np.array([]))
        mf0 = self.lensing_config.qlms_dd.get_sim_qlm_mf(self.k, mf_sims)  # Mean-field to subtract on the first iteration:
        if self.simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(mf_sims)
            mf0 = (mf0 - self.lensing_config.qlms_dd.get_sim_qlm(self.k, int(self.simidx)) / Nmf) * (Nmf / (Nmf - 1))
        return mf0


    def get_plm_it0(self):
        path_plm0 = opj(self.libdir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm0):
            # We now build the Wiener-filtered QE here since not done already
            plm0  = self.lensing_config.qlms_dd.get_sim_qlm(self.k, int(self.simidx))  #Unormalized quadratic estimate:
            plm0 -= self.mf0  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            R = qresp.get_response(self.k, self.lensing_config.lmax_ivf, 'p', self.lensing_config.cls_len, self.lensing_config.cls_len, {'e': self.lensing_config.fel, 'b': self.lensing_config.fbl, 't':self.lensing_config.ftl}, lmax_qlm=self.lensing_config.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.lensing_config.cpp * utils.cli(self.lensing_config.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, self.lensing_config.lmax_qlm, self.lensing_config.mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), self.lensing_config.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, self.lensing_config.mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, self.lensing_config.cpp > 0, self.lensing_config.mmax_qlm, True)
            np.save(path_plm0, plm0)

        return np.load(path_plm0)


    def get_meanfield_response_it0(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version :
            mf_resp = qresp.get_mf_resp(self.k, self.lensing_config.cls_unl, {'ee': self.lensing_config.fel_unl, 'bb': self.lensing_config.fbl_unl}, self.lensing_config.lmax_ivf, self.lensing_config.lmax_qlm)[0]
        else:
            print('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lensing_config.lmax_qlm + 1, dtype=float)

        return mf_resp


    def get_datmaps(self):
        assert self.k in ['p_p', 'p_eb'], '{} not supported. Implement if needed'.format(self.k)
        self.sims_MAP  = utils_sims.ztrunc_sims(self.survey_config.sims, self.lensing_config.nside, [self.lensing_config.zbounds])
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