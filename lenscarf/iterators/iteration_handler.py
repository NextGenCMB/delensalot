import os
from os.path import join as opj

import numpy as np
from git.lerepi.lerepi.config.handler import lensing_config

from plancklens import utils, qresp

from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators import cs_iterator as scarf_iterator
from lenscarf import remapping
from lenscarf import utils_sims
from lenscarf.utils import read_map
from lenscarf.opfilt.bmodes_ninv import template_dense 
from lenscarf.opfilt import opfilt_ee_wl


class scarf_iterator_pertmf():
    def __init__(self, k:str, simidx:int, version:str, cg_tol:float, libdir_iterators, lensing_config, survey_config):
        """Return iterator instance for simulation idx and qe_key type k

            Args:
                k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
                simidx: simulation index to build iterative lensing estimate on
                version: string to use to test variants of the iterator with otherwise the same parfile
                        (here if 'noMF' is in version, will not use any mean-fied at the very first step)
                cg_tol: tolerance of conjugate-gradient filter

        """

        libdir_iterator = libdir_iterators(k, simidx, version)
        print('Starting get itlib for {}'.format(libdir_iterator))
        self.lensing_config = lensing_config
        self.survey_config = survey_config
        if not os.path.exists(libdir_iterator):
            os.makedirs(libdir_iterator)

        tr = int(os.environ.get('OMP_NUM_THREADS', 8))

        # QE mean-field fed in as constant piece in the iteration steps:
        mf_sims = np.unique(lensing_config.mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
        mf0 = lensing_config.qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
        if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(mf_sims)
            mf0 = (mf0 - lensing_config.qlms_dd.get_sim_qlm(k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

        path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm0):
            # We now build the Wiener-filtered QE here since not done already
            plm0  = lensing_config.qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
            plm0 -= mf0  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            R = lensing_config.qresp.get_response(k, lensing_config.lmax_ivf, 'p', lensing_config.cls_len, lensing_config.cls_len, {'e': lensing_config.fel, 'b': lensing_config.fbl, 't':lensing_config.ftl}, lmax_qlm=lensing_config.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = lensing_config.cpp * utils.cli(lensing_config.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, lensing_config.lmax_qlm, lensing_config.mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), lensing_config.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, lensing_config.mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, lensing_config.cpp > 0, lensing_config.mmax_qlm, True)
            np.save(path_plm0, plm0)

        self.plm0 = np.load(path_plm0)
        self.R_unl = qresp.get_response(k, lensing_config.lmax_ivf, 'p', lensing_config.cls_unl, lensing_config.cls_unl,  {'e': lensing_config.fel_unl, 'b': lensing_config.fbl_unl, 't':lensing_config.ftl_unl}, lmax_qlm=lensing_config.lmax_qlm)[0]
        if k in ['p_p'] and not 'noRespMF' in version :
            self.mf_resp = qresp.get_mf_resp(k, lensing_config.cls_unl, {'ee': lensing_config.fel_unl, 'bb': lensing_config.fbl_unl}, lensing_config.lmax_ivf, lensing_config.lmax_qlm)[0]
        else:
            print('*** mf_resp not implemented for key ' + k, ', setting it to zero')
            self.mf_resp = np.zeros(lensing_config.lmax_qlm + 1, dtype=float)
        # Lensing deflection field instance (initiated here with zero deflection)
        ffi = remapping.deflection(lensing_config.lenjob_pbgeometry, lensing_config.lensres, np.zeros_like(plm0), lensing_config.mmax_qlm, tr, tr)
        if k in ['p_p', 'p_eb']:
            if lensing_config.isOBD:
                tpl = template_dense(200, lensing_config.ninvjob_geometry, tr, _lib_dir=lensing_config.BMARG_LIBDIR) # for template projection
            else:
                tpl = None # for template projection, here set to None
            wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
            sims_MAP  = utils_sims.ztrunc_sims(survey_config.sims, lensing_config.nside, [lensing_config.zbounds])
            ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in lensing_config.ninv_p] # inverse pixel noise map on consistent geometry
            self.filtr = opfilt_ee_wl.alm_filter_ninv_wl(lensing_config.ninvjob_geometry, ninv, ffi, lensing_config.transf_elm, (lensing_config.lmax_unl, lensing_config.mmax_unl), (lensing_config.lmax_ivf, lensing_config.mmax_ivf), tr, tpl,
                                                    wee=wee, lmin_dotop=min(lensing_config.lmin_elm, lensing_config.lmin_blm), transf_blm=lensing_config.transf_blm)
            self.datmaps = np.array(sims_MAP.get_sim_pmap(int(simidx)))
        else:
            assert 0
        self.k_geom = self.filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
        # Sets to zero all L-modes below Lmin in the iterations:


    def get_iterator(self, idx):
        iterator = scarf_iterator.iterator_pertmf(self.libdir_iterator, 'p',
        (self. lmax_qlm, self.lensing_config.mmax_qlm), self.datmaps, self.plm0, self.mf_resp,
        self.R_unl, lensing_config.cpp, lensing_config.cls_unl, self.filtr, self.k_geom, lensing_config.chain_descrs(lensing_config.lmax_unl, lensing_config.cg_tol), self.lensing_config.stepper,
        mf0=self.mf0, wflm0=lambda : alm_copy(lensing_config.ivfs.get_sim_emliklm(idx), None, lensing_config.lmax_unl, lensing_config.mmax_unl))
        return iterator