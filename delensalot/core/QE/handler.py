import os
import copy
from os.path import join as opj
import numpy as np

from plancklens import qresp, qest, utils as pl_utils
from lenspyx.lensing import get_geom 

from delensalot.config.visitor import transform, transform3d

from delensalot.core.ivf import filt_util, filt_cinv, filt_simple
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD
from delensalot.core.opfilt.opfilt_handler import QE_transformer

from delensalot.utility import utils_qe, utils_sims
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam

class base:
    def __init__(self, kwargs):
        self.field = kwargs['field']
        self.simidx = kwargs['simidx']
        self.estimator_key = kwargs['estimator_key']
        self.qe_filter_directional = kwargs['qe_filter_directional']
        self.libdir_QE = kwargs['libdir_QE']
        self.simulationdata = kwargs['simulationdata']
        self.nivjob_geominfo = kwargs['nivjob_geominfo']
        self.ttebl = kwargs['ttebl']
        self.cls_len = kwargs['cls_len']
        self.ftebl_len = kwargs['ftebl_len']
        self.ftebl_unl = kwargs['ftebl_unl']
        self.lm_max_ivf = kwargs['lm_max_ivf']
        self.lm_max_qlm = self.field.lm_max_qlm
        self.lm_max_unl = kwargs['lm_max_unl']
        self.estimator_type = kwargs['estimator_type']
        self.Nmf = kwargs['Nmf']
        self.version = kwargs['version']
        self.zbounds = kwargs['zbounds']
        self.cpp = kwargs['cpp']
        self.Lmin = kwargs['Lmin']
        self.blt_pert = kwargs['blt_pert']
        self.lm_max_blt = kwargs['lm_max_blt']
        self.blt_cacher = kwargs['blt_cacher']
        self.obd_libdir = kwargs['obd_libdir']
        self.obd_rescale = kwargs['obd_rescale']
        self.sht_threads = kwargs['sht_threads']
        self.cg_tol = kwargs['cg_tol']
        self.beam = kwargs['beam']
        self.OBD = kwargs['OBD']
        self.lmin_teb = kwargs['lmin_teb']
        self.TEMP = kwargs['TEMP']
        self.simidxs_mf = kwargs['simidxs_mf']
        self.QE_subtract_meanfield = kwargs['QE_subtract_meanfield']
        self.chain_descr = kwargs['chain_descr']
        self.nivt_desc = kwargs['nivt_desc']
        self.nivp_desc = kwargs['nivp_desc']

        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.R_unl = lambda: qresp.get_response(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.lm_max_qlm[0])[0]


        # FIXME currently only used for testing filter integration. These QE filter are not used for QE reoconstruction, but will be in the near future when Plancklens dependency is dropped. 
        if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'ptt']:
            self.filter = self.get_filter()


    def get_sim_qlm(self, simidx):

        return self.qlms_dd.get_sim_qlm(self.estimator_key, int(simidx))


    def get_wflm(self, simidx):
        if self.estimator_key in ['ptt']:
            return lambda: alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p']:
            return lambda: np.array([alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])

  
    def get_R_unl(self):
        return qresp.get_response(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]


    def get_meanfield(self, simidx):
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.estimator_key, simidx))
        fn_mf = opj(self.libdir_QE, 'mf_allsims.npy')
        if self.Nmf > 1:
            # MC MF, and exclude the current simidx
            ret = self.qlms_dd.get_sim_qlm_mf(self.estimator_key, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
            np.save(fn_mf, ret) # plancklens already stores that in qlms_dd/ but I want to have this more conveniently without the naming gibberish
            if simidx in self.simidxs_mf:    
                ret = (ret - self.qlms_dd.get_sim_qlm(self.estimator_key, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
        return ret


    def get_plm(self, simidx, component='alpha', sub_mf=True):
        libdir_MAPidx = self.libdir_MAP(self.estimator_key, simidx, self.version)
        fn_plm = opj(libdir_MAPidx, 'phi_plm_it000.npy') # Note: careful, this one doesn't have a simidx, so make sure it ends up in a simidx_directory (like MAP)
        if not os.path.exists(fn_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.estimator_key, int(simidx))  #Unormalized quadratic estimate:
            if sub_mf and self.version != 'noMF':
                plm -= self.mf(int(simidx))  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R))
            plm = alm_copy(plm, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(plm, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(plm, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.lm_max_qlm[1], True)
            np.save(fn_plm, plm)

        return np.load(fn_plm)


    def get_response_meanfield(self):
        if self.estimator_key in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.estimator_key, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.lm_max_qlm[0])[0]
        else:
            mf_resp = np.zeros(self.lm_max_qlm[0] + 1, dtype=float)

        return mf_resp


    def get_meanfield_normalized(self, simidx):
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = qresp.get_response(self.estimator_key, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
        WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R))
        almxfl(mf_QE, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.cpp > 0, self.lm_max_qlm[1], True)

        return mf_QE


    def get_blt(self, simidx):
        def get_template_blm(it, it_e, lmaxb=1024, lmin_plm=1, perturbative=False):
            fn_blt = 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.estimator_key, simidx, 0, 0, self.lm_max_blt[0])
            fn_blt += 'perturbative' * perturbative      

            elm_wf = self.filter.transf
            assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt
            mmaxb = lmaxb
            dlm = self.get_hlm(it, 'p')
            self.hlm2dlm(dlm, inplace=True)
            almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
            if perturbative: # Applies perturbative remapping
                get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
                geom, sht_tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
                d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
                dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
                del dp, dm, d1
                elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
            else: # Applies full remapping (this will re-calculate the angles)
                ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm)
                elm, blm = ffi.lensgclm(np.array([elm_wf, np.zeros_like(elm_wf)]), self.mmax_filt, 2, lmaxb, mmaxb)

                self.blt_cacher.cache(fn_blt, blm)

            return blm
        
        fn_blt = opj(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.estimator_key, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            blt = get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, perturbative=self.blt_pert)
            np.save(fn_blt, blt)

        return np.load(fn_blt)


    def get_filter(self): 
        QE_filters = transform(self, QE_transformer())
        filter = transform(self, QE_filters())
        return filter
    

    def get_field(self, fieldname, simidx):
        if fieldname == 'deflection':
            self.get_plm(simidx)
            self.get_wlm(simidx)
        elif fieldname == 'birefringence':
            self.get_olm(simidx)


    def get_meanfield_field(self, fieldname, estimator_key):
        if fieldname == 'deflection':
            mf_sims = np.unique(np.array([]) if not 'noMF' in self.version else np.array([]))
            mf0_p = self.qlms_dd.get_sim_qlm_mf('p' + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:
            mf0_o = self.qlms_dd.get_sim_qlm_mf('x' + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:
            return mf0_p, mf0_o
        elif fieldname == 'birefringence':
            return  self.qlms_dd.get_sim_qlm_mf('a' + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:

        # for field in self.fields:
            # return field
        # return self.field.get_component(simidx)
    
