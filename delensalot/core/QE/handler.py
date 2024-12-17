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
        self.fields = kwargs['QE_fields']
        self.simidxs = kwargs['simidxs']
        self.estimator_key = kwargs['estimator_key']
        self.qe_filter_directional = kwargs['qe_filter_directional']
        self.libdir = kwargs['libdir']
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
        self.mf_fn = kwargs['mf_fn']

        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.R_unl = lambda: qresp.get_response(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.lm_max_qlm[0])[0]


        # FIXME currently only used for testing filter integration. These QE filter are not used for QE reoconstruction, but will be in the near future when Plancklens dependency is dropped. 
        if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'ptt']:
            self.filter = self.get_filter()


    def set_filter_lib(self, filter):
        self.ivf = filter


    def set_qlms_lib(self, qlms_dd):
        self.qlms_dd = qlms_dd


    def get_sim_qlm(self, simidx):
        return self.qlms_dd.get_sim_qlm(self.estimator_key, int(simidx))


    def get_wflm(self, simidx):
        if self.estimator_key in ['ptt']:
            return lambda: alm_copy(self.ivf.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivf.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p']:
            return lambda: np.array([alm_copy(self.ivf.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivf.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])

  
    def get_R_unl(self, estimator_key):
        return qresp.get_response(estimator_key, self.lm_max_ivf[0], estimator_key[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]


    def get_klm(self, simidx):
        for field in self.fields:
            field.get_klm(simidx)


    def estimate_fields(self):
        for qfield in self.qfields:
            if qfield.value is None:
                qlm = self.qlms_dd.get_sim_qlm(self.estimator_keys[qfield.ID], self.simidx)  #Unormalized quadratic estimate
                qfield.update_klm(qlm)
        return self.qfields


    def calc_fields_normalized(self, sub_mf =True):
        self.estimate_fields(self)
        for qfield, kfield in zip(self.qfields, self.kfields):
            if sub_mf and self.version != 'noMF':
                kfield.value = self.mf(qfield.id, self.simidx)  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.estimator_keys[qfield.ID], self.lm_max_ivf[0], self.estimator_keys[qfield.ID], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            WF = kfield.CLfid * pl_utils.cli(kfield.CLfid + pl_utils.cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            kfield.value = alm_copy(kfield.value, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(kfield.value, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(kfield.value, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(kfield.value, kfield.CLfid > 0, self.lm_max_qlm[1], True)
            self.kfield.update_klm(kfield.value)
        return self.kfields


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
    

    def get_field(self, fieldname, simidx):
        if fieldname == 'deflection':
            self.get_plm(simidx)
            self.get_wlm(simidx)
        elif fieldname == 'birefringence':
            self.get_olm(simidx)


    def get_qmeanfield(self, fieldname, estimator_key, component=None):
        if component is None:
            return [self.get_meanfield(fieldname, estimator_key, component) for component in self.components]
        if fieldname == 'deflection':
            mf_sims = np.unique(np.array([]) if not 'noMF' in self.version else np.array([]))
            qmf = self.qlms_dd.get_sim_qlm_mf(component + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:
            return qmf
        elif fieldname == 'birefringence':
            return self.qlms_dd.get_sim_qlm_mf(component + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:


    def get_kmeanfield(self):
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.estimator_key, self.simidx))
        fn_mf = opj(self.libdir_QE, 'mf_allsims.npy')
        if self.Nmf > 1:
            # MC MF, and exclude the current simidx
            ret = self.qlms_dd.get_sim_qlm_mf(self.estimator_key, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
            np.save(fn_mf, ret) # plancklens already stores that in qlms_dd/ but I want to have this more conveniently without the naming gibberish
            if self.simidx in self.simidxs_mf:    
                ret = (ret - self.qlms_dd.get_sim_qlm(self.estimator_key, int(self.simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
        return ret
    
