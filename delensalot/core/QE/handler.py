
        # [self.filter.set_qlms_lib(self.qlms_dd) for self.QE_search in self.QE_search]
        # [self.filter.set_filter_lib(self.ivf) for self.QE_search in self.QE_search]import os
import os, copy
from os.path import join as opj
import numpy as np

from plancklens import qresp, qest

from delensalot.config.visitor import transform, transform3d
from delensalot.utils import cli

from delensalot.core import cachers
from delensalot.core.QE import filterqest

from delensalot.utility import utils_qe
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam

class base:
    def __init__(self, kwargs):
        # This class is for all fields. It manages the filter and qest libs, nothing else.
        self.estimator_key = kwargs['estimator_key']
        self.version = kwargs['version']
        self.fq = filterqest(kwargs['filter_desc'])

        self.cls_len = kwargs['cls_len']
        self.cls_unl = kwargs['cls_unl']

        self.field = kwargs['field']
        self.template_operator = kwargs['template_operator']

        self.mf
        self.plm
        self.lm_max_ivf
        
        self.lm_max_qlm


        self.libdir_QE
        self.mmax_filt
        self.lmax_filt
        self.hlm2dlm
        self.mmax_qlm
        self.fq.ffi
        self.libdir
        self.Nmf
        self.simidxs_mf

        # TODO make them per field
        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        

    def estimate_field(self, simidx, field):
        qlm = self.fq.qlms_dd.get_sim_qlm(self.estimator_key[field], int(simidx))  #Unormalized quadratic estimate
        field.update_klm(qlm)
        return self.field
    

    def get_template(self, dlm, field):
        self.template_operator.update_field(dlm)
        return self.template_operator.act(field)


    def get_sim_qlm(self, simidx):
        return self.fq.qlms_dd.get_sim_qlm(self.estimator_key, int(simidx))
    

    def get_wflm(self, simidx):
        self.fq.get_wflm(simidx)
  

    def get_response_unl(self, estimator_key):
        self.fq.get_response_unl(estimator_key)


    def get_klm(self, simidx):
        for field in self.fields:
            field.get_klm(simidx)


    def calc_fields_normalized(self, sub_mf=True, simidx):
        self.estimate_fields(self)
        for qfield, kfield in zip(self.qfields, self.kfields):
            if sub_mf and self.version != 'noMF':
                kfield.value = self.mf(qfield.id, simidx)  # MF-subtracted unnormalized QE
            R = self.fq.qresp.get_response(self.estimator_key[qfield.ID], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            WF = kfield.CLfid * cli(kfield.CLfid + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            kfield.value = alm_copy(kfield.value, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(kfield.value, cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(kfield.value, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(kfield.value, kfield.CLfid > 0, self.lm_max_qlm[1], True)
            self.kfield.update_klm(kfield.value)
        return self.kfields


    def get_response_meanfield(self, field, component):
        if self.estimator_key in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = self.fq.qresp.get_mf_resp(self.estimator_key, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.lm_max_qlm[0])[0]
        else:
            mf_resp = np.zeros(self.lm_max_qlm[0] + 1, dtype=float)

        return mf_resp


    def get_meanfield_normalized(self, simidx, field, component):
        # TODO make this per field
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = qresp.get_response(self.estimator_key, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
        WF = self.field.fiducial * cli(self.field.fiducial + cli(R))
        almxfl(mf_QE, cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.field.fiducial > 0, self.lm_max_qlm[1], True)

        return mf_QE
    

    def get_template(self, field, operator):
        self.blt_cacher  = cachers.cacher_npy(opj(self.libdir, 'BLT'))
        self.blt_pert
        self.blt_cacher
        self.lm_max_blt # poo
        self.Lmin
        return operator.act(field)


    def get_blt(self, simidx):
        def get_template_blm(it, it_e, lmaxb=1024, lmin_plm=1, perturbative=False):
            fn_blt = 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.estimator_key, simidx, 0, 0, self.lm_max_blt[0])
            fn_blt += 'perturbative' * perturbative      

            elm_wf = self.fq.transf
            assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt
            mmaxb = lmaxb
            dlm = self.get_hlm(it, 'p')
            self.hlm2dlm(dlm, inplace=True)
            almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
            if perturbative: # Applies perturbative remapping
                get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
                geom, sht_tr = self.fq.ffi.geom, self.fq.ffi.sht_tr
                d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
                dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
                del dp, dm, d1
                elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
            else: # Applies full remapping (this will re-calculate the angles)
                ffi = self.fq.ffi.change_dlm([dlm, None], self.mmax_qlm)
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


    def get_meanfield(self, fieldname, estimator_key, component=None):
        if component is None:
            return [self.get_meanfield(fieldname, estimator_key, component) for component in self.components]
        if fieldname == 'deflection':
            mf_sims = np.unique(np.array([]) if not 'noMF' in self.version else np.array([]))
            qmf = self.fq.qlms_dd.get_sim_qlm_mf(component + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration:
            return qmf
        elif fieldname == 'birefringence':
            return self.fq.qlms_dd.get_sim_qlm_mf(component + estimator_key[1:], mf_sims)  # Mean-field to subtract on the first iteration: