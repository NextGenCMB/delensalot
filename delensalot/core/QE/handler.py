
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
    def __init__(self, QE_search_desc):
        # This class is for a single field, but all simidxs. It manages the filter and qest libs, nothing else.
        self.estimator_key = QE_search_desc['estimator_key']
        self.version = QE_search_desc['version']
        self.fq = filterqest(QE_search_desc['QE_filterqest_desc'])

        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.field = QE_search_desc['field']
        self.template_operator = QE_search_desc['template_operator']
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
            R = self.fq.qresp.get_response_len(self.estimator_key[qfield.ID])[0]
            WF = kfield.CLfid * cli(kfield.CLfid + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            kfield.value = alm_copy(kfield.value, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(kfield.value, cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(kfield.value, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(kfield.value, kfield.CLfid > 0, self.lm_max_qlm[1], True)
            self.kfield.update_klm(kfield.value)
        return self.kfields


    def get_meanfield_normalized(self, simidx, field, component):
        # TODO make this per field
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = self.fq.qresp.get_response_len(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0])[0]
        WF = self.field.fiducial * cli(self.field.fiducial + cli(R))
        almxfl(mf_QE, cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.field.fiducial > 0, self.lm_max_qlm[1], True)

        return mf_QE
    

    def get_template(self, field):
        self.blt_cacher  = cachers.cacher_npy(opj(self.libdir, 'BLT'))
        return self.template_operator.act(field)


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