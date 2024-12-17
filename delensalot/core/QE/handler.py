
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
        self.fq = filterqest(QE_search_desc['QE_filterqest_desc'])

        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.field = QE_search_desc['field']
        self.template_operator = QE_search_desc['template_operator']
        # TODO make them per field
        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.qlms = None

    
    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx, field):
        #flow: if not cached, load file. if file does not exist, get qlm, update qlm, save qlm, cache qlm
        qlm = self.qlms.get_sim_qlm(self.estimator_key[field], int(simidx))  #Unormalized quadratic estimate
        field.update_qlm(qlm)
        return self.field
    

    def get_klm(self, simidx, subtract_meanfield):
        #flow: if not cached, load file. if file does not exist, get qlm, update qlm, save qlm, update klm, save klm, cache klm
        # self.estimate_fields(self)
        # for qfield, kfield in zip(self.qfields, self.kfields):
        #     if sub_mf and self.version != 'noMF':
        #         kfield.value = self.mf(qfield.id, simidx)  # MF-subtracted unnormalized QE
        #     R = self.get_response_len(self.estimator_key[qfield.ID])[0]
        #     WF = kfield.CLfid * cli(kfield.CLfid + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        #     kfield.value = alm_copy(kfield.value, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
        #     almxfl(kfield.value, cli(R), self.lm_max_qlm[1], True) # Normalized QE
        #     almxfl(kfield.value, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        #     almxfl(kfield.value, kfield.CLfid > 0, self.lm_max_qlm[1], True)
        #     self.kfield.update_klm(kfield.value)
        # return self.kfields

        # # calc normalized klm and store it in the respective directory if not already cached
        # _fn = self.QE_searchs[field].klm_fns[component].format(idx=simidx)
        # if not self.QE_searchs[field].cacher.is_cached(_fn):
        #     self.QE_searchs[field].get_meanfield(component)
        #     if subtract_meanfield:
        #         # TODO remove the current simidx from the meanfield calculation
        #         # qlm[simidx] - meanfield(simidx) # this is a placeholder, the actual implementation will be more complex
        #         pass
        #     # TODO normalize the qlms to klms
        #     # klms = cli(response) etc.
        #     klms = None
        #     self.QE_searchs[simidx].cacher.cache(_fn, klms)
        # return self.QE_searchs[simidx].cacher.load(_fn)
        pass
    

    def get_template(self, simidx, dlm, field):
        #flow if not cached, get dlm. dlm = klm
        dlm = self.field.get_klm(simidx, field)
        self.template_operator.update_field(dlm)
        return self.template_operator.act(field)


    def get_meanfield_qlm(self, fieldname, estimator_key, simidxs_mf, component=None):
        #flow: check cached and file, calc meanfield: get qlm for each simidx
        if component is None:
            return [self.get_meanfield(fieldname, estimator_key, component) for component in self.components]
        if fieldname == 'deflection':
            qmf = self.qlms.get_sim_qlm_mf(component + estimator_key[1:], simidxs_mf)  # Mean-field to subtract on the first iteration:
            return qmf
        elif fieldname == 'birefringence':
            return self.qlms.get_sim_qlm_mf(component + estimator_key[1:], simidxs_mf)  # Mean-field to subtract on the first iteration:
        

    def get_meanfield_klm(self, simidx, field, component):
        #flow: check cached and file, calc meanfield: get qlm for each simidx
        # TODO make this per field
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = self.get_response_len(self.estimator_key, self.lm_max_ivf[0], self.estimator_key[0])[0]
        WF = self.field.fiducial * cli(self.field.fiducial + cli(R))
        almxfl(mf_QE, cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.field.fiducial > 0, self.lm_max_qlm[1], True)

        return mf_QE
    

    def get_wflm(self, simidx):
        self.fq.get_wflm(simidx)


    def get_ivf(self, simidx):
        self.fq.get_ivf(simidx)
  

    def get_response_unl(self, estimator_key):
        return qresp.get_response(estimator_key, self.lm_max_ivf[0], estimator_key[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]
    

    def get_response_len(self, estimator_key):
        return qresp.get_response(estimator_key, self.lm_max_ivf[0], estimator_key[0], self.cls_len, self.cls_len, self.fteb_len, lmax_qlm=self.lm_max_qlm[0])[0]