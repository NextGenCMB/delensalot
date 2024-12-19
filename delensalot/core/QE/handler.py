
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
        # It does not quite aline well with the MAP classes, as the MAP equivalent is per simidx.
        self.ID = QE_search_desc["ID"]
        self.field = QE_search_desc['field']
        self.fq = filterqest(QE_search_desc['QE_filterqest_desc'])

        self.estimator_key = QE_search_desc['estimator_key']
        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.simidxs = QE_search_desc['simidxs']
        self.simidxs_mf = QE_search_desc['simidxs_mf']

    
    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx):
        #TODO add cacher and connect to field class
        #flow: if not cached, load file. if file does not exist, get qlm, update qlm, save qlm, cache qlm
        qlm = self.qlms.get_sim_qlm(self.estimator_key, int(simidx))  #Unormalized quadratic estimate
        self.field.update_qlm(qlm)
        return self.field
    

    def get_klm(self, simidx, subtract_meanfield):
        #TODO add cacher and connect to field class
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
        pass


    def get_meanfield_qlm(self, estimator_key, simidxs):
        # TODO add caching, and connect it to the field class
        return self.qlms.get_sim_qlm_mf(estimator_key, simidxs)  # Mean-field to subtract on the first iteration:
        

    def get_meanfield_klm(self, estimator_key, simidx):
        # TODO add caching, and connect it to the field class
        mf_QE = copy.deepcopy(self.get_meanfield_qlm(estimator_key, self.simidxs_mf))
        R = self.fq.get_response_len(estimator_key, self.lm_max_ivf[0])[0]
        WF = self.field.CLfid * cli(self.field.CLfid + cli(R))
        almxfl(mf_QE, cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.field.CLfid > 0, self.lm_max_qlm[1], True)

        return mf_QE
    

    def get_wflm(self, simidx):
        self.fq.get_wflm(simidx)


    def get_ivf(self, simidx):
        self.fq.get_ivf(simidx)
  

    def get_response_unl(self, estimator_key):
        return qresp.get_response(estimator_key, self.lm_max_ivf[0], estimator_key[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]
    

    def get_response_len(self, estimator_key):
        return qresp.get_response(estimator_key, self.lm_max_ivf[0], estimator_key[0], self.cls_len, self.cls_len, self.fteb_len, lmax_qlm=self.lm_max_qlm[0])[0]