
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

component2plancklensk = {'alpha': "p", 'beta': "a", 'omega': "x"}

class base:
    def __init__(self, QE_search_desc):
        # This class is for a single field, but all simidxs. It manages the filter and qest libs, nothing else.
        # It does not quite aline well with the MAP classes, as the MAP equivalent is per simidx.
        self.fq = filterqest.base(QE_search_desc['QE_filterqest_desc'])
        self.libdir = QE_search_desc['libdir']

        self.ID = QE_search_desc["ID"]
        self.field = QE_search_desc["field"]
        self.estimator_key = QE_search_desc['estimator_key']
        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.simidxs = QE_search_desc['simidxs']
        self.simidxs_mf = QE_search_desc['simidxs_mf']
        self.subtract_meanfield = QE_search_desc['subtract_meanfield']

    
    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx, component=None):
        if component is None:
            return [self.get_qlm(simidx, component) for component in self.field.components.split("_")]
        qlm = self.qlms.get_sim_qlm(component2plancklensk[component]+self.estimator_key[1:], int(simidx))  #Unormalized quadratic estimate
        self.field.cache_qlm(qlm, simidx, component=component)
        return qlm
    

    def get_klm(self, simidx, subtract_meanfield=None, component=None):
        if component is None:
            return [self.get_klm(simidx, subtract_meanfield, component) for component in self.field.components.split("_")]
        
        qlm = self.get_qlm(simidx, component)
        _submf = self.subtract_meanfield if subtract_meanfield is None else subtract_meanfield
        if _submf:
            mf_qlm = self.get_meanfield_qlm(self.simidxs_mf, component=component)
            qlm -= mf_qlm

        R = self.get_response_len(component)
        WF = self.field.CLfids[component] * cli(self.field.CLfids[component] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        klm = alm_copy(qlm, None, self.field.lm_max[0], self.field.lm_max[1])
        almxfl(klm, cli(R), self.field.lm_max[1], True) # Normalized QE
        almxfl(klm, WF, self.field.lm_max[1], True) # Wiener-filter QE
        almxfl(klm, self.field.CLfids[component] > 0, self.field.lm_max[1], True)
        self.field.cache_klm(klm, simidx, component)
        return klm


    def get_meanfield_qlm(self, simidxs, component=None):
        # TODO add caching, and connect it to the field class
        if component is None:
            return [self.get_meanfield_qlm(simidxs, component) for component in self.field.components.split("_")]
        return self.qlms.get_sim_qlm_mf(self.estimator_key, simidxs)
        

    def get_meanfield_klm(self, simidx):
        # TODO add caching, and connect it to the field class
        mf_QE = copy.deepcopy(self.get_meanfield_qlm(self.estimator_key, self.simidxs_mf))
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
        return qresp.get_response(component2plancklensk[estimator_key], self.fq.lm_max_ivf[0], estimator_key[0], self.cls_unl, self.cls_unl, self.fq.fteb_unl, lmax_qlm=self.field.lm_max[0])[0]
    

    def get_response_len(self, component):
        return qresp.get_response(component2plancklensk[component]+self.estimator_key[1:], self.fq.lm_max_ivf[0], component2plancklensk[component], self.cls_len, self.cls_len, self.fq.ftebl_len, lmax_qlm=self.field.lm_max[0])[0]