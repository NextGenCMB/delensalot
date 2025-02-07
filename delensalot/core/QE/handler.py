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

component2plancklensk = {'p': "p", 'f': "a", 'w': "x"}
id2plancklenssec = {'lensing': 'p', 'birefringence': 'a'}

class base:
    def __init__(self, QE_search_desc):
        # This class is for a single field, but all simidxs. It manages the filter and qest libs, nothing else.
        # It does not quite aline well with the MAP classes, as the MAP equivalent is per simidx.
        self.ID = QE_search_desc["ID"]
        self.estimator_key = QE_search_desc['estimator_key']
        self.estimator_key = id2plancklenssec[self.ID] + self.estimator_key[1:]
        QE_search_desc['QE_filterqest_desc'].update({'estimator_key': self.estimator_key})
        self.fq = filterqest.base(QE_search_desc['QE_filterqest_desc'])
        self.libdir = QE_search_desc['libdir']

        self.secondary = QE_search_desc["secondary"]


        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.simidxs = QE_search_desc['simidxs']
        self.simidxs_mf = QE_search_desc['simidxs_mf']
        self.subtract_meanfield = QE_search_desc['subtract_meanfield']
        self.comp2idx = {comp: idx for idx, comp in enumerate(self.secondary.component)}

        # NOTE this does not work for minimum variance estimator, as get_sim_qlm() 'key' argument is wrong.

    
    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx, component=None):
        if component is None:
            return np.array([self.get_qlm(simidx, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]
        if not self.secondary.is_cached(simidx, component):
            qlm = self.qlms.get_sim_qlm(component2plancklensk[component]+self.estimator_key[1:], int(simidx))  #Unormalized quadratic estimate
            self.secondary.cache_qlm(qlm, simidx, component=component)
        return self.secondary.get_qlm(simidx, component)
    

    def get_est(self, simidx, scale='k', subtract_meanfield=None, component=None):
        if component is None:
            return np.array([self.get_est(simidx, scale, subtract_meanfield, component).squeeze() for component in self.secondary.component])
        if isinstance(component, list):
            return np.array([self.get_est(simidx, scale, subtract_meanfield, comp).squeeze() for comp in component])
        
        if not self.secondary.is_cached(simidx, component, 'klm'):
            qlm = self.get_qlm(simidx, component)
            _submf = self.subtract_meanfield if subtract_meanfield is None else subtract_meanfield
            if _submf and len(self.simidxs_mf)>1: #NOTE >1 is really just a lower bound.
                mf_qlm = self.get_qmflm(self.simidxs_mf, component=component)
                qlm -= mf_qlm

            R = self.get_response_len(component)
            WF = self.secondary.CLfids[component*2][:self.secondary.lm_max[0]+1] * cli(self.secondary.CLfids[component*2][:self.secondary.lm_max[0]+1] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            klm = alm_copy(qlm, None, self.secondary.lm_max[0], self.secondary.lm_max[1])
            almxfl(klm, cli(R), self.secondary.lm_max[1], True) # Normalized QE
            almxfl(klm, WF, self.secondary.lm_max[1], True) # Wiener-filter QE
            almxfl(klm, self.secondary.CLfids[component*2][:self.secondary.lm_max[0]+1] > 0, self.secondary.lm_max[1], True)
            self.secondary.cache_klm(np.atleast_2d(klm), simidx, component)
        return self.secondary.get_est(simidx, component)


    def get_qmflm(self, simidxs, component=None):
        # TODO connect it to the field class
        if component is None:
            return np.array([self.get_qmflm(simidxs, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]
        return self.qlms.get_sim_qlm_mf(self.estimator_key, simidxs)
        

    def get_kmflm(self, simidx, component=None, scale='k'):
        # TODO connect it to the field class
        if component is None:
            return np.array([self.get_kmflm(simidx, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]

        if len(self.simidxs_mf) == 0:
            return np.zeros(Alm.getsize(*self.secondary.lm_max), dtype=complex)
        kmflm = self.get_qmflm(self.simidxs_mf, component=component)
        R = self.get_response_len(component)
        WF = self.secondary.CLfids[component*2] * cli(self.secondary.CLfids[component*2] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        kmflm = alm_copy(kmflm, None, self.secondary.lm_max[0], self.secondary.lm_max[1])
        almxfl(kmflm, cli(R), self.secondary.lm_max[1], True) # Normalized QE
        almxfl(kmflm, WF, self.secondary.lm_max[1], True) # Wiener-filter QE
        almxfl(kmflm, self.secondary.CLfids[component*2] > 0, self.secondary.lm_max[1], True)
        # FIXME correct removal
        kmflm -= self.get_est(simidx, component=component)*1/(1-len(self.simidxs_mf))
        return np.array(kmflm)
    

    def get_wflm(self, simidx):
        return self.fq.get_wflm(simidx)


    def get_ivflm(self, simidx):
        return self.fq.get_ivflm(simidx)
  

    def get_response_unl(self, component):
        return qresp.get_response(self.estimator_key, self.fq.lm_max_ivf[0], self.estimator_key[0], self.cls_unl, self.cls_unl, self.fq.ftebl_unl, lmax_qlm=self.secondary.lm_max[0])[self.comp2idx[component]] 
    

    def get_response_len(self, component):
        return qresp.get_response(self.estimator_key, self.fq.lm_max_ivf[0], self.estimator_key[0], self.cls_len, self.cls_len, self.fq.ftebl_len, lmax_qlm=self.secondary.lm_max[0])[self.comp2idx[component]]