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

components2plancklensk = {'p': "p", 'f': "a", 'w': "x"}

class base:
    def __init__(self, QE_search_desc):
        # This class is for a single field, but all simidxs. It manages the filter and qest libs, nothing else.
        # It does not quite aline well with the MAP classes, as the MAP equivalent is per simidx.
        self.fq = filterqest.base(QE_search_desc['QE_filterqest_desc'])
        self.libdir = QE_search_desc['libdir']

        self.ID = QE_search_desc["ID"]
        self.secondary = QE_search_desc["secondary"]
        self.estimator_key = QE_search_desc['estimator_key']
        self.cls_len = QE_search_desc['cls_len']
        self.cls_unl = QE_search_desc['cls_unl']

        self.simidxs = QE_search_desc['simidxs']
        self.simidxs_mf = QE_search_desc['simidxs_mf']
        self.subtract_meanfield = QE_search_desc['subtract_meanfield']

        # NOTE this does not work for minimum variance estimator, as get_sim_qlm() 'key' argument is wrong.

    
    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx, components=None):
        if components is None:
            return [self.get_qlm(simidx, components) for components in self.secondary.components]
        if isinstance(components, list):
            components = components[0]
        
        if not self.secondary.is_cached(simidx, components):
            qlm = self.qlms.get_sim_qlm(components2plancklensk[components]+self.estimator_key[1:], int(simidx))  #Unormalized quadratic estimate
            self.secondary.cache_qlm(qlm, simidx, components=components)
        return self.secondary.get_qlm(simidx, components)
    

    def get_klm(self, simidx, subtract_meanfield=None, components=None):
        if components is None:
            return np.array([self.get_klm(simidx, subtract_meanfield, components).squeeze() for components in self.secondary.components])
        if isinstance(components, list):
            components = components[0]
        
        if not self.secondary.is_cached(simidx, components, 'klm'):
            qlm = self.get_qlm(simidx, components)
            _submf = self.subtract_meanfield if subtract_meanfield is None else subtract_meanfield
            if _submf and len(self.simidxs_mf)>1: #NOTE >1 is really just a lower bound.
                mf_qlm = self.get_qmflm(self.simidxs_mf, components=components)
                qlm -= mf_qlm

            R = self.get_response_len(components)
            WF = self.secondary.CLfids[components*2][:self.secondary.lm_max[0]+1] * cli(self.secondary.CLfids[components*2][:self.secondary.lm_max[0]+1] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            klm = alm_copy(qlm, None, self.secondary.lm_max[0], self.secondary.lm_max[1])
            almxfl(klm, cli(R), self.secondary.lm_max[1], True) # Normalized QE
            almxfl(klm, WF, self.secondary.lm_max[1], True) # Wiener-filter QE
            almxfl(klm, self.secondary.CLfids[components*2][:self.secondary.lm_max[0]+1] > 0, self.secondary.lm_max[1], True)
            self.secondary.cache_klm(np.atleast_2d(klm), simidx, components)
        return self.secondary.get_klm(simidx, components)


    def get_qmflm(self, simidxs, components=None):
        # TODO connect it to the field class
        if components is None:
            return np.array([self.get_qmflm(simidxs, components) for components in self.secondary.components])
        if isinstance(components, list):
            components = components[0]
        return self.qlms.get_sim_qlm_mf(self.estimator_key, simidxs)
        

    def get_kmflm(self, simidx, components=None):
        # TODO connect it to the field class
        if components is None:
            return np.array([self.get_kmflm(simidx, components) for components in self.secondary.components])
        if isinstance(components, list):
            components = components[0]

        if len(self.simidxs_mf) == 0:
            return np.zeros(Alm.getsize(*self.secondary.lm_max), dtype=complex)
        kmflm = self.get_qmflm(self.simidxs_mf, components=components)
        R = self.get_response_len(components)
        WF = self.secondary.CLfids[components*2] * cli(self.secondary.CLfids[components*2] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        kmflm = alm_copy(kmflm, None, self.secondary.lm_max[0], self.secondary.lm_max[1])
        almxfl(kmflm, cli(R), self.secondary.lm_max[1], True) # Normalized QE
        almxfl(kmflm, WF, self.secondary.lm_max[1], True) # Wiener-filter QE
        almxfl(kmflm, self.secondary.CLfids[components*2] > 0, self.secondary.lm_max[1], True)
        # FIXME correct removal
        kmflm -= self.get_klm(simidx, components=components)*1/(1-len(self.simidxs_mf))
        return np.array(kmflm)
    

    def get_wflm(self, simidx):
        return self.fq.get_wflm(simidx)


    def get_ivflm(self, simidx):
        return self.fq.get_ivflm(simidx)
  

    def get_response_unl(self, components):
        return qresp.get_response(components2plancklensk[components]+self.estimator_key[1:], self.fq.lm_max_ivf[0], components2plancklensk[components], self.cls_unl, self.cls_unl, self.fq.ftebl_unl, lmax_qlm=self.secondary.lm_max[0])[0]
    

    def get_response_len(self, components):
        return qresp.get_response(components2plancklensk[components]+self.estimator_key[1:], self.fq.lm_max_ivf[0], components2plancklensk[components], self.cls_len, self.cls_len, self.fq.ftebl_len, lmax_qlm=self.secondary.lm_max[0])[0]