import numpy as np
import os
from os.path import join as opj

from delensalot.utils import cli
from delensalot.core.QE import filterqest
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy

from delensalot.core.QE import field as QE_field

class base:
    def __init__(self, CLfids, estimator_key, QE_filterqest_desc, ID='generic', libdir=None, simidxs_mf=[], subtract_meanfield=True, init_filterqest=False):
        self.estimator_key = estimator_key
        self.CLfids = CLfids
        self.simidxs_mf = simidxs_mf
        self.subtract_meanfield = subtract_meanfield
        self.ID = ID or 'generic'
        oek = list(estimator_key.values())[0]
        keystring = oek if len(oek) == 1 else '_'+oek.split('_')[-1] if "_" in oek else oek[-2:]
        self.libdir = libdir or opj(os.environ['SCRATCH'], 'QE_search_generic', keystring, self.ID)
        field_desc = {
            "ID": self.ID,
            "libdir": opj(self.libdir, 'estimate'),
            'CLfids': CLfids,
            'component': list(self.estimator_key.keys()),
        }
        self.secondary = QE_field.secondary(field_desc)
        self.fq = filterqest.base(**QE_filterqest_desc)
        if init_filterqest: self.init_filterqest()

        self.comp2idx = {comp: idx for idx, comp in enumerate(self.secondary.component)}


    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, simidx, component=None):
        if component is None:
            return np.array([self.get_qlm(simidx, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]
        if not self.secondary.is_cached(simidx, component):
            qlm = self.qlms.get_sim_qlm(self.estimator_key[component], int(simidx))  #Unormalized quadratic estimate
            self.secondary.cache_qlm(qlm, simidx, component=component)
        return self.secondary.get_qlm(simidx, component)
    

    def get_est(self, simidx, component=None, subtract_meanfield=None, scale='k'):
        if component is None:
            return np.array([self.get_est(simidx, component, subtract_meanfield, scale).squeeze() for component in self.secondary.component])
        if isinstance(component, list):
            return np.array([self.get_est(simidx, comp, subtract_meanfield, scale).squeeze() for comp in component])
        
        if not self.secondary.is_cached(simidx, component, 'klm'):
            qlm = self.get_qlm(simidx, component)
            Lmax = Alm.getlmax(qlm.size, None)
            _submf = subtract_meanfield or self.subtract_meanfield
            if _submf and len(self.simidxs_mf)>2: #NOTE >2 is really just a lower bound.
                mf_qlm = self.get_qmflm(self.simidxs_mf, component=component)
                qlm -= mf_qlm
            R = self.get_response_len(component)
            WF = self.secondary.CLfids[component*2][:Lmax+1] * cli(self.secondary.CLfids[component*2][:Lmax+1] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            klm = alm_copy(qlm, None, Lmax, Lmax)
            almxfl(klm, cli(R), Lmax, True) # Normalized QE
            almxfl(klm, WF, Lmax, True) # Wiener-filter QE
            almxfl(klm, self.secondary.CLfids[component*2][:Lmax+1] > 0, Lmax, True)
            self.secondary.cache_klm(np.atleast_2d(klm), simidx, component)
        return self.secondary.get_est(simidx, component, scale) 


    def get_qmflm(self, simidxs, component=None):
        # TODO connect it to the field class
        if component is None:
            return np.array([self.get_qmflm(simidxs, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]
        return self.qlms.get_sim_qlm_mf(self.estimator_key[component], simidxs)
        

    def get_kmflm(self, simidx, component=None, scale='k'):
        # TODO connect it to the field class
        if component is None:
            return np.array([self.get_kmflm(simidx, component) for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]

        if len(self.simidxs_mf) <= 2: # NOTE this is really just a lower bound
            return np.zeros(Alm.getsize(*self.fq.lm_max_qlm), dtype=complex)
        kmflm = self.get_qmflm(self.simidxs_mf, component=component)
        R = self.get_response_len(component)
        WF = self.secondary.CLfids[component*2] * cli(self.secondary.CLfids[component*2] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        kmflm = alm_copy(kmflm, None, self.fq.lm_max_qlm[0], self.fq.lm_max_qlm[1])
        almxfl(kmflm, cli(R), self.fq.lm_max_qlm[1], True) # Normalized QE
        almxfl(kmflm, WF, self.fq.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(kmflm, self.secondary.CLfids[component*2] > 0, self.fq.lm_max_qlm[1], True)
        # FIXME correct removal
        kmflm -= self.get_est(simidx, component=component)*1/(1-len(self.simidxs_mf))
        return np.array(kmflm)
    

    def get_wflm(self, simidx, lm_max):
        # NOTE returns the same for each component so can just take the first key here
        return self.fq.get_wflm(simidx, list(self.estimator_key.values())[0], lm_max)


    def get_ivflm(self, simidx, lm_max):
        # NOTE returns the same for each component so can just take the first key here
        return self.fq.get_ivflm(simidx, list(self.estimator_key.values())[0], lm_max)
    

    def get_response_unl(self, component, scale='p'):
        return self.fq.get_response_unl(self.estimator_key[component], self.estimator_key[component][0], self.fq.lm_max_qlm[0])[self.comp2idx[component]]
    

    def get_response_len(self, component, scale='p'):
        return self.fq.get_response_len(self.estimator_key[component], self.estimator_key[component][0], self.fq.lm_max_qlm[0])[self.comp2idx[component]]
    

    def isdone(self, simidx, component):
        if self.secondary.is_cached(simidx, component, 'klm'):
            return 0
        else:
            return -1