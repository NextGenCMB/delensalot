import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
import os
from os.path import join as opj

from delensalot.utils import cli
from delensalot.core.QE import filterqest
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, alm_copy_nd, almxfl_nd

from delensalot.core.QE import field


def rescale(arr, scale='p'):
    if scale == 'p':
        return arr
    elif scale == 'k':
        return arr * cli(0.5 * np.arange(arr.size) * np.arange(1, arr.size+1))**2
    else:
        raise ValueError('Unknown scale:', scale)
    
complist_lensing_template = ['p', 'w']
complist_lensing_template_idx = {val: i for i, val in enumerate(complist_lensing_template)}
complist_birefringence_template = ['f']

class Base:
    def __init__(self, CLfids, CLfidsNoLmin, estimator_key, QE_filterqest_desc, ID='generic', libdir=None, idxs_mf=[], subtract_meanfield=True, init_filterqest=False, qmflm_fn=None):
        self.estimator_key = estimator_key
        self.CLfids = CLfids
        self.CLfidsNoLmin = CLfidsNoLmin
        self.idxs_mf = idxs_mf
        self.subtract_meanfield = subtract_meanfield
        self.ID = ID or 'generic'
        self.qmflm_fn = qmflm_fn or {comp: None for comp in estimator_key.keys()}
        oek = list(estimator_key.values())[0]
        
        keystring = oek if len(oek) == 1 else '_'+oek.split('_')[-1] if "_" in oek else oek[-2:]
        self.libdir = libdir or opj(os.environ['SCRATCH'], 'QE_search_generic', keystring)
        if 'p' in estimator_key.keys() or 'w' in estimator_key.keys():
            component_ = [key for key in complist_lensing_template if key in self.estimator_key]
        elif 'f' in estimator_key.keys():
            component_ = ['f']
        field_desc = {
            "ID": self.ID,
            "libdir": opj(self.libdir, 'estimate'),
            'CLfids': CLfids,
            'component': component_,
        }
        self.secondary = field.Secondary(field_desc)
        QE_filterqest_desc.update({'libdir': opj(self.libdir)})
        self.fq = filterqest.PlancklensInterface(**QE_filterqest_desc)
        if init_filterqest: self.init_filterqest()

        self.chh = {comp: (
            self.CLfids[comp*2][:self.fq.lm_max_qlm[0]+1]
            * (0.5 * np.arange(self.fq.lm_max_qlm[0]+1) * np.arange(1,self.fq.lm_max_qlm[0]+2))**2
            if ('p' in estimator_key.keys() or 'w' in estimator_key.keys())
            else self.CLfids[comp*2][:self.fq.lm_max_qlm[0]+1]
        )for comp in self.secondary.component}

        self.comp2idx = {comp: idx for idx, comp in enumerate(self.secondary.component)}


    def init_filterqest(self):
        self.qlms = self.fq._init_filterqest()
        

    def get_qlm(self, idx, component=None):
        if component is None:
            return np.array([self.get_qlm(idx, component).squeeze() for component in self.secondary.component])
        if isinstance(component, list):
            component = component[0]
        if not self.secondary.is_cached(idx, component):
            qlm = self.qlms.get_sim_qlm(self.estimator_key[component], int(idx))  #Unormalized quadratic estimate
            self.secondary.cache_qlm(qlm, idx, component=component)
        return self.secondary.get_qlm(idx, component)
    

    def get_est(self, idx, component=None, subtract_meanfield=None, scale='k'):
        if component is None:
            return np.array([self.get_est(idx, component, subtract_meanfield, scale).squeeze() for component in self.secondary.component])
        if isinstance(component, list):
            return np.array([self.get_est(idx, comp, subtract_meanfield, scale).squeeze() for comp in component])
        
        if not self.secondary.is_cached(idx, component, 'klm'):
            qlm = self.get_qlm(idx, component)
            Lmax = Alm.getlmax(qlm.size, None)
            _submf = subtract_meanfield or self.subtract_meanfield
            if idx==0: print(f"(only printing idx 0) _submf = {_submf}")
            if _submf:
                mf_qlm = self.get_qmflm(idx, self.idxs_mf, component=component)
                qlm -= mf_qlm
            R = self.get_response_len(component)
            WF = self.secondary.CLfids[component*2][:Lmax+1] * cli(self.secondary.CLfids[component*2][:Lmax+1] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            klm = alm_copy(qlm, None, Lmax, Lmax)
            almxfl(klm, cli(R), Lmax, True) # Normalized QE
            almxfl(klm, WF, Lmax, True) # Wiener-filter QE
            almxfl(klm, self.secondary.CLfids[component*2][:Lmax+1] > 0, Lmax, True)
            self.secondary.cache_klm(np.atleast_2d(klm), idx, component)
        return self.secondary.get_est(idx, component, scale) 


    def get_qmflm(self, idx, idxs, component=None):
        if component is None:
            return np.array([self.get_qmflm(idx=idx, idxs=idxs, component=component) for component in self.secondary.component])
        if isinstance(component, list):
            return np.array([self.get_qmflm(idx=idx, idxs=idxs, component=comp).squeeze() for comp in component])
        if self.qmflm_fn[component] is not None:
            assert 0, "I think it works, but not sure I want to keep it"
            mf_qlm = np.atleast_2d(np.load(self.qmflm_fn[component]))
            print("MAKE SURE idxs for mf_qlm is correct!")
            idxs = np.arange(10)
            # FIXME if mf precalc is computed from same samples, need to remove that idx.. but we don't know the true len(idxs) here
            mf_qlm = (mf_qlm - self.get_qlm(idx, component)/len(idxs))*(len(idxs)/(len(idxs)-1))
            return mf_qlm
        else:
            mf_qlm = np.atleast_2d(self.qlms.get_sim_qlm_mf(self.estimator_key[component], idxs))
            mf_qlm = (mf_qlm - self.get_qlm(idx, component)/len(idxs))*(len(idxs)/(len(idxs)-1))
            return mf_qlm


    def get_kmflm(self, idx, idxs_mf=None, component=None, scale='k'):
        idxs_mf = idxs_mf if idxs_mf is not None else self.idxs_mf
        # NOTE not caching index-fixed meanfields, as this would require too much memory.
        if component is None:
            return np.array([self.get_kmflm(idx=idx, idxs_mf=idxs_mf, component=component).squeeze() for component in self.secondary.component])
        if isinstance(component, list):
            return np.array([self.get_kmflm(idx=idx, idxs_mf=idxs_mf, component=comp).squeeze() for comp in component])

        if self.qmflm_fn[component] is None and len(idxs_mf) <= 2: # NOTE this is really just a lower bound
            return np.zeros(shape=(1, Alm.getsize(*self.fq.lm_max_qlm)), dtype=complex)
        
        kmflm = self.get_qmflm(idx=idx, idxs=idxs_mf, component=component)

        Lmax = Alm.getlmax(kmflm.size, None)
        R = self.get_response_len(component)
        WF = self.CLfidsNoLmin[component*2][:Lmax+1] * cli(self.CLfidsNoLmin[component*2][:Lmax+1] + cli(R))  # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        kmflm = alm_copy_nd(kmflm, None, (Lmax,Lmax))
        almxfl_nd(kmflm, cli(R), Lmax, True) # Normalized QE
        almxfl_nd(kmflm, WF, Lmax, True) # Wiener-filter QE
        almxfl_nd(kmflm, self.CLfidsNoLmin[component*2][:Lmax+1] > 0, Lmax, True)
        kmflm = self._rescale(kmflm, scale='k')
        assert scale == 'k', "Only k scale is supported for kmflm at this time" # TODO can be implemented via _rescale_k2h
        return kmflm
    

    def get_wflm(self, idx, lm_max=None):
        # NOTE returns the same for each component so can just take the first key here
        return self.fq.get_wflm(idx, list(self.estimator_key.values())[0], lm_max)


    def get_ivflm(self, idx):
        # NOTE returns the same for each component so can just take the first key here
        return self.fq.get_ivflm(idx, list(self.estimator_key.values())[0])
    

    def get_response_unl(self, component, scale='p'):
        return rescale(self.fq.get_response_unl(self.estimator_key[component], self.estimator_key[component][0], self.fq.lm_max_qlm[0])[self.comp2idx[component]], scale=scale)
    

    def get_response_len(self, component, scale='p'):
        return rescale(self.fq.get_response_len(self.estimator_key[component], self.estimator_key[component][0], self.fq.lm_max_qlm[0])[self.comp2idx[component]], scale=scale)
    

    def isdone(self, idx, component):
        if self.secondary.is_cached(idx, component, 'klm'):
            return 0
        else:
            return -1

    def _get_h0_(self):
        lmax = self.fq.lm_max_qlm[0]
        ret = []
        for comp in self.secondary.component:
            scale = 'k' if self.ID in ['lensing'] else 'p' #NOTE Plancklens by default returns p scale (for lensing). Delensalot works with convergence
            R_unl0 = self.get_response_unl(comp, scale=scale)
            chh_comp = self.chh[comp]
            buff = cli(R_unl0[:lmax+1] + cli(chh_comp)) * (chh_comp > 0)
            ret.append(np.array(buff))
        return ret

    def _get_h0(self):
        lmax = self.fq.lm_max_qlm[0]
        comps = self.secondary.component
        ncomp = len(comps)
        scale = 'k' if self.ID in ['lensing'] else 'p'
        R_full = self.fq.get_response_unl(
            list(self.estimator_key.values())[0],
            list(self.estimator_key.values())[0][0],
            lmax
        )

        for i in range(ncomp):
            for j in range(ncomp):
                R_full[i, j] = rescale(R_full[i, j], scale=scale)

        C_full = np.zeros((ncomp, ncomp, lmax+1))
        for i, comp in enumerate(comps):
            C_full[i, i] = self.chh[comp][:lmax+1]

        # --- Build H0(L) = ( R + C^{-1} )^{-1}
        H0 = np.zeros_like(R_full)
        for L in range(lmax + 1):
            R_L = R_full[:, :, L]
            C_L = C_full[:, :, L]
            Cinv_L = np.zeros_like(C_L)
            for i in range(ncomp):
                if C_L[i, i] > 0:
                    Cinv_L[i, i] = 1.0 / C_L[i, i]

            M = R_L + Cinv_L
            H0[:, :, L] = np.linalg.pinv(M, rcond=1e-12)

        return H0


    # def _get_h0(self, Lc=20, eps0=0.01):
    #     """
    #     Returns H0 with optional low-L ridge regularization.
    #     Lc: transition scale (ell where ridge fades), Ridge term decays smoothly with ell^2 / (ell^2 + Lc^2)
    #     eps0: ridge amplitude (this is multiplicative factor on mean of denom at low-L, and enters linearly)
    #     """
    #     lmax = self.fq.lm_max_qlm[0]
    #     Ls = np.arange(lmax + 1)
    #     # 

    #     ret = []
    #     for comp in self.secondary.component:
    #         scale = 'k' if self.ID in ['lensing'] else 'p'
    #         R_unl0 = self.get_response_unl(comp, scale=scale)
    #         chh_comp = self.chh[comp]

    #         denom = R_unl0[:lmax+1] + cli(chh_comp)
    #         eps_L = eps0 * np.mean(denom[0:10]) * (Lc**2) / (Ls**2 + Lc**2)
    #         denom_reg = denom + eps_L  # ridge regularization at low-L
    #         buff = cli(denom_reg) * (chh_comp > 0)
    #         ret.append(np.array(buff))
    #     return ret
    

    def _rescale(self, hlm, scale):
        if scale == 'p':
            assert self.ID == 'lensing', "Only lensing is supported for p"
            return hlm
        elif scale == 'k':
            if self.ID == 'birefringence':
                return hlm
            else:
                lmax = Alm.getlmax(hlm[0].size, None)
                h2k =  0.5 * np.arange(lmax + 1) * np.arange(1, lmax + 2)
                return np.atleast_2d(almxfl(hlm[0], h2k, lmax, False))
            

    # NOTE preparation for future implementation
    def _rescale_k2h(self, klm, scale):
        if scale == 'p':
            assert self.ID == 'lensing', "Only lensing is supported for p"
            return hlm
        elif scale == 'k':
            if self.ID == 'birefringence':
                return hlm
            else:
                lmax = Alm.getlmax(hlm[0].size, None)
                h2k =  0.5 * np.arange(lmax + 1) * np.arange(1, lmax + 2)
                return np.atleast_2d(almxfl(hlm[0], h2k, lmax, False))