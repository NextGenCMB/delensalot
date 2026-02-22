import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from os.path import join as opj

from plancklens import qest, qresp

from delensalot.core.MAP import field, gradient, curvature, functionforwardlist
from delensalot.core.MAP.context import get_computation_context, preserve_context

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, alm_copy_nd
from delensalot.config.config_manager import get_config

import healpy as hp
import matplotlib.pyplot as plt


def _resp_project(QE, src, RG, RC, RGC, RCG, Lmax):
    z = np.zeros(Lmax + 1)
    RG  = np.asarray(RG)[:Lmax+1]
    RC  = np.asarray(RC)[:Lmax+1]
    RGC = np.asarray(RGC)[:Lmax+1]
    RCG = np.asarray(RCG)[:Lmax+1]

    if QE == "p_p":
        return RG if src == "p" else z

    if QE == "x_p":
        if src == "x": return RC
        if src == "a": return RCG
        return z

    if QE == "a_p":
        if src == "a": return RG
        if src == "x": return RGC
        return z

    raise ValueError(f"Unsupported QE key {QE} in _resp_project")

class Minimizer:
    def __init__(self, likelihood, itmax, libdir, use_QE_starting_point=True, use_QE_for_lowL=False):
        self.itmax = itmax
        self.libdir = libdir

        self.likelihood: Likelihood = likelihood

        self.use_QE_starting_point = use_QE_starting_point
        # self.use_QE_starting_point = False

        self.use_QE_for_lowL = use_QE_for_lowL
        # self.use_QE_for_lowL = False

        self.secondaries: field.Secondary = {
            quad.ID: field.Secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": opj(self.libdir, 'estimate/'),
        }) for quad in likelihood.gradient_lib.subs}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = list(self.sec2idx.keys())


    def get_est(self, request_it=None, secondary=None, component=None, scale='k', calc_flag=False):
        ctx, isnew = get_computation_context()  # Get the singleton instance for MPI rank
        component, secondary = (ctx.component or component, ctx.secondary or secondary)
        current_it = self.maxiterdone()
        if not isinstance(request_it, (list, np.ndarray)):
            request_it = request_it or current_it if request_it != 0 else request_it
        if isinstance(request_it, (list, np.ndarray)):
            if any(current_it < reqit for reqit in request_it):
                print(f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}")
                return
            return self._get_est(request_it, secondary, component, scale)

        if self.maxiterdone() < 0:
            raise RuntimeError(f"Could not find the QE starting points, expected them at {self.likelihood.secondaries['lensing'].libdir}."
                "If you believe they should exist, they likely haven't been copied to the right location. Check if CopytoQEDir() has been called.")
        if request_it <= current_it:
            return self._get_est(request_it, secondary, component, scale)

        elif (current_it < self.itmax and request_it >= current_it) or calc_flag:
            new_klms = self._compute_iterations(current_it, request_it, scale)
            if isinstance(secondary, list):
                return new_klms if len(secondary)>1 else new_klms[secondary[0]] if component is None else new_klms[secondary[0]][component]
            return new_klms if secondary is None else new_klms[secondary] if component is None else new_klms[secondary][component]

        print(f"Requested iteration {request_it} is beyond the maximum iteration")
        print("If you want to calculate it, set calc_flag=True")


    # helper function
    def _compute_iterations(self, current_it, request_it, scale):
        LCMB = 1
        for it in range(current_it + 1, request_it + 1):
            log.info(f'---------- starting iteration {it} ----------')
            est_prev = self.get_est(it-1, scale='d')
            est_prev = {sec: est_prev[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
            # print("est_prev before modifying, ", est_prev)
            if not self.use_QE_starting_point and it == 1:
                print("Setting starting point to zero")
                for sec, val in est_prev.items():
                    est_prev[sec] = np.zeros_like(val, dtype=complex)
            if self.use_QE_for_lowL: # NOTE this is for isoMAP setting
                print("Using QE starting point for L<={} for lensing deflection gradient".format(LCMB))
                est_qe_qlm = {sec: self.get_est(0, scale='d')[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
                for sec, val in est_prev.items():
                    if sec == 'lensing':
                        Lmax = Alm.getlmax(val.size, None)
                        # est_prev[sec][:,:Alm.getsize(LCMB, Lmax)] = est_qe_qlm[sec][:,:Alm.getsize(LCMB, Lmax)]
                        est_prev[sec][0,:Alm.getsize(LCMB, Lmax)] = est_qe_qlm[sec][0,:Alm.getsize(LCMB, Lmax)]
            # print("est_prev after modifying, ", est_prev)
            self.update_operator(est_prev)
            grad_tot = self.likelihood.get_likelihood_gradient(it=it)
            grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
            if it >= 2:
                grad_prev = self.likelihood.get_likelihood_gradient(it=it-1)
                grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                self.likelihood.curvature_lib.add_yvector(grad_tot, grad_prev, it)
            increment = self.likelihood.curvature_lib.get_increment(grad_tot, it)

            prev_klm = np.concatenate([np.ravel(arr) for arr in self._get_est(it-1, scale=scale)])
            qe_est_klm = {sec: self.get_est(0, scale='k')[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
            if not self.use_QE_starting_point and it == 1:
                print("Setting starting point to zero")
                prev_klm = np.zeros_like(prev_klm, dtype=complex)
            new_klms = self.likelihood.curvature_lib.grad2dict(increment + prev_klm)
            if self.use_QE_for_lowL: # NOTE this is for isoMAP setting
                print("Keeping QE starting point for L<={} lensing deflection gradient".format(LCMB))
                for sec, val in new_klms.items():
                    if sec=='lensing':
                        for compi, (comp, comp_val) in enumerate(val.items()):
                            if comp == 'p':
                                Lmax = Alm.getlmax(comp_val.size, None)
                                new_klms[sec][comp][:Alm.getsize(LCMB, Lmax)] = qe_est_klm[sec][compi][:Alm.getsize(LCMB, Lmax)]

            prev_klm_ = self._get_est(it-1, scale=scale)
            for seci, (sec, val) in enumerate(new_klms.items()):
                for compi, (comp, comp_val) in enumerate(val.items()):
                    fig = plt.figure(figsize=(8,6))
                    plt.plot(hp.alm2cl(new_klms[sec][comp]), label='new klm')
                    plt.plot(hp.alm2cl(prev_klm_[seci][compi]), label='previous klm')
                    plt.loglog()
                    plt.title("sec: {}, comp: {}".format(seci, compi))
                    plt.show()
                    plt.close()

            self.cache_klm(new_klms, it)
        return new_klms

    @preserve_context
    def _get_est(self, it, secondary=None, component=None, scale='k'):
        ctx, isnew = get_computation_context()
        component, secondary = component or ctx.component, secondary or ctx.secondary
        secondary = secondary or [sec for sec in self.likelihood.secondaries.keys()]
        ctx.set(secondary=secondary, component=component)
        ret = []
        if isinstance(it, (list, np.ndarray)):
            for it_ in it:
                ret.append(self.get_secondary_est(it_, scale=scale))
            return ret
        else:
            return self.get_secondary_est(it, scale=scale)
        

    def get_secondary_est(self, it, scale='k'):
        ctx, isnew = get_computation_context()
        secondary = ctx.secondary or list(self.secondaries.keys())
        ret = []
        if isinstance(secondary, (list, np.ndarray)):
            for sec in secondary:
                # scale = 'd' if sec in ['lensing'] else 'k'
                ret.append(self.secondaries[sec].get_est(it=it, scale=scale))
        else:
            ret.append(self.secondaries[secondary].get_est(it=it, scale=scale))
        return ret
        

    @preserve_context
    def get_gradient_meanfield(self, it, idxs, secondary=None, component=None, scale='k'):
        # TODO this returns wrong result if meanfields not in MAP directory...
        ctx, isnew = get_computation_context()
        component = component or ctx.component
        secondary = secondary or ctx.secondary or list(self.likelihood.secondaries.keys())
        ctx.set(secondary=secondary, component=component)

        ret_sum = None
        count = 0
        for idx_ in idxs:
            ctx.set(idx=idx_)
            est = self.get_secondary_est(it=it, scale=scale)

            if ret_sum is None:
                ret_sum = [np.zeros_like(a, dtype=complex) for a in est]

            for i, a in enumerate(est):
                ret_sum[i] += a
            count += 1
        if count > 0:
            ret_mean = [a / count for a in ret_sum]
        else:
            ret_mean = []

        return ret_mean


    def get_template(self, it, QE_perturbative=True, secondary=None, component=None, order='reversed'):
        est = self.get_est(it, scale='d')
        secondary = secondary or self.likelihood.seclist_sorted
        nulled_secondaries = [sec for sec in self.likelihood.secondaries.keys() if sec not in secondary]
        nulled_component = [sec for sec in self.likelihood.secondaries.keys() if sec not in secondary]
        for nulled in nulled_secondaries:
            est[self.likelihood.sec2idx[nulled]] = np.zeros_like(est[self.likelihood.sec2idx[nulled]], dtype=complex)
        est = {sec: est[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
        # from delensalot.utility.plot_helper import bandpass_alms
        # est['lensing'][0] = bandpass_alms(est['lensing'][0], 20, 3000)
        self.update_operator(est)
        # TODO get_template handling should be done here, not by wfivf_filter
        return self.likelihood.gradient_lib.wfivf_filter.get_template(it, QE_perturbative=QE_perturbative, secondary=secondary, component=component, order=order)


    def isiterdone(self, it):
        if it >= 0:
            return np.all([val for sec in self.likelihood.secondaries.values() for val in sec.is_cached(it=it)])
        return False    


    def maxiterdone(self):
        it = -2
        isdone = True
        while isdone:
            it += 1
            isdone = self.isiterdone(it+1)
        return it


    # NOTE exposed functions for job handler
    def cache_klm(self, new_klms, it):
        for secID, secondary in self.likelihood.secondaries.items():
            secondary.cache_klm(new_klms[secID], it=it)


    def copyQEtoDirectory(self, QE_searchs):
        # NOTE this turns them into convergence fields
        ctx, isnew = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        config = get_config()
        for secname, secondary in self.secondaries.items():
            # QE_searchs[self.sec2idx[secname]].init_filterqest()
            if not all(self.secondaries[secname].is_cached(it=0)):
                klm_QE = QE_searchs[self.sec2idx[secname]].get_est(ctx.idx)
                self.secondaries[secname].cache_klm(klm_QE, it=0)
                print(f"finished copying secondary {secname}", ctx.idx)
            if not self.likelihood.gradient_lib.subs[self.sec2idx[secname]].gfield.is_cached(it=0, type='meanfield'):
                kmflm_QE = QE_searchs[self.sec2idx[secname]].get_kmflm(ctx.idx)
                self.likelihood.gradient_lib.subs[self.sec2idx[secname]].gfield.cache(kmflm_QE, it=0, type='meanfield')
                print(f"finished copying meanfield {secname}", ctx.idx)
            if not self.likelihood.gradient_lib.wfivf_filter.wf_field.is_cached(it=0):
                lm_max_out = config.lm_max_pri
                wflm_QE = QE_searchs[self.sec2idx[secname]].get_wflm(ctx.idx, lm_max_out)
                self.likelihood.gradient_lib.wfivf_filter.wf_field.cache(np.array(wflm_QE), it=0)
                print("finished copying wf", ctx.idx)

    
    def get_likelihood_curvature(self, it):
        grad_tot = self.likelihood.get_likelihood_gradient(it=it-1)
        grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
        print('Got grad_tot for curvature at it=', it)
        return self.likelihood.get_likelihood_curvature(grad_tot, it=it)


    def __getattr__(self, name):
        # NOTE this forwards the method call to the likelihood object
        def method_forwarder(*args, **kwargs):
            if name in functionforwardlist and hasattr(self.likelihood, name):
                return getattr(self.likelihood, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return method_forwarder


class Likelihood:
    def __init__(self, data_container, gradient_lib, libdir, QE_searchs):
        self.data = None
        self.data_container = data_container # TODO looks like this can be removed
        self.libdir = libdir
        self.QE_searchs = QE_searchs

        self.gradient_lib: gradient.Gradient = gradient_lib

        # NOTE TODO likelihood should not have a secondary object, this should be handled at minimizer level
        self.secondaries: field.Secondary = {
            quad.ID: field.Secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": opj(self.libdir, 'estimate/'),
        }) for quad in gradient_lib.subs}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = list(self.sec2idx.keys())

        # NOTE this whole thing should be wrapped into a curvature_lib class, which can depend on the curvature starting point,
        # and just passed to the likelihood
        def dotop(glms1, glms2):
            ret, N = 0., 0
            for lmax, mmax in [sub.LM_max for sec in self.seclist_sorted for sub in self.gradient_lib.subs if sub.ID == sec]:
                siz = Alm.getsize(lmax, mmax)
                cl = alm2cl(glms1[N:N+siz], glms2[N:N+siz], None, mmax, None)
                ret += np.sum(cl * (2 * np.arange(len(cl)) + 1))
                N += siz
            return ret
        curvature_desc = {"bfgs_desc": {}}
        curvature_desc["bfgs_desc"].update({'dot_op': dotop})
        curvature_desc['libdir'] = opj(self.libdir, 'curvature/')
        # curvature_desc['h0'] = [h0 for QE_search in self.QE_searchs for h0 in QE_search._get_h0()]
        curvature_desc['h0'] = self.build_full_h0(use='unl', diagonal_only=False, curvature_scale='k')
        curvature_desc['sky_coverage'] = self.gradient_lib.wfivf_filter.sky_coverage
        self.curvature_lib: curvature.Base = curvature.Base(self.gradient_lib, **curvature_desc)
        

    def get_likelihood(self, it):
        """Returns the components of -2 ln p where ln p is the approximation to the posterior"""
        assert 0, 'implement if needed'
        #FIXME: hack, this assumes this is the no-BB pol iterator 'iso' lik with no mf.  In general the needed map is the filter's file calc_prep output
        fn = 'lik_itr%04d'%it
        if not self.cacher.is_cached(fn):
            e_fname = 'wflm_%s_it%s' % ('p', it)
            assert self.wf_cacher.is_cached(e_fname), 'cant do lik, Wiener-filtered delensed CMB not available'
            elm_wf = self.wf_cacher.load(e_fname)
            self.filter.set_ffi(self._get_ffi(it))
            elm = self.opfilt.calc_prep(self.get_data(), self.cls_filt, self.filter, self.filter.ffi.sht_tr)
            l2p = 2 * np.arange(self.filter.lmax_sol + 1) + 1
            lik_qd = -np.sum(l2p * alm2cl(elm_wf, elm, self.filter.lmax_sol, self.filter.mmax_sol, self.filter.lmax_sol))
            # quadratic cst term : (X^d N^{-1} X^d)
            dat_copy = self.get_data()
            self.filter.apply_map(dat_copy)
            # This only works for 'eb iso' type filters...
            l2p = 2 * np.arange(self.filter.lmax_len + 1) + 1
            lik_qdcst  = np.sum(l2p * alm2cl(dat_copy[0], self.dat_maps[0], self.filter.lmax_len, self.filter.mmax_len, self.filter.lmax_len))
            lik_qdcst += np.sum(l2p * alm2cl(dat_copy[1], self.dat_maps[1], self.filter.lmax_len, self.filter.mmax_len, self.filter.lmax_len))
            # Prior term
            hlm = self.get_hlm(it, 'p')
            chh = alm2cl(hlm, hlm, self.lmax_qlm, self.mmax_qlm, self.lmax_qlm)
            l2p = 2 * np.arange(self.lmax_qlm + 1) + 1
            lik_pri = np.sum(l2p * chh * cli(self.chh))
            # det part
            lik_det = 0. # assumed constant here, should fix this for simple cases like constant MFs
            if True:
                self.cacher.cache(fn, np.array([lik_qdcst, lik_qd, lik_det, lik_pri]))
            return  np.array([lik_qdcst, lik_qd, lik_det, lik_pri])
        return self.cacher.load(fn)
    

    def get_likelihood_gradient(self, it):
        return self.gradient_lib.get_gradient_total(it=it)
    
    
    def get_likelihood_curvature(self, grad_tot, it):
        return self.curvature_lib.get_curvature(grad_tot, it=it)


    def build_R_full(self, *, use='unl', diagonal_only=False):
        assert use in ('unl', 'len')
        QE_searchs = self.QE_searchs
        labels = []
        for si, s in enumerate(QE_searchs):
            for comp in s.secondary.component:
                labels.append((si, comp))
        N = len(labels)
        lmax = QE_searchs[0].fq.lm_max_qlm[0]
        comp_to_src = {
            'p': 'p',  # gradient lensing
            'w': 'x',  # curl lensing
            'f': 'a',  # rotation / birefringence-like
            'x': 'x',  # if you ever use 'x' directly
            'a': 'a',
        }
        R_full = np.zeros((N, N, lmax + 1), dtype=float)
        for i, (si, comp_i) in enumerate(labels):
            search_i = QE_searchs[si]
            # target QE key (e.g. 'p_p', 'x_p', 'a_p')
            QE_key = search_i.estimator_key[comp_i]
            if use == 'unl':
                cls = search_i.fq.cls_unl
                fal = search_i.fq.ftebl_unl
            else:
                cls = search_i.fq.cls_len
                fal = search_i.fq.ftebl_len

            lmax_ivf = search_i.fq.lm_max_ivf[0]

            for j, (sj, comp_j) in enumerate(labels):
                if i != j: continue
                src = comp_to_src.get(comp_j, None)
                if src is None:
                    raise ValueError(f"Don't know how to map component '{comp_j}' to plancklens source key")

                RG, RC, RGC, RCG = qresp.get_response(
                    QE_key,
                    lmax_ivf,
                    src,          # key0
                    cls,          # cls_weight (you used cls_weight, cls_len in notebook; here keep it consistent with your wrapper usage)
                    cls,          # cls_cmb (same choice as above; you can split if you really need)
                    fal,          # filtering
                    lmax_qlm=lmax
                )

                # project to the physical response for this (QE_key <- src)
                R_full[i, j, :] = _resp_project(QE_key, src, RG, RC, RGC, RCG, lmax)

        return R_full, labels

    def build_full_h0(self, use='unl', curvature_scale="k", diagonal_only=True):
        """
        Build isotropic H0 using Eq (4.3):
            H0_L = ( 1/N0 + 1/C )^{-1}
        curvature_scale:
            "phi" → return H0 in potential units
            "k"   → return H0 in k-like units for p and w
        diagonal_only:
            True  → only diagonal Fisher block
        """

        assert use in ('unl', 'len')
        assert curvature_scale in ("phi", "k")

        QE_searchs = self.QE_searchs
        labels = [(si, comp)
                for si, s in enumerate(QE_searchs)
                for comp in s.secondary.component]

        N = len(labels)
        lmax = QE_searchs[0].fq.lm_max_qlm[0]
        L = np.arange(lmax + 1)
        f = 0.5 * L * (L + 1)

        comp_to_src = {'p': 'p', 'w': 'x', 'f': 'a', 'x': 'x', 'a': 'a'}

        F_phi = np.zeros((N, N, lmax + 1))
        for i, (si, comp_i) in enumerate(labels):
            search_i = QE_searchs[si]
            QE_key = search_i.estimator_key[comp_i]
            if use == 'unl':
                cls = search_i.fq.cls_unl
                fal = search_i.fq.ftebl_unl
            else:
                cls = search_i.fq.cls_len
                fal = search_i.fq.ftebl_len

            lmax_ivf = search_i.fq.lm_max_ivf[0]

            for j, (sj, comp_j) in enumerate(labels):
                if diagonal_only and i != j: continue
                src = comp_to_src[comp_j]
                RG, RC, RGC, RCG = qresp.get_response(QE_key, lmax_ivf, src, cls, cls, fal, lmax_qlm=lmax)
                Rphi = _resp_project(QE_key, src, RG, RC, RGC, RCG, lmax)

                # ---- Prior handling (diagonal only) ----
                if i == j:
                    C_field = search_i.chh[comp_i][:lmax+1]
                    if comp_i in ('p', 'w'):
                        # stored in k-like units → convert to potential units
                        Cpot = np.zeros_like(C_field)
                        mask = (C_field > 0) & (f > 0)
                        Cpot[mask] = C_field[mask] / (f[mask]**2)

                        invC = np.zeros_like(Cpot)
                        invC[mask] = 1.0 / Cpot[mask]
                    else:
                        invC = np.zeros_like(C_field)
                        mask = C_field > 0
                        invC[mask] = 1.0 / C_field[mask]
                else:
                    invC = 0.0
                F_phi[i, j, :] = Rphi + invC


        H_phi = np.zeros_like(F_phi)
        for ell in range(lmax + 1):
            F_L = F_phi[:, :, ell]
            H_phi[:, :, ell] = np.linalg.pinv(F_L, rcond=1e-24)

        if curvature_scale == "k":
            A = np.ones((N, lmax + 1))
            for i, (_, comp) in enumerate(labels):
                if comp in ('p', 'w'):
                    A[i, :] = f
                else:
                    A[i, :] = 1.0

            H_k = np.zeros_like(H_phi)

            for i in range(N):
                for j in range(N):
                    H_k[i, j, :] = A[i, :] * A[j, :] * H_phi[i, j, :]
            H0 = H_k
        else:
            H0 = H_phi

        def _unphysical_Lmask(comp, L):
            if comp == 'p': # dipole measurable
                return L >= 2
            if comp == 'w': # curl dipole unphysical
                return L >= 2
            if comp == 'f':
                return L >= 2
            return np.ones_like(L, dtype=bool)

        for i, (_, comp_i) in enumerate(labels):
            keep = _unphysical_Lmask(comp_i, L)
            kill = ~keep
            if np.any(kill):
                H0[i, :, kill] = 0.0
                H0[:, i, kill] = 0.0

        return H0


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if name in functionforwardlist and hasattr(self.gradient_lib, name):
                return getattr(self.gradient_lib, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder