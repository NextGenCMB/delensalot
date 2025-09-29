import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from os.path import join as opj

from delensalot.core.MAP import field, gradient, curvature, functionforwardlist
from delensalot.core.MAP.context import get_computation_context

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, alm_copy_nd

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}

class Minimizer:
    def __init__(self, likelihood, itmax, libdir, use_QE_starting_point=True):
        self.itmax = itmax
        self.libdir = libdir

        self.likelihood: Likelihood = likelihood
        self.use_QE_starting_point = use_QE_starting_point

        self.secondaries: field.Secondary = {
            quad.ID: field.Secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": opj(self.libdir, 'estimate/'),
        }) for quad in likelihood.gradient_lib.subs}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))


    def get_est(self, request_it=None, secondary=None, component=None, scale='k', calc_flag=False):
        ctx, isnew = get_computation_context()  # Get the singleton instance for MPI rank
        component, secondary = (ctx.component or component, ctx.secondary or secondary)
        current_it = self.maxiterdone()

        if not isinstance(request_it, (list, np.ndarray)):
            request_it = request_it or current_it

        if isinstance(request_it, (list, np.ndarray)):
            if any(current_it < reqit for reqit in request_it):
                print(f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}")
                return
            return self._get_est(request_it, secondary, component, scale)

        if self.maxiterdone() < 0:
            raise RuntimeError(f"Could not find the QE starting points, expected them at {self.likelihood.secondaries['lensing'].libdir}")

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
        for it in range(current_it + 1, request_it + 1):
            log.info(f'---------- starting iteration {it} ----------')
            est_prev = self.get_est(it-1, scale='d')
            est_prev = {sec: est_prev[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
            if not self.use_QE_starting_point and it == 1:
                for sec, val in est_prev.items():
                    est_prev[sec] = np.zeros_like(val, dtype=complex)
            self.update_operator(est_prev)
            grad_tot = self.likelihood.get_likelihood_gradient(it=it)
            grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
            if it >= 2:
                grad_prev = self.likelihood.get_likelihood_gradient(it=it-1)
                grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                self.likelihood.curvature_lib.add_yvector(grad_tot, grad_prev, it)
            increment = self.likelihood.curvature_lib.get_increment(grad_tot, it)
            prev_klm = np.concatenate([np.ravel(arr) for arr in self._get_est(it-1, scale=scale)])
            # TODO Need to test this
            if not self.use_QE_starting_point and it == 1:
                print("STILL NEEDS TESTING: Zeroing previous klm for first iteration as not using QE starting point")
                prev_klm = np.zeros_like(prev_klm, dtype=complex)
            new_klms = self.likelihood.curvature_lib.grad2dict(increment + prev_klm)
            self.cache_klm(new_klms, it)

        return new_klms

    # helper function
    # FIXME merge this with get_secondary_est()
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
        for sec in secondary:
            # scale = 'd' if sec in ['lensing'] else 'k'
            ret.append(self.secondaries[sec].get_est(it=it, scale=scale))
        return ret
        

    # def _get_est_meanfield(self, it, secondary=None, component=None, scale='k'):
    #     ctx, isnew = get_computation_context()
    #     component, secondary = component or ctx.component, secondary or ctx.secondary
    #     secondary = secondary or [sec for sec in self.likelihood.secondaries.keys()]
    #     ctx.set(secondary=secondary, component=component)
    #     ret = []
    #     if isinstance(it, (list, np.ndarray)):
    #         for it_ in it:
    #             ret.append(self.likelihood.get_est_meanfield(it_, scale=scale))
    #         return ret
    #     else:
    #         return self.likelihood.get_est_meanfield(it, scale=scale)


    def get_template(self, it, secondary=None, component=None):
        est = self.get_est(it, scale='d')
        secondary = secondary or self.likelihood.seclist_sorted
        nulled_secondaries = [sec for sec in self.likelihood.secondaries.keys() if sec not in secondary]
        nulled_component = [sec for sec in self.likelihood.secondaries.keys() if sec not in secondary]
        for nulled in nulled_secondaries:
            est[self.likelihood.sec2idx[nulled]] = np.zeros_like(est[self.likelihood.sec2idx[nulled]], dtype=complex)
        est = {sec: est[self.likelihood.sec2idx[sec]] for sec in self.likelihood.seclist_sorted}
        self.update_operator(est)
        return self.likelihood.gradient_lib.wfivf_filter.get_template(it, secondary=secondary, component=component)


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
        for secname, secondary in self.secondaries.items():
            QE_searchs[self.sec2idx[secname]].init_filterqest()
            if not all(self.secondaries[secname].is_cached(it=0)):
                klm_QE = QE_searchs[self.sec2idx[secname]].get_est(ctx.idx)
                self.secondaries[secname].cache_klm(klm_QE, it=0)
            if not self.likelihood.gradient_lib.subs[self.sec2idx[secname]].gfield.is_cached(it=0, type='meanfield'):
                kmflm_QE = QE_searchs[self.sec2idx[secname]].get_kmflm(ctx.idx)
                self.likelihood.gradient_lib.subs[self.sec2idx[secname]].gfield.cache(kmflm_QE, it=0, type='meanfield')
            if not self.likelihood.gradient_lib.wfivf_filter.wf_field.is_cached(it=0):
                lm_max_out = self.likelihood.gradient_lib.subs[0].gradient_operator.operators[-1].operators[-1].lm_max_out
                wflm_QE = QE_searchs[self.sec2idx[secname]].get_wflm(ctx.idx, lm_max_out)
                self.likelihood.gradient_lib.wfivf_filter.wf_field.cache(np.array(wflm_QE), it=0)


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
        self.data_container = data_container
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
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))

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
        curvature_desc['h0'] = [h0 for QE_search in self.QE_searchs for h0 in QE_search._get_h0()]
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
    
    
    def get_likelihood_curvature(self, it):
        return self.curvature_lib.get_curvature(it=it)
    

    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if name in functionforwardlist and hasattr(self.gradient_lib, name):
                return getattr(self.gradient_lib, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder