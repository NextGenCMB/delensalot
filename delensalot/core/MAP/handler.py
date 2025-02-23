import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from os.path import join as opj

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli


from . import field
from . import gradient
from . import curvature
from . import filter
from . import operator

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}

from delensalot.core.MAP.context import ComputationContext
from delensalot.core.MAP.context import get_computation_context
class Minimizer:
    def __init__(self, estimator_key, likelihood, itmax, libdir, idx, idx2):
        
        # NOTE this is the minimizer
        self.estimator_key = estimator_key
        self.itmax = itmax
        self.libdir = libdir
        self.idx = idx
        self.idx2 = idx2 or idx

        self.likelihood: Likelihood = likelihood
        self.ctx, isnew = get_computation_context()
        self.ctx.set(idx=idx, idx2=idx2)



    def get_est(self, request_it=None, secondary=None, component=None, scale='k', calc_flag=False, idx=None, idx2=None):
        ctx, isnew = get_computation_context()  # Get the singleton instance for MPI rank
        request_it, idx, idx2, component, secondary = (
            ctx.it or request_it, ctx.idx or idx, ctx.idx2 or idx2, ctx.component or component, ctx.secondary or secondary
        )
        self.likelihood.copyQEtoDirectory(self.likelihood.QE_searchs)
        current_it = self.maxiterdone()
        ctx.set(it=request_it)

        # Handle request_it as a list or single value
        if isinstance(request_it, (list, np.ndarray)):
            request_it = request_it if np.any(request_it) else current_it
        else:
            request_it = request_it or current_it

        # If request_it is a list, return results for all valid iterations
        if isinstance(request_it, (list, np.ndarray)):
            if all(current_it < reqit for reqit in request_it):
                print(f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}")
                return  # Exit early
            return [self._get_est_single(it_, secondary, component, scale) for it_ in request_it if it_ <= current_it]

        # If no previous iterations exist, error out
        if self.maxiterdone() < 0:
            raise RuntimeError(
                f"Could not find the QE starting points, expected them at {self.likelihood.secondaries['lensing'].libdir}"
            )

        # Case 1: Requested iteration already computed
        if request_it <= current_it:
            return self._get_est_single(request_it, secondary, component, scale)

        # Case 2: New iterations need to be computed
        elif (current_it < self.itmax and request_it >= current_it) or calc_flag:
            new_klms = self._compute_iterations(current_it, request_it, idx, idx2, secondary, component, scale)
            return new_klms if secondary is None else new_klms[secondary] if component is None else new_klms[secondary][component]

        # Case 3: Requested iteration beyond allowed max
        print(f"Requested iteration {request_it} is beyond the maximum iteration")
        print("If you want to calculate it, set calc_flag=True")


    # Helper function to compute individual iteration
    def _compute_iterations(self, current_it, request_it, idx, idx2, secondary, component, scale):
        ctx = get_computation_context()[0]
        idx, it, idx2 = ctx.idx, ctx.it, ctx.idx2 or ctx.idx
        for it in range(current_it + 1, request_it + 1):  # Iterations 1+ are calculated
            ctx.set(idx=idx, idx2=idx2, it=it)
            log.info(f'---------- starting iteration {it} ----------')

            self.update_operator(idx, it-1, idx2=idx2)
            grad_tot = self.get_gradient_total()
            grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])

            if it >= 2:
                self.ctx.set(it=it-1, secondary=secondary, component=component)
                grad_prev = self.get_gradient_total()
                grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                self.likelihood.curvature.add_yvector(grad_tot, grad_prev, it)

            increment = self.likelihood.curvature.get_increment(grad_tot, it)
            prev_klm = np.concatenate([np.ravel(arr) for arr in self.get_est(it - 1, scale=scale)])
            new_klms = self.likelihood.curvature.grad2dict(increment + prev_klm)
            self.cache_klm(new_klms, it)

        return new_klms


    # Helper function to retrieve or compute a single iteration result
    def _get_est_single(self, it, secondary, component, scale):
        """Retrieve stored estimates for a single iteration."""
        if secondary is None:
            return [self.likelihood.secondaries[sec].get_est(scale=scale) for sec in self.likelihood.secondaries.keys()]
        elif isinstance(secondary, list):
            return [self.likelihood.secondaries[sec].get_est(scale=scale) for sec in secondary]
        else:
            return self.likelihood.secondaries[secondary].get_est(scale=scale)




    def isiterdone(self, it):
        ctx, isnew = get_computation_context()
        if it >= 0:
            ctx.set(it=it)
            return np.all([val for sec in self.likelihood.secondaries.values() for val in sec.is_cached()])
        return False    


    def maxiterdone(self):
        ctx, isnew = get_computation_context()
        it = ctx.it

        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        ctx.set(it=it)
        return itr


    # exposed functions for job handler
    def cache_klm(self, new_klms, it):
        for secID, secondary in self.likelihood.secondaries.items():
            secondary.cache_klm(new_klms[secID])


    def __getattr__(self, name):
        # NOTE this forwards the method call to the likelihood object
        def method_forwarder(*args, **kwargs):
            if hasattr(self.likelihood, name):
                return getattr(self.likelihood, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return method_forwarder


class Likelihood:
    def __init__(self, data_container, gradient_lib, libdir, QE_searchs, lm_max_sky, estimator_key, idx, idx2=None):
        
        self.data = None 

        self.estimator_key = estimator_key
        self.data_container = data_container
        self.libdir = libdir
        self.QE_searchs = QE_searchs
        self.idx = idx
        self.idx2 = idx2 or idx
        self.lm_max_sky = lm_max_sky

        self.secondaries: field.secondary = {
            quad.ID: field.secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": opj(self.libdir, 'estimate/'),
        }) for quad in gradient_lib.quads}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))

        self.gradient_lib: gradient.Joint  = gradient_lib

        def dotop(glms1, glms2):
            ret, N = 0., 0
            for lmax, mmax in [quad.LM_max for sec in self.seclist_sorted for quad in self.gradient_lib.quads if quad.ID == sec]:
                siz = Alm.getsize(lmax, mmax)
                cl = alm2cl(glms1[N:N+siz], glms2[N:N+siz], None, mmax, None)
                ret += np.sum(cl * (2 * np.arange(len(cl)) + 1))
                N += siz
            return ret
        curvature_desc = {"bfgs_desc": {}}
        curvature_desc["bfgs_desc"].update({'dot_op': dotop})
        curvature_desc['libdir'] = opj(self.libdir, 'curvature/')
        curvature_desc['h0'] = [h0 for QE_search in self.QE_searchs for h0 in QE_search._get_h0()]
        self.curvature: curvature.base = curvature.base(self.gradient_lib, **curvature_desc)
        

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
    

    def isiterdone(self, it):
        if it >= 0:
            return np.all([val for sec in self.secondaries.values() for val in sec.is_cached(idx=self.idx, idx2=self.idx2, it=it)])
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr
    

    # exposed functions for job handler
    def cache_klm(self, new_klms, it):
        for secID, secondary in self.secondaries.items():
            for component in secondary.component:
                secondary.cache_klm(new_klms[secID][component], idx=self.idx, idx2=self.idx2, it=it, component=component)


    def get_data(self):
        # TODO this could be provided by the data_container directly

        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            # NOTE dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if len(self.estimator_key.split('_'))==1:
                if len(self.estimator_key) == 3:
                    data_key = self.estimator_key[1:]
                elif len(self.estimator_key) == 1:
                    data_key = self.estimator_key
            else:
                data_key = self.estimator_key.split('_')[-1]
            if data_key in ['p', 'eb', 'be']:
                return alm_copy(
                    self.data_container.get_sim_obs(self.idx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_sky)
            if data_key in ['ee']:
                return alm_copy(
                    self.data_container.get_sim_obs(self.idx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_sky)[0]
            elif data_key in ['tt']:
                return alm_copy(
                    self.data_container.get_sim_obs(self.idx, space='alm', spin=0, field='temperature'),
                    None, *self.lm_max_sky)
            elif data_key in ['p']:
                EBobs = alm_copy(
                    self.data_container.get_sim_obs(self.idx, space='alm', spin=0, field='polarization'),
                    None, *self.lm_max_sky)
                Tobs = alm_copy(
                    self.data_container.get_sim_obs(self.idx, space='alm', spin=0, field='temperature'),
                    None, *self.lm_max_sky)         
                ret = np.array([Tobs, *EBobs])
                return ret
            else:
                assert 0, 'implement if needed'
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.idx), dtype=float)
            else:
                assert 0, 'implement if needed'


    # NOTE This can be called from application level. Once the starting points are calculated, this can be used to prepare the MAP run
    def copyQEtoDirectory(self, QE_searchs):
        # NOTE this turns them into convergence fields
        ctx, isnew = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        for secname, secondary in self.secondaries.items():
            QE_searchs[self.sec2idx[secname]].init_filterqest()
            ctx.set(it=0)
            if not all(self.secondaries[secname].is_cached()):
                klm_QE = QE_searchs[self.sec2idx[secname]].get_est(self.idx)
                self.secondaries[secname].cache_klm(klm_QE)
            
            if not self.gradient_lib.quads[self.sec2idx[secname]].gfield.is_cached():
                kmflm_QE = QE_searchs[self.sec2idx[secname]].get_kmflm(self.idx)
                self.gradient_lib.quads[self.sec2idx[secname]].gfield.cache_meanfield(kmflm_QE)

            #TODO cache QE wflm into the filter directory
            if not self.gradient_lib.wf_filter.wf_field.is_cached():
                lm_max_out = self.gradient_lib.quads[0].gradient_operator.operators[-1].operators[0].lm_max_out
                wflm_QE = QE_searchs[self.sec2idx[secname]].get_wflm(self.idx, lm_max_out)
                self.gradient_lib.wf_filter.wf_field.cache(np.array(wflm_QE))


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if hasattr(self.gradient_lib, name):
                return getattr(self.gradient_lib, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder