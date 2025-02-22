import numpy as np
from os.path import join as opj

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli
from delensalot.core.cg import cd_solve

from . import field
from . import gradient
from . import curvature
from . import filter
from . import operator

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}


class minimizer:
    def __init__(self, estimator_key, likelihood, itmax, libdir, idx, idx2):
        # NOTE this is the minimizer

        self.estimator_key = estimator_key
        self.itmax = itmax
        self.libdir = libdir
        self.idx = idx
        self.idx2 = idx2 or idx

        self.likelihood = likelihood

    
    def get_est(self, request_it=None, secondary=None, component=None, scale='k', calc_flag=False):
        self.likelihood._copyQEtoDirectory(self.likelihood.QE_searchs)
        current_it = self.maxiterdone()
        request_it = request_it or current_it
        if isinstance(request_it, (list,np.ndarray)):
            if all([current_it<reqit for reqit in request_it]): print(f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}")
            # assert not calc_flag and any([current_it<reqit for reqit in request_it]), "Cannot calculate new iterations if it is a list, please set calc_flag=False"
            return [self.get_est(it, secondary, component, scale=scale, calc_flag=False) for it in request_it[request_it<=current_it]]
        if self.maxiterdone() < 0:
            assert 0, "Could not find the QE starting points, I expected them e.g. at {}".format(self.likelihood.secondaries['lensing'].libdir)
        if request_it <= current_it: # data already calculated
            if secondary is None:
                return [self.likelihood.secondaries[secondary].get_est(self.idx, request_it, component, scale=scale) for secondary in self.likelihood.secondaries.keys()]
            elif isinstance(secondary, list):
                return [self.likelihood.secondaries[sec].get_est(self.idx, request_it, component, scale=scale) for sec in secondary]
            else:
                return self.likelihood.secondaries[secondary].get_est(self.idx, request_it, component, scale=scale)
        elif (current_it < self.itmax and request_it >= current_it) or calc_flag:
            for it in range(current_it+1, request_it+1): # NOTE it=0 is QE and is implicitly skipped. current_it is the it we have a solution for already
                print(f'---------- starting iteration {it} ----------')
                grad_tot, grad_prev = [], []
                self.likelihood.update_operator(it-1)
                grad_tot = self.likelihood.gradient_lib.get_gradient_total(idx=self.idx, idx2=self.idx2, it=it, component=component)
                grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
                print(f'grad tot', grad_tot)
                if it>=2: #NOTE it=1 cannot build the previous diff, as current diff is QE
                    grad_prev = self.likelihood.gradient_lib.get_gradient_total(idx=self.idx, idx2=self.idx2, it=it-1, component=component)
                    grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                    self.likelihood.curvature.add_yvector(grad_tot, grad_prev, self.idx, it, idx2=self.idx2)
                increment = self.likelihood.curvature.get_increment(grad_tot, self.idx, it, idx2=self.idx2)
                prev_klm = np.concatenate([np.ravel(arr) for arr in self.get_est(it-1, scale=scale)])
                new_klms = self.likelihood.curvature.grad2dict(increment+prev_klm)
                self.cache_klm(new_klms, it)
            # TODO return a list of requested secondaries and components, not dict
            # return self.load_klm(it, secondary, component)
            return new_klms if secondary is None else new_klms[secondary] if component is None else new_klms[secondary][component]
        elif current_it < self.itmax and request_it >= current_it and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')
        elif request_it > self.itmax and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')


    def isiterdone(self, it):
        if it >= 0:
            return np.all([val for sec in self.likelihood.secondaries.values() for val in sec.is_cached(idx=self.idx, idx2=self.idx2, it=it)])
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr


    def get_wflm(self, it):
        return self.wf_filter.get_wflm(it)

    
    def get_ivfreslm(self, it):
        return self.ivf_filter.get_ivfreslm(it)
    

    # FIXME update to work with new gradient library
    def get_gradient_quad(self, it=None, secondary=None, component=None, data=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_quad(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_quad(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return [self.gradients[idx].get_gradient_quad(it, component, data) for idx in sec_idx]

    def get_gradient_meanfield(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_meanfield(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_meanfield(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_meanfield(it, component) for idx in sec_idx])
    
    def get_gradient_prior(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_prior(it-1, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_prior(it-1, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_prior(it-1, component) for idx in sec_idx])
    
    def get_gradient_total(self, it=None, secondary=None, component=None, data=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_total(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_total(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_total(it, component, data) for idx in sec_idx])


    def get_template(self, it, secondary=None, component=None):
        return self.wf_filter.get_template(it, secondary, component)


    # exposed functions for job handler
    def cache_klm(self, new_klms, it):
        for secID, secondary in self.likelihood.secondaries.items():
            for component in secondary.component:
                secondary.cache_klm(new_klms[secID][component], idx=self.idx, idx2=self.idx2, it=it, component=component)


class likelihood:
    def __init__(self, data_container, gradient_lib, libdir, QE_searchs, lm_max_sky, estimator_key, idx, idx2=None):
        # NOTE this is the minimizer
        self.data = None 

        self.estimator_key = estimator_key
        self.data_container = data_container
        self.libdir = libdir
        self.QE_searchs = QE_searchs
        self.idx = idx
        self.idx2 = idx2 or idx
        self.lm_max_sky = lm_max_sky

        self.secondaries = {
            quad.ID: field.secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": opj(self.libdir, 'estimate/'),
        }) for quad in gradient_lib.quads}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))

        self.gradient_lib = gradient_lib

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
        self.curvature = curvature.base(self.gradient_lib, **curvature_desc)
        

    def get_likelihood(self, it):
        """Returns the components of -2 ln p where ln p is the approximation to the posterior"""
        #FIXME: hack, this assumes this is the no-BB pol iterator 'iso' lik with no mf.  In general the needed map is the filter's file calc_prep output
        assert 0, 'implement if needed'
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
    

    def get_gradient_total(self, it=None, secondary=None, component=None, data=None):
        self.gradient_lib.get_gradient_total(idx=self.idx, idx2=self.idx2, it=it, component=component)
    

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
    

    def update_operator(self, it):
        # NOTE updaing a single operator here is enough to update all operators,
        # as they all point to the same operator.lensing and birefringence
        self.gradient_lib.update_operator(it)
        # self.wf_filter.update_operator(it)
        # self.gradients[0].update_operator(it)


    def get_wflm(self, it):
        return self.wf_filter.get_wflm(it)

    
    def get_ivfreslm(self, it):
        return self.ivf_filter.get_ivfreslm(it)
    

    # FIXME update to work with new gradient library
    def get_gradient_quad(self, it=None, secondary=None, component=None, data=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_quad(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_quad(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return [self.gradients[idx].get_gradient_quad(it, component, data) for idx in sec_idx]

    def get_gradient_meanfield(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_meanfield(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_meanfield(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_meanfield(it, component) for idx in sec_idx])
    
    def get_gradient_prior(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_prior(it-1, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_prior(it-1, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_prior(it-1, component) for idx in sec_idx])
    
    def get_gradient_total(self, it=None, secondary=None, component=None, data=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_total(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_total(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_total(it, component, data) for idx in sec_idx])


    def get_template(self, it, secondary=None, component=None):
        return self.wf_filter.get_template(it, secondary, component)


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
    def _copyQEtoDirectory(self, QE_searchs):
        # NOTE this turns them into convergence fields
        for secname, secondary in self.secondaries.items():
            QE_searchs[self.sec2idx[secname]].init_filterqest()
            if not all(self.secondaries[secname].is_cached(self.idx, it=0)):
                klm_QE = QE_searchs[self.sec2idx[secname]].get_est(self.idx)
                self.secondaries[secname].cache_klm(klm_QE, self.idx, it=0)
            
            if not self.gradient_lib.quads[self.sec2idx[secname]].gfield.is_cached(self.idx, it=0):
                kmflm_QE = QE_searchs[self.sec2idx[secname]].get_kmflm(self.idx)
                self.gradient_lib.quads[self.sec2idx[secname]].gfield.cache_meanfield(kmflm_QE, self.idx, it=0)

            #TODO cache QE wflm into the filter directory
            if not self.gradient_lib.wf_filter.wf_field.is_cached(self.idx, it=0):
                wflm_QE = QE_searchs[self.sec2idx[secname]].get_wflm(self.idx, self.lm_max_pri)
                self.gradient_lib.wf_filter.wf_field.cache_field(np.array(wflm_QE), self.idx, it=0)
