import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import almxfl, alm_copy, Alm
from delensalot.utils import cli

import healpy as hp

from . import gradient
from . import curvature
from . import filter

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}

class base:
    def __init__(self, simulationdata, secondaries, filter_desc, gradient_descs, curvature_desc, desc, template_desc, simidx):
        
        # this class handles the filter, gradient, and curvature libs, nothing else
        self.secondaries = secondaries
        self.simulationdata = simulationdata
        self.filter = filter.base(filter_desc, secondaries['lensing'])
        self.sec2idx = {secondary_ID: idx for idx, (secondary_ID, secondary) in enumerate(secondaries.items())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(secondaries.keys())}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))
        self.gradients = []
        self.chh = {}
        for sec in self.seclist_sorted:
            self.chh.update({sec: gradient_descs[sec]['gfield'].chh})
            if sec == 'lensing':
                self.gradients.append(gradient.lensing(gradient_descs[sec], self.filter, simidx))
            elif sec == 'birefringence':
                self.gradients.append(gradient.birefringence(gradient_descs[sec], self.filter, simidx))

        h0arr = np.array([v for val in self.get_h0(desc["Runl0"]).values() for v in val.values()])
        curvature_desc.update({"h0": h0arr})
        self.curvature = curvature.base(curvature_desc, self.gradients)

        self.itmax = desc['itmax']
        self.simidx = simidx
        # TODO do later
        if False:
            self.template_cacher = cachers.cacher_npy(template_desc['libdir'])
            self.template_operators = template_desc['template_operators']

    
    def get_est(self, simidx, request_it, secondary=None, component=None, scale='k', calc_flag=False):
        current_it = self.maxiterdone()
        if isinstance(request_it, (list,np.ndarray)):
            assert all([current_it>=reqit for reqit in request_it]), f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}"
            # assert not calc_flag and any([current_it<reqit for reqit in request_it]), "Cannot calculate new iterations if it is a list, please set calc_flag=False"
            return [self.get_est(simidx, it, secondary, component, scale=scale, calc_flag=False) for it in request_it]
        if self.maxiterdone() < 0:
            assert 0, "Could not find the QE starting points, I expected them e.g. at {}".format(self.secondaries['lensing'].libdir)
        if request_it <= current_it: # data already calculated
            if secondary is None:
                return [self.secondaries[secondary].get_est(simidx, request_it, component, scale=scale) for secondary in self.secondaries.keys()]
            elif isinstance(secondary, list):
                return [self.secondaries[sec].get_est(simidx, request_it, component, scale=scale) for sec in secondary]
            else:
                return self.secondaries[secondary].get_est(simidx, request_it, component, scale=scale)
        elif (current_it < self.itmax and request_it >= current_it) or calc_flag:
            for it in range(current_it+1, request_it+1):
                # NOTE it = 0 is QE and is implicitly skipped. current_it is the it we have a solution for already
                grad_tot, grad_prev = [], []
                print(f'---------- starting iteration {it} ----------')
                for gradient in self.gradients:
                    gradient.update_operator(simidx, it-1)
                    grad_tot.append(gradient.get_gradient_total(it)) #calculates the filtering, the sum, and the quadratic combination
                grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
                if it>=2: #NOTE it=1 cannot build the previous diff, as current diff is QE
                    for gradient in self.gradients:
                        grad_prev.append(gradient.get_gradient_total(it-1))
                    grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                    self.curvature.add_yvector(grad_tot, grad_prev, simidx, it)
                increment = self.curvature.get_increment(grad_tot, simidx, it)
                prev_klm = np.concatenate([np.ravel(arr) for arr in self.get_est(simidx, it-1, scale=scale)])
                new_klms = self.curvature.grad2dict(increment+prev_klm)
                
                self.cache_klm(new_klms, simidx, it)
                # return self.load_klm(simidx, it, secondary, component)
                # FIXME return a list of requested secondaries and components, not dict
            return new_klms if secondary is None else new_klms[secondary] if component is None else new_klms[secondary][component]
        elif current_it < self.itmax and request_it >= current_it and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')
        elif request_it > self.itmax and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')
        

    def get_template(self, field):
        fn_blt = self.template_cacher.get_fn(self.template_operators[field])
        if not self.template_cacher.is_cached(self.simidx):
            self.template_operator.update_field(self.template_operators[field])
            blm = self.template_operator.act(field)
            self.blt_cacher.cache(fn_blt, blm)
        return self.template_cacher.load(fn_blt)
    

    def isiterdone(self, it):
        if it >= 0:
            return np.all([val for sec in self.secondaries.values() for val in sec.is_cached(self.simidx, it)])
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr


    def get_h0(self, R_unl0):
        ret = {}
        for grad in self.gradients:
            ret.update({grad.ID: {}})
            lmax = grad.secondary.lm_max[0]
            for compi, comp in enumerate(grad.secondary.component):
                ckp = self.chh[grad.ID][comp]
                R_unl = R_unl0[grad.ID][comp][:lmax+1]*cli(self.__p2k(lmax=lmax))**2

                buff = cli(R_unl + cli(ckp)) * (ckp > 0)
                ret[grad.ID].update({comp: np.array(buff)})
        return ret


    def __p2k(self, lmax):
        return 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)


    # exposed functions for convenience
    def get_wflm(self, secondary, it):
        return self.gradients[self.sec2idx[secondary]].get_wflm(it)

    
    def get_ivfreslm(self, secondary, it):
        return self.gradients[self.sec2idx[secondary]].get_ivfreslm(it)
    

    def get_gradient_quad(self, it=None, secondary=None, component=None):
        if it > self.maxiterdone():
            print(f"Requested iteration {it} is beyond the maximum iteration")
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_quad(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_quad(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_quad(it, component) for idx in sec_idx])


    def get_gradient_meanfield(self, it=None, secondary=None, component=None):
        if it > self.maxiterdone():
            print(f"Requested iteration {it} is beyond the maximum iteration")
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_meanfield(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_meanfield(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_meanfield(it, component) for idx in sec_idx])
    

    def get_gradient_prior(self, it=None, secondary=None, component=None):
        if it > self.maxiterdone():
            print(f"Requested iteration {it} is beyond the maximum iteration")
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_prior(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_prior(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_prior(it, component) for idx in sec_idx])
    

    def get_gradient_total(self, it=None, secondary=None, component=None):
        if it > self.maxiterdone():
            print(f"Requested iteration {it} is beyond the maximum iteration")
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_total(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_total(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_total(it, component) for idx in sec_idx])


    # exposed functions for job handler
    def cache_klm(self, new_klms, simidx, it):
        for fieldID, field in self.secondaries.items():
            for component in field.component:
                field.cache_klm(new_klms[fieldID][component], simidx, it, component=component)

    def load_klm(self, simidx, it, secondary, component):
        # FIXME update
        for fieldID, field in self.secondaries.items():
            for component in field.component:
                field.get_est(simidx, it, component=component)