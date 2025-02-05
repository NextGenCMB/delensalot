import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import almxfl, alm_copy, Alm
from delensalot.utils import cli

import healpy as hp

from . import gradient
from . import curvature
from . import filter

class base:
    def __init__(self, simulationdata, secondaries, filter_desc, gradient_descs, curvature_desc, desc, template_desc, simidx):
        # this class handles the filter, gradient, and curvature libs, nothing else
        self.secondaries = secondaries
        self.simulationdata = simulationdata
        # NOTE gradient and curvature share the field increments, so naming must be consistent. Can use the gradient_descs['inc_fn'] for this
        self.filter = filter.base(filter_desc, secondaries['lensing'])
        
        self.gradients = []
        for secondary_ID, secondary in secondaries.items():
            if secondary.ID == 'lensing':
                self.gradients.append(gradient.lensing(gradient_descs[secondary_ID], self.filter, simidx))
            elif secondary.ID == 'birefringence':
                self.gradients.append(gradient.birefringence(gradient_descs[secondary_ID], self.filter, simidx))
        
        # TODO i order them here as h0arr needs a specific ordering, but this is not a good solution

        h0dict = self.get_h0(desc["Runl0"])
        self.h0dict = h0dict
        h0arr = [h0dict[field.ID] for field in self.gradients]
        h0arr = [arr[i] for arr in h0arr for i in range(arr.shape[0])]
        curvature_desc.update({"h0": h0arr})
        self.curvature = curvature.base(curvature_desc, self.gradients)

        self.itmax = desc['itmax']
        self.simidx = simidx
        # TODO do later
        if False:
            self.template_cacher = cachers.cacher_npy(template_desc['libdir'])
            self.template_operators = template_desc['template_operators']
        

    # get_klm is used for certain iteration, current iteration, and final iteration 
    # can also ask for certain field and component. gradient search is always done with
    # the joint operators
    def get_klm(self, simidx, request_it, field_ID=None, component=None):
        current_it = self.maxiterdone()
        if self.maxiterdone() < 0:
            assert 0, "Could not find the QE starting points, I expected them e.g. at {}".format(self.secondaries['lensing'].libdir)
        if request_it <= current_it: # data already calculated
            if field_ID is None:
                return [self.get_klm(simidx, request_it, fieldID, component) for fieldID, field in self.secondaries.items()]
            return np.array(self.secondaries[field_ID].get_klm(simidx, request_it, component))
        elif current_it < self.itmax and request_it > current_it:
            for it in range(np.max([1,current_it]), request_it):
                # it = 0 is QE and is implicitly skipped. current_it is the it we have a solution for already
                grad_tot, grad_prev = [], []
                print('---------- starting iteration ', it, 'taking result from iteration', it-1, '. maxiterdone:', self.maxiterdone())
                for gradient in self.gradients:
                    gradient.update_operator(simidx, it-1)
                    _component = gradient.secondary.components if component is None else np.atleast_2d(component)
                    grad_tot.append(gradient.get_gradient_total(it, component=_component)) #calculates the filtering, the sum, and the quadratic combination
                grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
                if it>2: #NOTE it=1 cannot build the previous diff, as current diff is QE
                    for gradient in self.gradients:
                        _component = gradient.secondary.components if component is None else np.atleast_2d(component)
                        grad_prev.append(gradient.get_gradient_total(it-1, component=_component))
                    grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                    self.curvature.add_yvector(grad_tot, grad_prev, simidx, it)
                # NOTE it=0 uses h0 for the curvature

                # files = []
                # for k in range(np.max([0, it - self.BFGS_H.L]), it):
                #     files.append([[grad_tot, grad_prev], self.curvature.field.fns.format(idx=simidx, it=k, itm1=k-1)])
                increment = self.curvature.get_increment(grad_tot, simidx, it-1)
                new_klms = self.curvature.grad2dict(increment+self.get_klm(simidx, it-1))
                
                self.cache_klm(new_klms, simidx, it)
        else:
            assert False, "Requested iteration is beyond the maximum iteration"


    def get_template(self, field):
        fn_blt = self.template_cacher.get_fn(self.template_operators[field])
        if not self.template_cacher.is_cached(self.simidx):
            self.template_operator.update_field(self.template_operators[field])
            # almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
            blm = self.template_operator.act(field)
            self.blt_cacher.cache(fn_blt, blm)
        return self.template_cacher.load(fn_blt)
    

    def get_meanfield_it(self, it, calc=False):
        # fn = opj(self.mf_dirname, 'mf%03d_it%03d.npy'%(self.Nmf, it))
        # if not calc:
        #     if os.path.isfile(fn):
        #         mf = np.load(fn)
        #     else:
        #         mf = self.get_meanfield_it(self, it, calc=True)
        # else:
        #     plm = rec.load_plms(self.libdir_MAP(self.k, self.simidxs[0], self.version), [0])[-1]
        #     mf = np.zeros_like(plm)
        #     for simidx in self.simidxs_mf:
        #         log.info("it {:02d}: adding sim {:03d}/{}".format(it, simidx, self.Nmf-1))
        #         mf += rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), [it])[-1]
        #     np.save(fn, mf/self.Nmf)
        return None


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


    def cache_klm(self, new_klms, simidx, it):
        for fieldID, field in self.secondaries.items():
            for component in field.components:
                field.cache_klm(new_klms[fieldID][component], simidx, it, component=component)


    # exposed functions
    def get_wflm(self, field, it):
        field2idx = {grad.ID: idx for idx, grad in enumerate(self.gradients)}
        return self.gradients[field2idx[field]].get_wflm(it)

    
    def get_ivflm(self, field, it):
        field2idx = {grad.ID: idx for idx, grad in enumerate(self.gradients)}
        return self.gradients[field2idx[field]].get_ivflm(it)
        

    def get_h0(self, R_unl0):
        ret = {}
        idx2gradient = {grad.ID: idx for idx, grad in enumerate(self.gradients)}
        for field_id, field in self.secondaries.items():
            h0 = []
            for componenti, component in enumerate(field.components):
                self.ckk_prior = field.CLfids[component*2][:self.gradients[idx2gradient[field_id]].LM_max[0]+1]
                buff = cli(R_unl0[field.ID][componenti][:self.gradients[idx2gradient[field_id]].LM_max[0]+1] + cli(self.ckk_prior))   #~ (1/Cpp + 1/N0)^-1
                buff *= (self.ckk_prior > 0)
                h0.append(buff)
            ret.update({'{}'.format(field.ID): np.array(h0)}) 
        return ret