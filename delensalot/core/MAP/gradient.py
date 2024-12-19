import numpy as np

from delensalot.core import cachers

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli

from . import filter


"""
fields:
    gradient (which is klm)
    gradient increment from curvature
    total gradient increment from previous
    2-term gradient increment from previous (no prior change)
    we can build all gradients from the increments: dont store gradient, but can cache.

    either build total or individual gradients from increments, or get new from iteration
        for building, i need the initial gradient (klm_it0), which is stored, and all increments
        for new, i need the field increments
"""

class base:
    def __init__(self, gradient_desc, filter_desc):
        self.ID = gradient_desc['ID']
        self.field = gradient_desc['field']
        self.filter = filter(filter_desc)
        self.quad_fns = gradient_desc['quad_fns']
        self.gradient_increment_fns = gradient_desc['gradient_increment_fns']
        self.gradient_increment_total_fns = gradient_desc['gradient_increment_total_fns']
        self.chh = gradient_desc['chh']
        self.inner = gradient_desc['inner']
        
        self.cacher = cachers.cacher_npy(gradient_desc['ID'])


    def get_gradient_total(self, it):
        if self.cacher.is_cached(self.gradient_increment_fns.format(idx=self.simidx, it=it)):
            return self.build_gradient_total()
        else:
            grad = self.get_gradient_prior(it)
            grad += self.get_gradient_quad(it)
            grad += self.get_gradient_meanfield(it)
            return grad


    def get_gradient_quad(self, it, component=None):
        # check if result is cached and return
        # build new quad gradient from qlm expression
        # NOTE no caching here as we build
        XWF = self.filter.get_WF(it)
        ivf = self.filter.get_ivf(it, XWF)
        
        qlms = 0
        #FIXME this is not the correct way to get the quad
        for n in [0,1,2]:
            qlms += ivf*self.inner(XWF)


    def get_gradient_meanfield(self, it, component=None):
        # NOTE this currently only uses the QE gradient meanfield
        if component is None:
            for component in self.field.components:
                return self.get_gradient_meanfield(it, component)
        else:
            # this is for existing iteration
            if self.cacher.is_cached(self.kmflm_fns[component].format(idx=self.simidx, it=0)):
                return self.cacher.load(self.kmflm_fns[component].format(idx=self.simidx, it=0))
            else:
                #this is next iteration. For now, just return previous iteration
                return self.cacher.load(self.kmflm_fns[component].format(idx=self.simidx, it=0))


    def get_gradient_prior(self, it, component=None):
        # recursive call to get all components
        if component is None:
            for component in self.field.components:
                return self.get_gradient_prior(it, component)
        else:
            # this is for existing iteration
            if self.cacher.is_cached(self.klm_fns[component].format(idx=self.simidx, it=it)):
                priorlm = self.cacher.load(self.kmflm_fns[component].format(idx=self.simidx, it=it))
                almxfl(priorlm, cli(self.chh[component]), self.field.lm_max_qlm[1], True)
                return np.array(priorlm)


    def get_WF(self):
        curr_iter = self.maxiterdone()
        return self.filter.get_WF(curr_iter)
    

    def get_ivf(self):
        curr_iter = self.maxiterdone()
        XWF = self.filter.get_WF(curr_iter)
        return self.filter.get_ivf(curr_iter, XWF)


    def update_field(self, field):
        self.filter.update_field(field)
        self.inner.update_field(field)


    def build_gradient_total(self, it):
        g = 0
        if it == 0:
            return self.field.klm_fns(idx=self.simidx, it=0)
            g += self.load_grad_prior(curr_iter)
            g += self.load_grad_det(curr_iter)
            g += self.load_grad_quad(curr_iter)
            return g
        return self._build(curr_iter)


    def load_grad_det(self, it):
        return self.cacher.load(self.det_fn.format(it=it))


    def load_grad_prior(self, it):
        ret = self.field.get_klm(it)
        almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        return ret


    def load_grad_quad(self, it, key):
        return self.cacher.load(self.quad_fns[key].format(it=it))


    def _build(self, it):
        rlm = self.field.get_klm(idx=self.simidx, it=0)
        for i in range(it):
            rlm += self.cacher.load(self.field_increment_fns(i))
        return rlm


    def update_gradient(self):
        pass

    
    def isiterdone(self, it):
        return self.cacher.is_cached(self.klm_fns.format(it=it))
    

    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr