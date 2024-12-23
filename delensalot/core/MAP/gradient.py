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
    def __init__(self, gradient_desc, filter_desc, simidx):
        self.ID = gradient_desc['ID']
        self.field = gradient_desc['field']
        self.gfield = gradient_desc['gfield']

        self.filter = filter.base(filter_desc)
        self.inner = gradient_desc['inner']

        self.simidx = simidx


    def get_gradient_total(self, it, component=None):
        # if already cached, load it, otherwise calculate the new one
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=self.simidx, it=it)):
            return self.gfield.get_total(self.simidx, it, component)
        else:
            g = 0
            g += self.get_gradient_prior(it)
            g += self.get_gradient_meanfield(it)
            g += self.get_gradient_quad(it)
            return g


    def get_gradient_quad(self, it, component=None):
        # check if result is cached and return
        # build new quad gradient from qlm expression
        XWF = self.filter.get_WF(it)
        ivf = self.filter.get_ivf(it, XWF)
        
        qlms = 0
        #FIXME this is not the correct way to get the quad
        for n in [0,1,2]:
            qlms += ivf*self.inner(XWF)


    def get_gradient_meanfield(self, it, component=None):
        return self.gfield.get_meanfield(self.simidx, it, component)


    def get_gradient_prior(self, it, component=None):
        return self.gfield.get_prior(self.simidx, it, component)
    

    def get_gradient_quad(self, it, component=None):
        return self.gfield.get_quad(self.simidx, it, component)


    def get_WF(self):
        curr_iter = self.maxiterdone()
        return self.filter.get_WF(curr_iter)
    

    def get_ivf(self):
        curr_iter = self.maxiterdone()
        XWF = self.filter.get_WF(curr_iter)
        return self.filter.get_ivf(curr_iter, XWF)


    def update_operator(self, simidx, it):
        self.filter.update_operator(simidx, it)
        self.inner.set_field(simidx, it)


    def update_gradient(self):
        pass