import numpy as np

from . import field
from . import gradient
from . import curvature

class base:
    def __init__(self, fields_desc, gradient_descs, curvature_desc, **kwargs):
        self.fields = field(fields_desc)
        # NOTE gradient and curvature share the field increments, so naming must be consistent. Can use the gradient_descs['inc_fn'] for this
        self.gradients = [gradient(gradient_desc, field) for gradient_desc, field in zip(gradient_descs, self.fields)]
        self.curvature = curvature(curvature_desc, self.gradients)


    def run(self):
        self.get_current_MAP()
        self.get_gradient()
        self.get_curvature()
        self.update_MAP()


    def get_current_MAP(self, simidx):
        it_curr = self.get_curr_iter(simidx)

        self.klm_currs = []
        for field in self.fields:
            for component in field.components:
                self.klm_currs.append(field.get_klm(it_curr - 1, component))


    def get_gradient(self):
        for gi, gradient in enumerate(self.gradients):
            gradient.calc_gradient(self.klm_currs[gi])


    def get_curvature(self):
        self.curvature.update_curvature() # This updates the vectors to be used for the curvature calculation


    def update_MAP(self):
        deltag = self.curvature.get_gradient_inc(self.klm_currs) # This calls the 2-loop curvature update
        for field in self.fields:
            for component in field.components:
                increment = field.calc_increment(deltag, component)
                field.update_klm(increment, component) 
        


    def get_curr_iter(self, it):
        """Returns True if the iteration 'it' has been performed already and False if not
        """
        if it <= 0:
            return self.cacher.is_cached(self.klm_fns.format(it=0))
        return self.hess_cacher.is_cached(self.sk_fns(it - 1))

