import numpy as np

from . import gradient
from . import curvature

class base:
    def __init__(self, fields, filter_desc, gradient_descs, curvature_desc, desc, simidx):
        # this class handles the filter, gradient, and curvature libs, nothing else
        self.fields = fields
        # NOTE gradient and curvature share the field increments, so naming must be consistent. Can use the gradient_descs['inc_fn'] for this
        self.gradients = [gradient(gradient_desc, filter_desc, fields) for gradient_desc in gradient_descs]
        self.curvature = curvature(curvature_desc, self.gradients)
        self.itmax = desc.get('itmax')
        self.simidx = simidx


    def estimate_fields(self):
        curr_MAPp = self.get_current_MAPpoint()
        fields = self.get_current_fields()
        self.update_operators(fields)
        gradient = self.get_gradient(curr_MAPp)
        H = self.update_curvature(gradient)
        self.update_MAP(H)


    def get_current_MAPpoint(self):
        for field in self.fields:
            comps = []
            for component in field.components:
                comps.append(field.get_klm(self.maxiterdone() - 1, component))
        self.klm_currs = np.array(comps)
        return self.klm_currs


    def get_current_fields(self):
        return self.fields
    

    def get_template(self, dlm, field):
        self.template_operator.update_field(dlm)
        return self.template_operator.act(field)
    

    def update_operators(self, fields):
        # For each operator that is dependent on a field, we need to update the field
        for gradient in self.gradients:
            gradient.update_field(fields)
        for curvature in self.curvature:
            curvature.update_field(fields)
            

    def get_gradient(self, curr_MAPp):
        gradients = []
        for gi, gradient in enumerate(self.gradients):
            gradients.append(gradient.calc_gradient(len(curr_MAPp[gi]), self.maxiterdone() - 1), self.maxiterdone() - 1) # self.klm_currs[gi]
        return np.array(gradients)


    def update_curvature(self, gradient):
        self.curvature.update_curvature(gradient) # This updates the vectors to be used for the curvature calculation


    def update_MAP(self, curvature):
        deltag = self.curvature.get_gradient_inc(self.klm_currs) # This calls the 2-loop curvature update
        for field in self.fields:
            for component in field.components:
                increment = field.calc_increment(deltag, component)
                field.update_klm(increment, component) 


    def get_curr_klm(self, it):
        # returns the current iteration klm
        if it <= 0:
            return self.cacher.is_cached(self.klm_fns.format(it=0))
        return self.hess_cacher.is_cached(self.sk_fns(it - 1))
    

    def isiterdone(self, simidx, it):
        return self.cacher.is_cached(self.klm_fns.format(simidx=simidx, it=it))
    

    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr

