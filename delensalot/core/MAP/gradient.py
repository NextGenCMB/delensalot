import numpy as np

from delensalot.core import cachers

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli

from . import filter


class base:
    def __init__(self, gradient_desc, filter_desc, field):
        self.field = field
        self.filter = filter(filter_desc)
        self.meanfield_fns = gradient_desc['meanfield_fns']
        self.pri_fns = gradient_desc['prior_fns']
        self.quad_fns = gradient_desc['quad_fns']
        self.increment_fns = gradient_desc['increment_fns']
        self.chh = gradient_desc['chh']
        self.inner = gradient_desc['inner']
        self.ID = gradient_desc['ID']

        self.cacher = cachers.cacher_npy(gradient_desc['ID'])


    def calc_gradient(self, ncomponents, curr_iter):
        self.calc_gradient_prior(ncomponents, curr_iter)
        self.calc_gradient_quad(ncomponents, curr_iter)
        self.calc_gradient_meanfield(ncomponents, curr_iter)


    def calc_gradient_quad(self, curr_iter):
        XWF = self.filter.get_XWF(curr_iter)
        ivf = self.filter.get_ivf(curr_iter, XWF)
        
        qlms = 0
        #FIXME this is not the correct way to get the quad
        for n in [0,1,2]:
            qlms += ivf*self.inner(XWF)


    def calc_gradient_meanfield(self, curr_iter):
        return self.cacher.load(self.meanfield_fns.format(it=curr_iter))


    def calc_gradient_prior(self, curr_iter):
        for field in self.fields:
            for component in field.components:
                ret = self.field.get_klm(curr_iter, component)
        almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        return ret


    def update_field(self, field):
        self.filter.update_field(field)
        self.inner.update_field(field)


    def load_gradient(self, curr_iter):
        """Loads the total gradient at iteration iter.
        All necessary alm's must have been calculated previously
        Compared to formalism of the papers, this returns -g_LM^{tot}
        """
        if curr_iter == 0:
            for component in self.field.components:
                g  = self.load_grad_prior(curr_iter)
                g += self.load_grad_det(curr_iter)
                g += self.load_grad_quad(curr_iter)
            return g
        return self._build(curr_iter)


    def load_det(self, it):
        return self.cacher.load(self.det_fn.format(it=it))


    def load_prior(self, it):
        """Compared to formalism of the papers, this returns -g_LM^{PR}"""
        ret = self.field.get_klm(it)
        almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        return ret


    def load_quad(self, it, key):
        return self.cacher.load(self.quad_fns[key].format(it=it))


    def _build(self, it):
        rlm = self.load_gradient(0)
        for i in range(it):
            rlm += self.hess_cacher.load(self.increment_fns(i))
        return rlm


    def update_gradient(self):
        pass