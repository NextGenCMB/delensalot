import numpy as np

from delensalot.core.iterator import bfgs
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli


class base:
    def __init__(self, curvature_desc, gradients):
        self.gradients = gradients
        self.ID = curvature_desc['ID']
        self.field = curvature_desc['field']
        self.h0 = curvature_desc['h0']
        curvature_desc["bfgs_desc"].update({"apply_H0k": self.apply_H0k, "apply_B0k": self.apply_B0k})
        self.BFGS_H = bfgs.BFGS_Hessian(self.h0, **curvature_desc["bfgs_desc"])


    def add_yvector(self, gtot, gprev, simidx, it):
        self.field.cache_field(gtot.flatten()-gprev.flatten(), simidx, it)


    def add_svector(self, gtot, gprev, simidx, it):
        self.field.cache_field(gtot.flatten()-gprev.flatten(), simidx, it)


    def get_new_gradient(self, gtot, simidx, it):
        # TODO check if cached, otherwise calculate
        if not self.field.is_cached(simidx, it):
            gnew = self.BFGS_H.get_mHkgk(gtot.flatten(), it)
            self.field.cache_field(gnew.flatten(), simidx, it)
        return self.field.get_field(simidx, it)
        # gtot = gcurr, yk = gcurr - gprev


    def grad2dict(self, grad):
        N = 0
        ret = {}
        for gradient in self.gradients:
            ret.update({gradient.ID:{}})
            for component in gradient.field.components.split("_"):
                siz = Alm.getsize(*gradient.field.lm_max)
                ret[gradient.ID][component] = grad[N:N+siz]
                N += siz
        return ret


    def apply_H0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for h0 in self.h0:
            siz = Alm.getsize(len(h0)-1, len(h0)-1)
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], h0, len(h0), False)
            N += siz
        return ret


    def apply_B0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for h0 in self.h0:
            siz = Alm.getsize(len(h0), len(h0))
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], cli(h0), len(h0), False) #TOD0 this assumes >= 0
            N += siz
        return ret