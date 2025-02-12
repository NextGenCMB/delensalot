import numpy as np

from delensalot.core.MAP import BFGS
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli
from delensalot.core.iterator.steps import harmonicbump

from delensalot.core import cachers

import healpy as hp

class base:
    def __init__(self, curvature_desc, gradients):
        self.gradients = gradients
        self.ID = curvature_desc['ID']
        self.field = curvature_desc['field']
        self.h0 = curvature_desc['h0']
        curvature_desc["bfgs_desc"].update({"apply_H0k": self.apply_H0k, "apply_B0k": self.apply_B0k})
        curvature_desc["bfgs_desc"].update({'cacher': cachers.cacher_npy(self.field.libdir)})
        self.BFGS_H = BFGS.BFGS_Hessian(self.h0, **curvature_desc["bfgs_desc"])
        self.stepper = harmonicbump(**{'lmax_qlm': 4000,'mmax_qlm': 4000,'a': 0.5,'b': 0.499,'xa': 400,'xb': 1500},)


    def add_svector(self, incr, simidx, it):
        self.field.cache_field(incr, 'sk', simidx, it)


    def add_yvector(self, gtot, gprev, simidx, it):
        self.field.cache_field(gtot-gprev, 'yk', simidx, it)


    def step(self, klms):
        N = 0
        for gradient in self.gradients:
            for compi, comp in enumerate(gradient.secondary.component):
                size = hp.Alm.getsize(gradient.secondary.lm_max[0])
                klms[N:N+size] = self.stepper.build_incr(klms[N:N+size], 0)
                N += size
        return klms


    def get_increment(self, gtot, simidx, it):
        if not self.field.is_cached('sk', simidx, it):
            for it_ in range(1,it):
                self.BFGS_H.add_ys(self.field.fns['yk'].format(idx=simidx, it=it_+1, itm1=it_), self.field.fns['sk'].format(idx=simidx, it=it_, itm1=it_-1), it_-1)
            gnew = self.BFGS_H.get_mHkgk(gtot, it-1)
            self.step(gnew)
            self.field.cache_field(gnew, 'sk', simidx, it)
        return self.field.get_field('sk', simidx, it)
        # gtot = gcurr, yk = gcurr - gprev


    def grad2dict(self, grad):
        N = 0
        ret = {}
        for gradient in self.gradients:
            ret.update({gradient.ID:{}})
            for component in gradient.secondary.component:
                siz = Alm.getsize(*gradient.secondary.lm_max)
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