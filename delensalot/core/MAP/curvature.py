import numpy as np

from delensalot.core.MAP import BFGS
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli
from delensalot.core.iterator.steps import harmonicbump

from delensalot.core import cachers
from delensalot.core.MAP import field as MAP_field

import healpy as hp

class base:
    def __init__(self, gradients, h0, bfgs_desc, libdir):
        self.ID = "curvature"
        self.gradients = gradients
        self.field = MAP_field.curvature(
            {"ID": "curvature",
            "libdir": libdir,
            "fns": {'yk': f"diff_grad1d_simidx{{idx}}_it{{it}}m{{itm1}}",
                    'sk': f"incr_grad1d_simidx{{idx}}_it{{it}}m{{itm1}}",
            }})
  
        self.h0 = h0
        bfgs_desc.update({"apply_H0k": self.apply_H0k, "apply_B0k": self.apply_B0k})
        bfgs_desc.update({'cacher': cachers.cacher_npy(self.field.libdir)})
        self.BFGS_H = BFGS.BFGS_Hessian(self.h0, **bfgs_desc)
        
        self.stepper = {grad.ID: harmonicbump(**{'lmax_qlm': grad.LM_max[0],'mmax_qlm': grad.LM_max[0],'a': 0.5,'b': 0.499,'xa': 400,'xb': 1500},) for grad in self.gradients}


    def add_svector(self, incr, simidx, it):
        self.field.cache_field(incr, 'sk', simidx, it)


    def add_yvector(self, gtot, gprev, simidx, it):
        self.field.cache_field(gtot-gprev, 'yk', simidx, it)


    def step(self, klms):
        N = 0
        for gradient in self.gradients:
            for compi, comp in enumerate(gradient.gfield.component):
                size = hp.Alm.getsize(gradient.LM_max[0])
                klms[N:N+size] = self.stepper[gradient.ID].build_incr(klms[N:N+size], 0)
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
            for component in gradient.gfield.component:
                siz = Alm.getsize(*gradient.LM_max)
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