import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from delensalot.core.MAP import BFGS
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli
from delensalot.core.iterator.steps import harmonicbump

from delensalot.core import cachers
from delensalot.core.MAP import field as MAP_field

import healpy as hp

class base:
    def __init__(self, gradient_lib, h0, bfgs_desc, libdir):
        
        self.ID = "curvature"
        self.gradient_lib = gradient_lib
        self.field = MAP_field.curvature(
            {"ID": "curvature",
            "libdir": libdir,
            "fns": {'yk': f"diff_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
                    'sk': f"incr_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
            }})
  
        self.h0 = h0
        bfgs_desc.update({"apply_H0k": self.apply_H0k, "apply_B0k": self.apply_B0k})
        bfgs_desc.update({'cacher': cachers.cacher_npy(self.field.libdir)})
        self.BFGS_H = BFGS.BFGS_Hessian(self.h0, **bfgs_desc)
        
        self.stepper = {quad.ID: harmonicbump(**{'lmax_qlm': quad.LM_max[0],'mmax_qlm': quad.LM_max[1],'a': 0.5,'b': 0.499,'xa': 400,'xb': 1500},) for quad in self.gradient_lib.quads}


    def add_svector(self, incr, idx, it, idx2):
        self.field.cache_field(incr, 'sk', idx, it, idx2)


    def add_yvector(self, gtot, gprev, idx, it, idx2):
        self.field.cache_field(gtot-gprev, 'yk', idx, it, idx2)


    def step(self, klms):
        N = 0
        for quad in self.gradient_lib.quads:
            for compi, comp in enumerate(quad.gfield.component):
                size = hp.Alm.getsize(quad.LM_max[0])
                klms[N:N+size] = self.stepper[quad.ID].build_incr(klms[N:N+size], 0)
                N += size
        return klms


    def get_increment(self, gtot, idx, it, idx2):
        if not self.field.is_cached('sk', idx, it, idx2):
            for it_ in range(1,it):
                self.BFGS_H.add_ys(self.field.fns['yk'].format(idx=idx, idx2=idx2, it=it_+1, itm1=it_), self.field.fns['sk'].format(idx=idx, idx2=idx2, it=it_, itm1=it_-1), it_-1)
            gnew = self.BFGS_H.get_mHkgk(gtot, it-1)
            self.step(gnew)
            self.field.cache_field(gnew, 'sk', idx, it, idx2)
        return self.field.get_field('sk', idx, it, idx2)
        # gtot = gcurr, yk = gcurr - gprev


    def grad2dict(self, grad):
        N = 0
        ret = {}
        for quad in self.gradient_lib.quads:
            ret.update({quad.ID:{}})
            for component in quad.gfield.component:
                siz = Alm.getsize(*quad.LM_max)
                ret[quad.ID][component] = grad[N:N+siz]
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