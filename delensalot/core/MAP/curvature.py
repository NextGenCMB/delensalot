import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from delensalot.core.MAP import bfgs
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli
from delensalot.core.iterator.steps import harmonicbump

from delensalot.core import cachers
from delensalot.core.MAP import field as MAP_field

from delensalot.core.MAP.context import get_computation_context

import healpy as hp

class Base:
    def __init__(self, gradient_lib, h0, bfgs_desc, libdir):
        self.ID = "curvature"
        self.gradient_lib = gradient_lib
        self.field = MAP_field.Curvature(
            {"ID": "curvature",
            "libdir": libdir,
            "fns": {'yk': f"diff_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
                    'sk': f"incr_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
            }})
  
        self.h0 = h0
        bfgs_desc.update({"apply_H0k": self.apply_H0k, "apply_B0k": self.apply_B0k})
        bfgs_desc.update({'cacher': cachers.cacher_npy(self.field.libdir)})
        self.bfgs_h = bfgs.BFGSHessian(self.h0, **bfgs_desc)
        
        self.stepper = {sub.ID: harmonicbump(**{'lmax_qlm': sub.LM_max[0],'mmax_qlm': sub.LM_max[1],'a': 0.2,'b': 0.199,'xa': 400,'xb': 1500},) for sub in self.gradient_lib.subs}


    def add_svector(self, incr, it):
        self.field.cache(incr, it=it, type='sk')


    def add_yvector(self, gtot, gprev, it):
        self.field.cache(gtot-gprev, it=it, type='yk')


    def step(self, klms):
        N = 0
        for sub in self.gradient_lib.subs:
            for compi, comp in enumerate(sub.gfield.component):
                size = hp.Alm.getsize(sub.LM_max[0])
                klms[N:N+size] = self.stepper[sub.ID].build_incr(klms[N:N+size], 0)
                N += size
        return klms


    def get_increment(self, gtot, it):
        ctx, isnew = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        if not self.field.is_cached(it, 'sk'):
            for it_ in range(1,it):
                self.bfgs_h.add_ys(self.field.fns['yk'].format(idx=idx, idx2=idx2, it=it_+1, itm1=it_), self.field.fns['sk'].format(idx=idx, idx2=idx2, it=it_, itm1=it_-1), it_-1)
            gnew = self.bfgs_h.get_mHkgk(gtot, it-1)
            self.step(gnew)
            self.field.cache(gnew, it, 'sk')
        return self.field.get_field(it, 'sk')
        # gtot = gcurr, yk = gcurr - gprev


    def grad2dict(self, grad):
        N = 0
        ret = {}
        for subs in self.gradient_lib.subs:
            ret.update({subs.ID:{}})
            for component in subs.gfield.component:
                siz = Alm.getsize(*subs.LM_max)
                ret[subs.ID][component] = grad[N:N+siz]
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