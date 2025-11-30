import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from delensalot.core import cachers
from delensalot.core.MAP import field as MAP_field, bfgs
from delensalot.core.MAP.context import get_computation_context

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl

class nrstep(object):
    def __init__(self, lmax_qlm:int, mmax_qlm:int, val=1.):
        self.lmax_qlm = lmax_qlm
        self.mmax_qlm = mmax_qlm
        self.val = val

    def steplen(self, itr, incrnorm):
        return self.val

    def build_incr(self, incrlm, itr):
        print('incr step val %.5f'%self.val)
        return incrlm * self.val

class harmonicbump(nrstep):
    def __init__(self, lmax_qlm, mmax_qlm, xa=400, xb=1500, a=0.5, b=0.499, scale=50, flt=None):
        """Harmonic bumpy step that were useful for s06b and s08b

        """
        super().__init__(lmax_qlm, mmax_qlm)
        filt = np.ones(self.lmax_qlm + 1, dtype=float)
        if flt is not None:
            filt[:min(len(flt), lmax_qlm+1)] = flt[:min(len(flt), lmax_qlm+1)]
        self.scale = scale
        self.bump_params = (xa, xb, a, b)
        self.filt = filt

    def steplen(self, itr, incrnorm):
        xa, xb, a, b = self.bump_params
        return self.bp(np.arange(self.lmax_qlm + 1),xa, a, xb, b, scale=self.scale)


    def build_incr(self, incrlm, itr):
        fl = self.steplen(itr, incrlm)
        almxfl(incrlm, fl * self.filt, self.mmax_qlm, True)
        return incrlm

    @staticmethod
    def bp(x, xa, a, xb, b, scale=50):
            """Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale

            """
            x0 = (xa + xb) * 0.5
            r = lambda x_: np.arctan(np.sign(b - a) * (x_ - x0) / scale) + np.sign(b - a) * np.pi * 0.5
            return a + r(x) * (b - a) / r(xb)
            
class Base:
    def __init__(self, gradient_lib, h0, bfgs_desc, libdir, sky_coverage):
        self.ID = "curvature"
        self.gradient_lib = gradient_lib
        self.field = MAP_field.Curvature(
            {"ID": "curvature",
            "libdir": libdir,
            "fns": {'yk': f"diff_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
                    'sk': f"incr_grad1d_simidx{{idx}}_{{idx2}}_it{{it}}m{{itm1}}",
            }})
        # setting_fullsky = lambda sub: {'lmax_qlm': sub.LM_max[0], 'mmax_qlm': sub.LM_max[1], 'a': 0.2, 'b': 0.199, 'xa': 400, 'xb': 1500}
        setting_fullsky = lambda sub: {'lmax_qlm': sub.LM_max[0], 'mmax_qlm': sub.LM_max[1], 'a': 0.5, 'b': 0.499, 'xa': 400, 'xb': 1500} # NOTE main branch setting
        setting_masked = lambda sub: {'lmax_qlm': sub.LM_max[0], 'mmax_qlm': sub.LM_max[1], 'a': 0.02, 'b': 0.399,'xa': 1, 'xb': 15}

        self.h0 = h0
        bfgs_desc.update({"applyH0k": self.applyH0k, "applyB0k": self.applyB0k})
        bfgs_desc.update({'cacher': cachers.cacher_npy(self.field.libdir)})
        subs_layout = []
        for sub in self.gradient_lib.subs:
            for compi, comp in enumerate(sub.gfield.component):
                subs_layout.append(sub.LM_max)
        bfgs_desc.update({'subs_layout': subs_layout})
        self.bfgs_h = bfgs.BFGSHessian(self.h0, **bfgs_desc)
        
        setting_hb = setting_masked if sky_coverage == "masked" else setting_fullsky
        # setting_hb = setting_fullsky # NOTE need to manually set this for now
        self.stepper = {sub.ID: harmonicbump(**setting_hb(sub),) for sub in self.gradient_lib.subs}

        self.iprior_list = np.diagonal(self.gradient_lib.ipriormatrix).T
        self.dot_op = bfgs_desc.get('dot_op', np.sum)


    def add_svector(self, incr, it):
        self.field.cache(incr, it=it, type='sk')


    def add_yvector(self, gtot, gprev, it):
        self.field.cache(gtot-gprev, it=it, type='yk')


    def step(self, klms):
        N = 0
        for sub in self.gradient_lib.subs:
            for compi, comp in enumerate(sub.gfield.component):
                size = Alm.getsize(*sub.LM_max)
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


    def applyH0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for h0 in self.h0:
            siz = Alm.getsize(len(h0)-1, len(h0)-1)
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], h0, len(h0), False)
            N += siz
        return ret


    def applyB0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for h0 in self.h0:
            siz = Alm.getsize(len(h0)-1, len(h0)-1)
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], cli(h0), len(h0), False) #TOD0 this assumes >= 0
            N += siz
        return ret

    
    def get_curvature(self, grad_tot, it, secondary=None, component=None, idx2=None):
        ctx, isnew = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        for it_ in range(1,it):
            self.bfgs_h.add_ys(self.field.fns['yk'].format(idx=idx, idx2=idx2, it=it_+1, itm1=it_), self.field.fns['sk'].format(idx=idx, idx2=idx2, it=it_, itm1=it_-1), it_-1)
        return self.bfgs_h.get_curvature_spectra(grad_tot, it)


