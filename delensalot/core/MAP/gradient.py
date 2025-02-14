import numpy as np

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

class base:
    def __init__(self, gradient_desc, filter, simidx):
        self.ID = gradient_desc['ID']

        self.gfield = gradient_desc['gfield']

        self.gradient_operator = gradient_desc['gradient_operator'] # NOTE this is whatever comes out of the inner for calculating the gradient wrt. the secondary
        self.ivf_filter = filter['ivf'] # NOTE this is a joint of secondary operators
        self.wf_filter = filter['wf'] # NOTE WF is ivf ivf^dagger, so could in principle be simplified

        self.simidx = simidx

        self.lm_max_sky = gradient_desc['lm_max_sky']
        self.lm_max_pri = gradient_desc['lm_max_pri']
        self.LM_max = gradient_desc['LM_max']

        self.ffi = gradient_desc['ffi']


    def get_gradient_total(self, it, component=None, data=None):
        if component is None:
            component = self.gfield.component
        if isinstance(it, (list,np.ndarray)):
            return np.array([self.get_gradient_total(it_, component) for it_ in it])
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=self.simidx, it=it)):
            return self.gfield.get_total(self.simidx, it, component)
        else:
            g = 0
            g += self.get_gradient_prior(it-1, component)
            g += self.get_gradient_meanfield(it, component)
            g -= self.get_gradient_quad(it, component, data)
            # self.gfield.cache_total(g, self.simidx, it) # NOTE this is implemented, but not used to save disk space
            return g


    def get_gradient_quad(self, it, component=None, data=None):
        assert 0, "subclass this"


    def get_gradient_meanfield(self, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_meanfield(it_, component) for it_ in it])
        return self.gfield.get_meanfield(self.simidx, it, component)


    def get_gradient_prior(self, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(it_, component) for it_ in it])
        return self.gfield.get_gradient_prior(self.simidx, it, component)


class lensing(base):

    def __init__(self, gradient_desc, filter, simidx):
        super().__init__(gradient_desc, filter, simidx)
    

    def get_gradient_quad(self, it, component=None, data=None):
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        if not self.gfield.quad_is_cached(self.simidx, it):
            assert data is not None, "data must be provided for lensing gradient calculation"
            wflm = self.wf_filter.get_wflm(self.simidx, it, data)
            ivfreslm = self.ivf_filter.get_ivfreslm(self.simidx, it, data, wflm)
            
            resmap_c = np.empty((self.ffi.geom.npix(),), dtype=wflm.dtype)
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_sky[0], self.lm_max_sky[1], self.ffi.sht_tr, map=resmap_r) # ivfmap

            gcs_r = self.ffi.geom.synthesis(self.gradient_operator.act(wflm, spin=3, lm_max_pri=self.lm_max_pri, lm_max_sky=self.lm_max_sky), 3, *self.lm_max_pri, self.ffi.sht_tr) # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)

            gcs_r = self.ffi.geom.synthesis(self.gradient_operator.act(wflm, spin=1, lm_max_pri=self.lm_max_pri, lm_max_sky=self.lm_max_sky), 1, *self.lm_max_pri, self.ffi.sht_tr) # xwfglm
            gc_c -= resmap_c * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
             
            gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
            gc = self.ffi.geom.adjoint_synthesis(gc_r, 1, self.LM_max[0], self.LM_max[0], self.ffi.sht_tr)
            # NOTE at last, cast qlms to alm space with LM_max and also cast it to convergence
            fl1 = -np.sqrt(np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl1, self.LM_max[1], True)
            almxfl(gc[1], fl1, self.LM_max[1], True)
            fl2 = cli(0.5 * np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl2, self.LM_max[1], True)
            almxfl(gc[1], fl2, self.LM_max[1], True)
            self.gfield.cache_quad(gc, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)


class birefringence(base):

    def __init__(self, gradient_desc, filter, simidx):
        super().__init__(gradient_desc, filter, simidx)
    

    def get_gradient_quad(self, it, component=None, data=None):
        if not self.gfield.quad_is_cached(self.simidx, it):
            assert data is not None, "data must be provided for lensing gradient calculation"
            XWF = self.wf_filter.get_wflm(self.simidx, it)
            ivf = self.ivf_filter.get_ivfreslm(self.simidx, it, data, XWF)
            
            ivfmap = self.ffi.geom.synthesis(ivf, 2, self.lm_max_sky[0], self.lm_max_sky[1], self.ffi.sht_tr)

            xwfglm = self.gradient_operator.act(XWF, spin=2, lm_max_pri=None, lm_max_sky=None)
            xwfmap = self.ffi.geom.synthesis(xwfglm, 2, self.lm_max_sky[0], self.lm_max_sky[1], self.ffi.sht_tr)
 
            qlms = -4 * ( ivfmap[0] * xwfmap[1] - ivfmap[1] * xwfmap[0] )
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)