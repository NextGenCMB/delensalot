import numpy as np
from os.path import join as opj

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

from delensalot.core.MAP import field
from delensalot.core.MAP import operator

class base:
    def __init__(self, gradient_desc):
        self.ID = gradient_desc['ID']
        libdir = gradient_desc['libdir']

        self.lm_max_pri = gradient_desc['lm_max_pri']
        self.lm_max_sky = gradient_desc['lm_max_sky']

        self.ffi = gradient_desc['ffi']
        self.chh = gradient_desc['chh']
        self.component = gradient_desc['component']

        self.ivf_filter = gradient_desc['ivf_filter']
        self.wf_filter = gradient_desc['wf_filter']

        gfield_desc = {
            "ID": self.ID,
            "libdir": opj(libdir, 'gradients'),
            "libdir_prior": opj(libdir, 'estimate'),
            "meanfield_fns": f'mf_glm_{self.ID}_simidx{{idx}}_it{{it}}',
            "quad_fns": f'quad_glm_{self.ID}_simidx{{idx}}_it{{it}}',
            "prior_fns": 'klm_{component}_simidx{idx}_it{it}', # prior is just field, and then we do a simple divide by spectrum (almxfl)
            "total_increment_fns": f'ginclm_{self.ID}_simidx{{idx}}_it{{it}}',    
            "total_fns": f'gtotlm_{self.ID}_simidx{{idx}}_it{{it}}',    
            "chh": self.chh,
            "component": self.component,
        }
        self.gfield = field.gradient(gfield_desc)


    def get_gradient_total(self, simidx, it, component=None):
        if component is None:
            component = self.gfield.component
        if isinstance(it, (list,np.ndarray)):
            return np.array([self.get_gradient_total(it_, component) for it_ in it])
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=simidx, it=it)):
            return self.gfield.get_total(simidx, it, self.LM_max, component)
        else:
            g = 0
            g += self.get_gradient_prior(simidx, it-1, component)
            g += self.get_gradient_meanfield(simidx, it, component)
            g -= self.get_gradient_quad(simidx, it, component)
            # self.gfield.cache_total(g, simidx, it) # NOTE this is implemented, but not used to save disk space
            return g


    def get_gradient_quad(self, simidx, it, component=None):
        assert 0, "subclass this"


    def get_gradient_meanfield(self, simidx, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_meanfield(it_, component) for it_ in it])
        return self.gfield.get_meanfield(simidx, it, component=component)


    def get_gradient_prior(self, simidx, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(it_, component) for it_ in it])
        return self.gfield.get_gradient_prior(simidx, it, component=component)


class lensing(base):

    def __init__(self, gradient_desc):
        super().__init__(gradient_desc)
        self.gradient_operator = self.get_operator(gradient_desc['sec_operator'])


    def get_gradient_quad(self, simidx, it, component=None):
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        if not self.gfield.quad_is_cached(simidx, it):
            wflm = self.wf_filter.get_wflm(simidx, it, self.data)
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(simidx, it, self.data, wflm))

            resmap_c = np.ascontiguousarray(np.empty((self.ffi.geom.npix(),), dtype=wflm.dtype))
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.ffi.geom.synthesis(ivfreslm, 2, *self.lm_max_sky, self.ffi.sht_tr, map=resmap_r) # ivfmap
            
            gcs_r = self.gradient_operator.act(wflm, spin=3, out='map') # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)

            gcs_r = self.gradient_operator.act(wflm, spin=1, out='map') # xwfglm
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
            self.gfield.cache_quad(gc, simidx, it=it)
        return self.gfield.get_quad(simidx, it, component=component)
    

    def get_operator(self, filter_operator):
        return operator.joint([operator.spin_raise({'lm_max': self.lm_max_pri}), filter_operator])


class birefringence(base):

    def __init__(self, gradient_desc):
        super().__init__(gradient_desc)
        self.gradient_operator = self.get_operator(gradient_desc['sec_operator'])
    

    def get_gradient_quad(self, simidx, it, component=None):
        if not self.gfield.quad_is_cached(simidx, it):
            wflm = self.wf_filter.get_wflm(simidx, it, self.data)
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(simidx, it, self.data, wflm))
            
            ivfmap = self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_sky[0], self.lm_max_sky[1], self.ffi.sht_tr)

            xwfglm = self.gradient_operator.act(wflm, spin=2) # FIXME are the lmaxes correct?
            xwfmap = self.ffi.geom.synthesis(xwfglm, 2, *self.lm_max_pri, self.ffi.sht_tr)
 
            qlms = -4 * (ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms, simidx, it=it)
        return self.gfield.get_quad(simidx, it, component=component)
    

    def get_operator(self, filter_operator):
        return operator.joint([operator.multiply({"factor": -1j}), filter_operator])