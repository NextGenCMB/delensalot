import numpy as np

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli


class base:
    def __init__(self, gradient_desc, filter, simidx):
        self.ID = gradient_desc['ID']
        self.secondary = gradient_desc['secondary']
        self.gfield = gradient_desc['gfield']
        self.gradient_operator = gradient_desc['gradient_operator']
        self.ivf_filter = filter['ivf']
        self.wf_filter = filter['wf']
        self.simidx = simidx
        self.noisemodel_coverage = gradient_desc['noisemodel_coverage']
        self.estimator_key = gradient_desc['estimator_key']
        self.simulationdata = gradient_desc['simulationdata']
        self.lm_max_ivf = gradient_desc['lm_max_ivf']
        self.lm_max_unl = gradient_desc['lm_max_unl']
        self.LM_max = gradient_desc['LM_max']
        self.ffi = gradient_desc['ffi']


    def get_gradient_total(self, it, component=None):
        if component is None:
            component = self.secondary.component
        if isinstance(it, (list,np.ndarray)):
            return np.array([self.get_gradient_total(it_, component) for it_ in it])
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=self.simidx, it=it)):
            return self.gfield.get_total(self.simidx, it, component)
        else:
            g = 0
            g += self.get_gradient_prior(it-1, component)
            g += self.get_gradient_meanfield(it, component)
            g -= self.get_gradient_quad(it, component)
            # NOTE this is implemented, but not used to save disk space
            # self.gfield.cache_total(g, self.simidx, it)
            return g


    def get_gradient_quad(self, it, component=None):
        assert 0, "subclass this"


    def get_gradient_meanfield(self, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_meanfield(it_, component) for it_ in it])
        return self.gfield.get_meanfield(self.simidx, it, component)


    def get_gradient_prior(self, it, component=None):
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(it_, component) for it_ in it])
        return self.gfield.get_gradient_prior(self.simidx, it, component)


    def update_operator(self, simidx, it):
        self.ivf_filter.update_operator(simidx, it)
        self.wf_filter.update_operator(simidx, it)
        self.gradient_operator.set_field(simidx, it)


    def update_gradient(self):
        pass


    def isiterdone(self, it):
        if it >= 0:
            return np.all(self.secondary.is_cached(self.simidx, it))
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr
    

    def get_data(self, lm_max):
        if self.noisemodel_coverage == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
            if self.k in ['pee']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)[0]
            elif self.k in ['ptt']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)
            elif self.k in ['p']:
                EBobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
                Tobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)         
                ret = np.array([Tobs, *EBobs])
                return ret
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.simidx), dtype=float)
            else:
                assert 0, 'implement if needed'



class lensing(base):

    def __init__(self, gradient_desc, filter, simidx):
        super().__init__(gradient_desc, filter, simidx)
    

    def get_gradient_quad(self, it, component=None):
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        if not self.gfield.quad_is_cached(self.simidx, it):
            data = self.get_data(self.lm_max_ivf)
            wflm = self.wf_filter.get_wflm(self.simidx, it, data)
            ivfreslm = self.ivf_filter.get_ivfreslm(self.simidx, it, data, wflm)
            
            resmap_c = np.empty((self.ffi.geom.npix(),), dtype=wflm.dtype)
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr, map=resmap_r) # ivfmap

            gcs_r = self.ffi.geom.synthesis(self.gradient_operator.act(wflm, spin=3, lmax_in=self.lm_max_unl[0], lm_max=self.lm_max_unl), 3, *self.lm_max_unl, self.ffi.sht_tr) # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)

            gcs_r = self.ffi.geom.synthesis(self.gradient_operator.act(wflm, spin=1, lmax_in=self.lm_max_unl[0], lm_max=self.lm_max_unl), 1, *self.lm_max_unl, self.ffi.sht_tr) # xwfglm
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
    

    def get_gradient_quad(self, it, component=None):
        if not self.gfield.quad_is_cached(self.simidx, it):
            data = self.get_data(self.lm_max_ivf)
            XWF = self.wf_filter.get_wflm(self.simidx, it)
            ivf = self.ivf_filter.get_ivfreslm(self.simidx, it, data, XWF)
            
            ivfmap = self.ffi.geom.synthesis(ivf, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)

            xwfglm = self.gradient_operator.act(XWF, spin=2, lmax_in=None, lm_max=None)
            xwfmap = self.ffi.geom.synthesis(xwfglm, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
 
            qlms = -4 * ( ivfmap[0] * xwfmap[1] - ivfmap[1] * xwfmap[0] )
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)