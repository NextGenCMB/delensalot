import numpy as np

from delensalot.core import cachers

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

from . import filter


"""
fields:
    gradient (which is klm)
    gradient increment from curvature
    total gradient increment from previous
    2-term gradient increment from previous (no prior change)
    we can build all gradients from the increments: dont store gradient, but can cache.

    either build total or individual gradients from increments, or get new from iteration
        for building, i need the initial gradient (klm_it0), which is stored, and all increments
        for new, i need the field increments
"""

class base:
    def __init__(self, gradient_desc, filter, simidx):
        self.ID = gradient_desc['ID']
        self.field = gradient_desc['field']
        self.gfield = gradient_desc['gfield']
        self.inner = gradient_desc['inner']
        self.filter = filter
        self.simidx = simidx
        self.noisemodel_coverage = gradient_desc['noisemodel_coverage']
        self.estimator_key = gradient_desc['estimator_key']
        self.simulationdata = gradient_desc['simulationdata']
        self.lm_max_ivf = gradient_desc['lm_max_ivf']


    def get_gradient_total(self, it, component=None):
        # if already cached, load it, otherwise calculate the new one
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=self.simidx, it=it)):
            print('total is cached at iter, ', it)
            return self.gfield.get_total(self.simidx, it, component)
        else:
            print("building total gradient for iter, ", it)
            g = 0
            g += self.get_gradient_prior(it)
            g += self.get_gradient_meanfield(it)
            g += self.get_gradient_quad(it)
            return g


    def get_gradient_quad(self, it, component=None):
        qlms = self.gfield.get_quad(self.simidx, it, component)
        data = self.get_data(self.lm_max_ivf)
        if qlms is None:
            # build new quad gradient from qlm expression
            XWF = self.filter.get_WF(data, self.simidx, it)
            ivf = self.filter.get_ivf(data, XWF, self.simidx, it)
            
            #FIXME this is not the correct way to get the quad
            # for n in [0,1,2]:
                # qlms += ivf*self.inner(XWF)
            qlms = self.gfield.get_quad(self.simidx, 0, component)
            self.gfield.cache_quad(qlms, self.simidx, it=it)
            return qlms
        return self.gfield.get_quad(self.simidx, it, component)


    def get_gradient_meanfield(self, it, component=None):
        return self.gfield.get_meanfield(self.simidx, it, component)


    def get_gradient_prior(self, it, component=None):
        return self.gfield.get_prior(self.simidx, it, component)


    def get_WF(self):
        curr_iter = self.maxiterdone()
        return self.filter.get_WF(curr_iter)
    

    def get_ivf(self):
        curr_iter = self.maxiterdone()
        XWF = self.filter.get_WF(curr_iter)
        return self.filter.get_ivf(curr_iter, XWF)


    def update_operator(self, simidx, it):
        self.filter.update_operator(simidx, it)
        self.inner.set_field(simidx, it)


    def update_gradient(self):
        pass


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