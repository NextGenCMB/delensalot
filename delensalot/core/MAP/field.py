from os.path import join as opj
import numpy as np

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl

from delensalot.core import cachers

class base:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.lm_max = field_desc['lm_max']
        self.CLfids = field_desc['CLfids']
        self.components = field_desc['components']
        self.fns =  field_desc['fns'] # fns must be dict() with keys as components, and it as format specifiers
        self.meanfield_fns = field_desc['meanfield_fns']
        self.increment_fns = field_desc['increment_fns']
        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_klm(self, idx, it, component=None):
        if component is None:
            return [self.get_klm(idx, it, component) for component in self.components]
        if it < 0:
            return np.zeros(Alm.getsize(*self.lm_max), dtype=complex) 
        return self.cacher.load(self.fns[component].format(idx=idx, it=it)) if self.cacher.is_cached(self.fns[component].format(idx=idx, it=it)) else self.sk2klm(it)


    def sk2klm(self, idx, it, component):
        rlm = self.cacher.load(self.fns[component].format(idx=idx, it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm


    def cache_klm(self, klm, idx, it=None, component=None):
        if component is None:
            for ci, component in enumerate(self.components):
                self.cache_klm(klm[ci], component)
            self.cacher.save(self.fns[component].format(idx=idx, it=it), klm)


    def is_cached(self, component):
        return self.cacher.is_cached(self.fns[component])
    

class gradient:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.lm_max = field_desc['lm_max']
        self.meanfield_fns = field_desc['meanfield_fns']
        self.prior_fns = field_desc['prior_fns']
        self.quad_fns = field_desc['quad_fns']
        self.total_fns = field_desc['total_fns']
        self.total_increment_fns = field_desc['total_increment_fns']
        self.chh = field_desc['chh']
        self.components = field_desc['components']

        self.cacher = cachers.cacher_npy(opj(self.libdir, 'gradient'))


    def get_prior(self, simidx, it, component):
        # recursive call to get all components
        if component is None:
            for component in self.components:
                return self.get_prior(simidx, it, component)
        else:
            # this is for existing iteration
            if self.cacher.is_cached(self.prior_fns.format(idx=simidx, it=it)):
                priorlm = self.cacher.load(self.prior_fns[component].format(idx=simidx, it=it))
                almxfl(priorlm, cli(self.chh[component]), self.lm_max[1], True)
                return np.array(priorlm)
        return self.cacher.load(self.prior_fns.format(it=it, idx=simidx))
    

    def get_meanfield(self, simidx, it, component):
        # NOTE this currently only uses the QE gradient meanfield
        if component is None:
            for component in self.components:
                return self.get_meanfield(it, component)
        else:
            # this is for existing iteration
            if self.cacher.is_cached(self.meanfield_fns[component].format(idx=simidx, it=0)):
                return self.cacher.load(self.meanfield_fns[component].format(idx=simidx, it=0))
            else:
                #this is next iteration. For now, just return previous iteration
                return self.cacher.load(self.meanfield_fns[component].format(idx=simidx, it=0))
            

    def get_total(self, simidx, it, component):
        # if it is the 'next' it, then build the new gradient, otherwise load it
        if self.cacher.is_cached(self.total_fns.format(idx=simidx, it=it)):
            return self.cacher.load(self.total_fns.format(idx=simidx, it=it))
        elif it == 0:
            g += self.get_prior(it)
            g += self.get_meanfield(it)
            g += self.get_quad(it)
            return g
        return self._build(simidx, it)
    

    def get_quad(self, it, component):
        return self.quad_fns[component].format(it=it)
    

    def _build(self, simidx, it):
        rlm = self.get_total(idx=simidx, it=0)
        for i in range(it):
            rlm += self.cacher.load(self.total_increment_fns(i))
        return rlm
    

    def isiterdone(self, it):
        return self.cacher.is_cached(self.klm_fns.format(it=it))
    

    def maxiterdone(self):
        it = -2
        isdone = True
        while isdone:
            it += 1
            isdone = self.isiterdone(it + 1)
        return it
    
class filter:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.lm_max = field_desc['lm_max']
        self.components = field_desc['components']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir, 'filter'))