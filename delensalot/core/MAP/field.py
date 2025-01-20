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
        self.component2idx = {component: i for i, component in enumerate(self.components.split("_"))}


    def get_klm(self, idx, it, component=None):
        "components are stored with leading dimension"
        if component is None:
            return [self.get_klm(idx, it, component).squeeze() for component in self.components.split("_")]
        if it < 0:
            return np.atleast_2d([np.zeros(Alm.getsize(*self.lm_max), dtype=complex) for component in self.components.split("_")])
        return np.atleast_2d(self.cacher.load(self.fns[component].format(idx=idx, it=it))) if self.cacher.is_cached(self.fns[component].format(idx=idx, it=it)) else np.atleast_2d(self.sk2klm(it))


    def sk2klm(self, idx, it, component):
        rlm = self.cacher.load(self.fns[component].format(idx=idx, it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm


    def cache_klm(self, klm, idx, it, component=None):
        if component is None:
            for ci, component in enumerate(self.components.split("_")):
                self.cache_klm(np.atleast_2d(klm[ci]), idx, it, component)
            return
        self.cacher.cache(self.fns[component].format(idx=idx, it=it), np.atleast_2d(klm))


    def is_cached(self, simidx, it, component):
        return self.cacher.is_cached(opj(self.fns[component].format(idx=simidx, it=it)))
    

class gradient:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.libdir_prior = field_desc['libdir_prior']
        self.lm_max = field_desc['lm_max']
        self.meanfield_fns = field_desc['meanfield_fns']
        self.prior_fns = field_desc['prior_fns']
        self.quad_fns = field_desc['quad_fns']
        self.total_fns = field_desc['total_fns']
        self.total_increment_fns = field_desc['total_increment_fns']
        self.chh = field_desc['chh']
        self.components = field_desc['components']
        self.component2idx = {component: i for i, component in enumerate(self.components.split("_"))}

        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.cacher_field = cachers.cacher_npy(opj(self.libdir_prior))


    def get_prior(self, simidx, it, component=None):
        if component is None:
            return np.atleast_2d([self.get_prior(simidx, it, component_) for component_i, component_ in enumerate(self.components.split("_"))])
        if not self.cacher_field.is_cached(self.prior_fns.format(component=component, idx=simidx, it=it)):
            assert 0, "cannot find prior at {}".format(self.cacher_field.lib_dir+"/"+self.prior_fns.format(component=component, idx=simidx, it=it))
        else:
            priorlm = self.cacher_field.load(self.prior_fns.format(component=component, idx=simidx, it=it))
            almxfl(priorlm[0], cli(self.chh[component]), self.lm_max[1], True)
        return priorlm[0]

    
    def get_meanfield(self, simidx, it, component=None):
        # NOTE this currently only uses the QE gradient meanfield
        # this is for existing iteration
        it=0
        if self.cacher.is_cached(self.meanfield_fns.format(idx=simidx, it=it)):
            return (result := self.cacher.load(self.meanfield_fns.format(idx=simidx, it=it))) if component is None else result[self.component2idx[component]]
        else:
            assert 0, "cannot find meanfield"
            

    def get_total(self, simidx, it, component):
        # if it is the 'next' it, then build the new gradient, otherwise load it
        if self.cacher.is_cached(self.total_fns.format(idx=simidx, it=it)):
            return self.cacher.load(self.total_fns.format(idx=simidx, it=it))
        elif it == 0:
            g += self.get_prior(simidx, it, component)
            g += self.get_meanfield(simidx, it, component)
            g += self.get_quad(simidx, it, component)
            return g
        return self._build(simidx, it, component)
    

    def get_quad(self, simidx, it, component):
        if not self.cacher.is_cached(self.quad_fns.format(idx=simidx, it=it)):
            return None
        else:
            return (result := self.cacher.load(self.quad_fns.format(idx=simidx, it=it))) if component is None else result[self.component2idx[component]]


    def _build(self, simidx, it, component):
        rlm = self.get_total(idx=simidx, it=0, component=component)
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
    

    def cache_prior(self, priorlm, simidx, it):
        self.cacher_field.cache(self.prior_fns.format(idx=simidx, it=it), priorlm)


    def cache_meanfield(self, kmflm, simidx, it):
        self.cacher.cache(self.meanfield_fns.format(idx=simidx, it=it), kmflm)


    def cache_quad(self, quadlm, simidx, it):
        self.cacher.cache(self.quad_fns.format(idx=simidx, it=it), quadlm)
    

class filter:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.lm_max = field_desc['lm_max']
        self.components = field_desc['components']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_field(self, simidx, it):
        return self.cacher.load(self.fns.format(idx=simidx, it=it))
    

    def cache_field(self, fieldlm, simidx, it):
        self.cacher.cache(self.fns.format(idx=simidx, it=it), fieldlm)

    
    def is_cached(self, simidx, it):
        return self.cacher.is_cached(self.fns.format(idx=simidx, it=it))
    

class curvature:
    # NOTE these are the dphi fields
    def __init__(self, field_desc):
        self.libdir = field_desc['libdir']
        # self.lm_max = field_desc['lm_max']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_field(self, simidx, it):
        return self.cacher.load(self.fns.format(idx=simidx, it=it, itm1=it-1))
    

    def cache_field(self, fieldlm, simidx, it):
        self.cacher.cache(self.fns.format(idx=simidx, it=it, itm1=it-1), fieldlm)

    
    def is_cached(self, simidx, it):
        return self.cacher.is_cached(self.fns.format(idx=simidx, it=it, itm1=it-1))