from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl


class secondary:
    def __init__(self, secondary_desc):
        self.ID = secondary_desc['ID'] if 'ID' in secondary_desc else None
        self.libdir = secondary_desc['libdir']
        self.CLfids  = secondary_desc['CLfids'] if 'CLfids' in secondary_desc else None
        self.lm_max = secondary_desc['lm_max'] if 'lm_max' in secondary_desc else None
        self.component = secondary_desc['component'] if 'component' in secondary_desc else None

        self.qlm_fns = {sec: f"qlm_{sec}_simidx{{idx}}" for sec in self.component}
        self.klm_fns = {sec: f"klm_{sec}_simidx{{idx}}" for sec in self.component}
        self.qmflm_fns = {sec: f"qmflm_{sec}_simidx{{idx}}" for sec in self.component}
        
        self.cacher = cachers.cacher_npy(self.libdir, verbose=False)


    def get_qlm(self, simidx, component=None):
        if component is None:
            return [self.get_qlm(simidx, component) for component in self.component]
        return self.cacher.load(self.qlm_fns[component].format(idx=simidx))
    

    def get_est(self, simidx, component=None):
        if component is None:
            return [self.get_est(simidx, component).squeeze() for component in self.component]
        return self.cacher.load(self.klm_fns[component].format(idx=simidx))


    def cache_qlm(self, klm, simidx, component=None):
        if component is None:
            assert len(klm) == len(self.component), "%d %d"%(len(klm), len(self.component))
            for ci, component in enumerate(self.component):
                self.cache_qlm(klm[ci], component)
        self.cacher.cache(self.qlm_fns[component].format(idx=simidx), klm)


    def cache_klm(self, klm, simidx, component=None):
        if component is None:
            for ci, component in enumerate(self.component):
                self.cache_klm(klm[ci], component)
        self.cacher.cache(self.klm_fns[component].format(idx=simidx), klm)


    def is_cached(self, simidx, component, type='qlm'):
        if type == 'klm':
            return self.cacher.is_cached(self.klm_fns[component].format(idx=simidx))
        else:
            return self.cacher.is_cached(self.qlm_fns[component].format(idx=simidx))


class template:
    def __init__(self, secondary_desc):
        self.ID = secondary_desc['ID'] if 'ID' in secondary_desc else None
        self.libdir = secondary_desc['libdir']
        self.lm_max = secondary_desc['lm_max'] if 'lm_max' in secondary_desc else None
        self.component = secondary_desc['component'] if 'component' in secondary_desc else None
        self.fns = {sec: f"templm_{sec}_simidx{{idx}}" for sec in self.component}
        
        self.cacher = cachers.cacher_npy(self.libdir, verbose=True)


    def get_field(self, simidx, component=None):
        if component is None:
            return [self.get_field(simidx, component).squeeze() for component in self.component]
        return self.cacher.load(self.fns[component].format(idx=simidx))


    def cache_field(self, klm, simidx, component=None):
        if component is None:
            for ci, component in enumerate(self.component):
                self.cache_field(klm[ci], component)
        self.cacher.cache(self.fns[component].format(idx=simidx), klm)


    def is_cached(self, simidx, component):
        return self.cacher.is_cached(self.fns[component].format(idx=simidx))