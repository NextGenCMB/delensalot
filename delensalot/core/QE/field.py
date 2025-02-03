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
        self.components = secondary_desc['components'] if 'components' in secondary_desc else None

        self.qlm_fns = {sec: f"qlm_{sec}_simidx{{idx}}" for sec in self.components}
        self.klm_fns = {sec: f"klm_{sec}_simidx{{idx}}" for sec in self.components}
        self.qmflm_fns = {sec: f"qmflm_{sec}_simidx{{idx}}" for sec in self.components}
        
        self.cacher = cachers.cacher_npy(self.libdir, verbose=False)


    def get_qlm(self, simidx, components=None):
        if components is None:
            return [self.get_qlm(simidx, components) for components in self.components]
        return self.cacher.load(self.qlm_fns[components].format(idx=simidx))
    

    def get_klm(self, simidx, components=None):
        if components is None:
            return [self.get_klm(simidx, components).squeeze() for components in self.components]
        return self.cacher.load(self.klm_fns[components].format(idx=simidx))


    def cache_qlm(self, klm, simidx, components=None):
        if components is None:
            assert len(klm) == len(self.components), "%d %d"%(len(klm), len(self.components))
            for ci, components in enumerate(self.components):
                self.cache_qlm(klm[ci], components)
        self.cacher.cache(self.qlm_fns[components].format(idx=simidx), klm)


    def cache_klm(self, klm, simidx, components=None):
        if components is None:
            for ci, components in enumerate(self.components):
                self.cache_klm(klm[ci], components)
        self.cacher.cache(self.klm_fns[components].format(idx=simidx), klm)


    def is_cached(self, simidx, components, type='qlm'):
        if type == 'klm':
            return self.cacher.is_cached(self.klm_fns[components].format(idx=simidx))
        else:
            return self.cacher.is_cached(self.qlm_fns[components].format(idx=simidx))
        
class template:
    def __init__(self, secondary_desc):
        self.ID = secondary_desc['ID'] if 'ID' in secondary_desc else None
        self.libdir = secondary_desc['libdir']
        self.lm_max = secondary_desc['lm_max'] if 'lm_max' in secondary_desc else None
        self.components = secondary_desc['components'] if 'components' in secondary_desc else None
        self.fns = {sec: f"templm_{sec}_simidx{{idx}}" for sec in self.components}
        
        self.cacher = cachers.cacher_npy(self.libdir, verbose=True)


    def get_field(self, simidx, components=None):
        if components is None:
            return [self.get_field(simidx, components).squeeze() for components in self.components]
        return self.cacher.load(self.fns[components].format(idx=simidx))


    def cache_field(self, klm, simidx, components=None):
        if components is None:
            for ci, components in enumerate(self.components):
                self.cache_field(klm[ci], components)
        self.cacher.cache(self.fns[components].format(idx=simidx), klm)


    def is_cached(self, simidx, components):
        return self.cacher.is_cached(self.fns[components].format(idx=simidx))