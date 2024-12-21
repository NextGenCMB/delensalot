from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl


class base:
    def __init__(self, **field_desc):
        self.CLfids  = field_desc['CLfids']
        self.id = field_desc['ID']
        self.lm_max = field_desc['lm_max']
        self.components = field_desc['components']
        self.qlm_fns =  field_desc['qlm_fns'] # fns must be dict() with keys as components, and formatter for simidx
        self.klm_fns =  field_desc['klm_fns'] # fns must be dict() with keys as components, and formatter for simidx
        self.qlmmf_fns = field_desc['qlmmf_fns']
        self.cacher = cachers.cacher_npy(field_desc['components'])


    def get_qlm(self, it, component=None):
        if component is None:
            return [self.get_qlm(it, component) for component in self.components]
        if it < 0:
            return np.zeros(Alm.getsize(*self.lm_max), dtype=complex) 
        return self.cacher.load(self.qlm_fns[component]) if self.cacher.is_cached(self.qlm_fns[component]) else None
    

    def get_klm(self, it, component=None):
        if component is None:
            return [self.get_klm(it, component) for component in self.components]
        if it < 0:
            return np.zeros(Alm.getsize(*self.lm_max), dtype=complex) 
        return self.cacher.load(self.klm_fns[component]) if self.cacher.is_cached(self.klm_fns[component]) else None


    def update_qlm(self, klm, component=None):
        if component is None:
            for ci, component in enumerate(self.components):
                self.update_qlm(klm[ci], component)
        self.cacher.save(self.qlm_fns[component], klm)


    def update_klm(self, klm, component=None):
        if component is None:
            for ci, component in enumerate(self.components):
                self.update_klm(klm[ci], component)
        self.cacher.save(self.klm_fns[component], klm)


    def is_cached(self, component):
        return self.cacher.is_cached(self.qlm_fns[component])