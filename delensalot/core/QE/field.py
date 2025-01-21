from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl


class base:
    def __init__(self, field_desc):
        self.ID = field_desc['ID'] if 'ID' in field_desc else None
        self.libdir = field_desc['libdir']
        self.CLfids  = field_desc['CLfids'] if 'CLfids' in field_desc else None
        self.lm_max = field_desc['lm_max'] if 'lm_max' in field_desc else None
        self.components = field_desc['components'] if 'components' in field_desc else None
        self.qlm_fns =  field_desc['qlm_fns'] if 'qlm_fns' in field_desc else None # fns must be dict() with keys as components, and formatter for simidx
        self.klm_fns =  field_desc['klm_fns'] if 'klm_fns' in field_desc else None# fns must be dict() with keys as components, and formatter for simidx
        self.qmflm_fns = field_desc['qmflm_fns'] if 'qmflm_fns' in field_desc else None
        
        self.cacher = cachers.cacher_npy(self.libdir, verbose=True)


    def get_qlm(self, simidx, component=None):
        if component is None:
            return [self.get_qlm(simidx, component) for component in self.components.split("_")]
        return self.cacher.load(self.qlm_fns[component].format(idx=simidx))
    

    def get_klm(self, simidx, component=None):
        if component is None:
            return [self.get_klm(simidx, component).squeeze() for component in self.components.split("_")]
        return self.cacher.load(self.klm_fns[component].format(idx=simidx))


    def cache_qlm(self, klm, simidx, component=None):
        if component is None:
            assert len(klm) == len(self.components.split("_")), "%d %d"%(len(klm), len(self.components.split("_")))
            for ci, component in enumerate(self.components.split("_")):
                self.cache_qlm(klm[ci], component)
        self.cacher.cache(self.qlm_fns[component].format(idx=simidx), klm)


    def cache_klm(self, klm, simidx, component=None):
        if component is None:
            for ci, component in enumerate(self.components.split("_")):
                self.cache_klm(klm[ci], component)
        self.cacher.cache(self.klm_fns[component].format(idx=simidx), klm)


    def is_cached(self, simidx, component, type='qlm'):
        if type == 'klm':
            return self.cacher.is_cached(self.klm_fns[component].format(idx=simidx))
        else:
            return self.cacher.is_cached(self.qlm_fns[component].format(idx=simidx))