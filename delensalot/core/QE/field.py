from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl


class basefield:
    def __init__(self, **field_desc):
        self.prior = field_desc['prior']
        self.id = field_desc['ID']
        self.lm_max = field_desc['lm_max']
        self.value = field_desc['value']
        self.components = field_desc['components']
        self.fns =  field_desc['fns'] # fns must be dict() with keys as components, and formatter for simidx
        self.cacher = cachers.cacher_npy(field_desc['components'])
        self.hess_cacher = cachers.cacher_npy(opj(field_desc["lib_dir"], 'hessian'))


    def get_klm(self, it, component=None):
        if component is None:
            return [self.get_klm(it, component) for component in self.components]
        if it < 0:
            return np.zeros(Alm.getsize(*self.lm_max_qlm), dtype=complex) 
        return self.cacher.load(self.klm_fns[component].format(it=it)) if self.cacher.is_cached(self.fn[component].format(it=it)) else self.sk2klm(it)


    def sk2klm(self, it, component):
        rlm = self.cacher.load(self.klm_fns[component].format(it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm
    

    def update_klm(self, klm, id=None, component=None):
        if component is None:
            for ci, component in enumerate(self.components):
                self.update_klm(klm[ci], component)
        if id is None:
            self.cacher.save(self.klm_fns[component], klm)
        else:
            self.cacher.save(self.klm_fns[component].format(it=id), klm)


    def is_cached(self, component):
        return self.cacher.is_cached(self.klm_fns[component])