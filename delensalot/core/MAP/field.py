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
        self.klm_fns =  field_desc['klm_fns'] # klm_fns must be dict() with keys as components
        self.cacher = cachers.cacher_npy(field_desc['components'])
        self.hess_cacher = cachers.cacher_npy(opj(field_desc["lib_dir"], 'hessian'))


    def get_klm(self, it, component):
        if it < 0:
            return np.zeros(Alm.getsize(*self.lm_max_qlm), dtype=complex) 
        return self.cacher.load(self.klm_fns[component].format(it=it)) if self.cacher.is_cached(self.klm_fns[component].format(it=it)) else self.sk2klm(it)


    def sk2klm(self, it, component):
        rlm = self.cacher.load(self.klm_fns[component].format(it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm