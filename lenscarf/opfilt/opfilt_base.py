import numpy as np
from lenscarf import remapping
from lenscarf.utils_scarf import pbdGeometry

class scarf_alm_filter_wl(object):
    def __init__(self, lmax_sol:int, mmax_sol:int, ffi:remapping.deflection):
        """Base class for cmb filtering entering the iterative lensing estimators


        """
        self.ffi = ffi
        self.lmax_sol = lmax_sol
        self.mmax_sol = mmax_sol

    def set_ffi(self, ffi:remapping.deflection):
        self.ffi = ffi

    def get_qlms(self, dat_map:np.ndarray, alm_wf:np.ndarray, q_geom:pbdGeometry):
        assert 0, 'implement this'

    def dot_op(self):
        assert 0, 'implement this'