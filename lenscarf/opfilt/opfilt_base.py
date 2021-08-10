import numpy as np
from lenscarf import remapping
from lenscarf.utils_scarf import pbdGeometry

class scarf_alm_filter_wl(object):
    def __init__(self, lmax_sol:int, mmax_sol:int, ffi:remapping.deflection):
        """Base class for cmb filtering entering the iterative lensing estimators

            Args:
                lmax_sol: maximum l of unlensed CMB alm's
                mmax_sol: maximum m of unlensed CMB alm's
                ffi: lenscarf spin-1 deflection field instance


        """
        self.ffi = ffi
        self.lmax_sol = lmax_sol
        self.mmax_sol = mmax_sol

        self.lmax_len = lmax_sol
        self.mmax_len = mmax_sol

    def set_ffi(self, ffi:remapping.deflection):
        """Update of lensing deflection instance"""
        self.ffi = ffi

    def get_qlms(self, dat_map:np.ndarray, alm_wf:np.ndarray, q_geom:pbdGeometry, alm_wf_leg2=None or np.ndarray):
        """Estimate of the quadratic likelihood piece, for data dat_map and alm_wf wiener filtered estimate"""
        assert 0, 'sub-class this'

    def dot_op(self):
        """This must give the scalar product instance betweem two cg-solution estimates"""
        assert 0, 'sub-class this'

    def synalm(self, cmbcls:dict):
        assert 0, 'subclass this'