import numpy as np
from delensalot import remapping
from lenspyx.remapping.utils_geom import pbdGeometry

class scarf_alm_filter_wl(object):
    def __init__(self, lmax_sol:int, mmax_sol:int, ffi:remapping.deflection):
        """Base class for cmb filtering entering the iterative lensing estimators

            Args:
                lmax_sol: maximum l of unlensed CMB alm's
                mmax_sol: maximum m of unlensed CMB alm's
                ffi: delensalot spin-1 deflection field instance


        """
        self.ffi = ffi
        self.lmax_sol = lmax_sol
        self.mmax_sol = mmax_sol

        self.lmax_len = lmax_sol
        self.mmax_len = mmax_sol

    def set_ffi(self, ffi:remapping.deflection or list[remapping.deflection]):
        """Update of lensing deflection instance"""
        #TODO this should be anisotopry source object instead of deflection really
        self.ffi = ffi

    def apply_map(self, dat_map:np.ndarray):
        """Applies inverse noise operator"""
        assert 0, 'sub-class this'

    def get_qlms(self, dat_map:np.ndarray, alm_wf:np.ndarray, q_geom:pbdGeometry, alm_wf_leg2=None or np.ndarray):
        """Estimate of the quadratic likelihood piece, for data dat_map and alm_wf wiener filtered estimate"""
        assert 0, 'sub-class this'

    def get_qlms_mf(self, mfkey, q_pbgeom:pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Estimate of the quadratic likelihood piece, for data dat_map and alm_wf wiener filtered estimate"""
        assert 0, 'sub-class this'

    def dot_op(self):
        """This must give the scalar product instance betweem two cg-solution estimates"""
        assert 0, 'sub-class this'

    def synalm(self, cmbcls:dict, cmb_phas=None):
        assert 0, 'subclass this'