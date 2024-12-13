import numpy as np

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl

class base:
    def __init__(self, **operator_desc):
        pass


    def act(self, obj, adjoint=False):
        assert 0, "subclass this"


    def update_field(self, field):
        self.field = field


class joint(base):
    def __init__(self, **operators):
        super().__init__(**operators)
        self.operators = operators
    

    def act(self, obj):
        for operator in self.operators:
            buff = operator.act(obj)
            obj = buff


class lensing(base):
    def __init__(self, **operator_desc):
        super().__init__(**operator_desc)
    

    def act(self, obj, adjoint=False):
        gc = q_pbgeom.geom.adjoint_synthesis(obj, 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(gc[0], fl, mmax_qlm, True)
        almxfl(gc[1], fl, mmax_qlm, True)


class birefringence(base):
    def __init__(self, **operator_desc):
        super().__init__(**operator_desc)
        self.field = operator_desc['field']


    def act(self, obj, adjoint=False):
        if adjoint:
            return np.exp(np.imag*self.field)*obj
        return np.exp(-np.imag*self.field)*obj


class spin_raise(base):
    def __init__(self, **operator_desc):
        super().__init__(**operator_desc)
    

    def act(self, elm_wf, adjoint=False):
        assert adjoint == False, "adjoint not implemented"

        def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:pbdGeometry):
            """Wiener-filtered gradient leg to feed into the QE
            """
            assert spin in [1, 3], spin
            lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
            i1, i2 = (2, -1) if spin == 1 else (-2, 3)
            fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
            fl[:spin] *= 0.
            fl = np.sqrt(fl)
            elm = np.atleast_2d(almxfl(elm_wf, fl, self.mmax_sol, False))
            ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
            return ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)

        gcs_r = _get_gpmap(elm_wf, 3, q_pbgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = _get_gpmap(elm_wf, 1, q_pbgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r
        lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        return gc_r
    
    def adjoint(self, obj):
        assert 0, "implement if needed"
        return self.act(obj, adjoint=True)
    

class beam(base):
    def __init__(self, **operator_desc):
        super().__init__(**operator_desc)
        self.beamwidth = operator_desc['beamwidth']
        self.mmax = operator_desc['mmax']


    def act(self, obj, adjoint=False):
        if adjoint:
            return cli(almxfl(obj, self.beam, self.mmax))
        return almxfl(obj, self.beam, self.mmax, adjoint=adjoint)   


    def adjoint(self, obj):
        return self.act(obj, adjoint=True)
    
