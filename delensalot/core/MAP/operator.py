import numpy as np

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.core import cachers
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl

class base:
    def __init__(self):
        self.field_cacher = cachers.cacher_mem()


    def act(self, obj, adjoint=False):
        assert 0, "subclass this"


    def update_field(self, field, fn):
        self.field_cacher.cache(fn, field)


class multiply:
    def __init__(self, factor):
        self.factor = factor
    

    def act(self, obj, adjoint=False):
        if adjoint:
            return np.adj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj):
        return self.act(obj, adjoint=True)


class joint:
    def __init__(self, operators):
        self.operators = operators
    

    def act(self, obj, adjoint=False):
        for operator in self.operators:
            buff = operator.act(obj)
            obj = buff
        return obj
    

    def adjoint(self, obj):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj)
            obj = buff
        return obj
    

    def update_fields(self, fields, fns=''):
        # FIXME each operator has its own fields to update
        for operator in self.operators:
            operator.update_field(fields, fns)


class ivf_operator:
    def __init__(self, operators):
        self.operators = operators
        self.lm_max = operators[0].lm_max if hasattr(operators[0], 'lm_max') else operators[1].lm_max # FIXME hoping these operators have what im looking for
    

    def act(self, obj):
        for operator in self.operators:
            buff = operator.act(obj)
            obj = buff
        return obj
    

    def adjoint(self, obj):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj)
            obj = buff
        return obj
    

    def update_fields(self, fields, fns=''):
        # FIXME each operator has its own fields to update
        for operator in self.operators:
            operator.update_field(fields, fns)
    

class WF_operator:
    def __init__(self, operators):
        self.operators = operators
        self.lm_max = operators[0].lm_max if hasattr(operators[0], 'lm_max') else operators[1].lm_max # FIXME hoping these operators have what im looking for
    

    def act(self, obj):
        buff = None
        for operator in self.operators[::-1]:
            buff = operator.act(obj)
            obj = buff
        for operator in self.operators:
            buff = operator.adjoint(obj)
            obj = buff
        return obj
    

    def adjoint(self, obj):
        buff = None
        for operator in self.operators:
            buff = operator.adjoint.act(obj)
            obj = buff
        for operator in self.operators[::-1]:
            buff = operator.act(obj)
            obj = buff
        return obj
    

    def update_fields(self, fields, fns=''):
        # FIXME each operator has its own fields to update
        for operator in self.operators:
            operator.update_field(fields, fns)


class lensing(base):
    def __init__(self, operator_desc):
        super().__init__()
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"],
        self.perturbative = operator_desc["perturbative"],
        self.field_fns = operator_desc["field_fns"],


    def act(self, obj, lm_max_qlm, adjoint=False):
        assert adjoint == False, "adjoint not implemented"
        if self.perturbative: # Applies perturbative remapping
            pass
            # get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            # geom, sht_tr = self.fq.ffi.geom, self.fq.ffi.sht_tr
            # d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
            # dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
            # del dp, dm, d1
            # elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
        else:  
            # ffi = self.fq.ffi.change_dlm([dlm, None], self.mmax_qlm)
            # elm, blm = ffi.lensgclm(np.array([elm_wf, np.zeros_like(elm_wf)]), self.mmax_filt, 2, lmaxb, mmaxb)
            gc = self.q_pbgeom.geom.adjoint_synthesis(obj, 1, lm_max_qlm[0], lm_max_qlm[1], self.ffi.sht_tr)
            fl = -np.sqrt(np.arange(lm_max_qlm[0] + 1, dtype=float) * np.arange(1, lm_max_qlm[0] + 2))
            almxfl(gc[0], fl, lm_max_qlm[1], True)
            almxfl(gc[1], fl, lm_max_qlm[1], True)
        return gc
    

    def adjoint(self, obj):
        return self.act(obj, adjoint=True)


class birefringence(base):
    def __init__(self, operator_desc):
        super().__init__()
        self.field_fns = operator_desc['field_fns']
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"],


    def act(self, obj, adjoint=False):
        if adjoint:
            return np.exp(np.imag*self.field)*obj
        return np.exp(-np.imag*self.field)*obj
    
    
    def adjoint(self, obj):
        return self.act(obj, adjoint=True)


class spin_raise:
    def __init__(self, operator_desc):
        pass


    def act(self, elm_wf, adjoint=False):
        assert adjoint == False, "adjoint not implemented"

        def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom):
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

        gcs_r = _get_gpmap(elm_wf, 3, self.q_pbgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = _get_gpmap(elm_wf, 1, self.q_pbgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r
        lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
        gc_r = gc_c.view(self.rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        return gc_r


    def adjoint(self, obj):
        assert 0, "implement if needed"
        return self.act(obj, adjoint=True)
    
    
    def update_field(self, field, fn=''):
        pass
    

class beam:
    def __init__(self, operator_desc):
        self.beamwidth = operator_desc['beamwidth']
        self.mmax = operator_desc['mmax']
        self.beam = operator_desc['beam']


    def act(self, obj, adjoint=False):
        if adjoint:
            return cli(almxfl(obj, self.beam, self.mmax))
        return almxfl(obj, self.beam, self.mmax, adjoint=adjoint)   


    def adjoint(self, obj):
        return self.act(obj, adjoint=True)
    

    def __mul__(self, obj, other):
        return self.act(obj)
    
