from os.path import join as opj

import numpy as np

import healpy as hp

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.core import cachers

from delensalot.config.config_helper import data_functions as df

from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy

class base:
    def __init__(self, libdir):
        self.field_cacher = cachers.cacher_npy(libdir)


    def act(self, obj, lm_max=None, adjoint=False):
        assert 0, "subclass this"


    def set_field(self, simidx, it, component=None):
        assert 0, "subclass this"


class multiply:
    def __init__(self, descr):
        self.factor = descr["factor"]
    

    def act(self, obj, spin=None, lm_max=None, adjoint=False):
        if adjoint:
            return np.adj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj, spin=None, lm_max=None):
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, simidx, it, component=None):
        pass


class joint:
    def __init__(self, operators):
        self.operators = operators
    

    def act(self, obj, spin, lmax_in, lm_max):
        for operator in self.operators:
            buff = operator.act(obj, spin, lmax_in, lm_max)
            obj = buff
        return obj
    

    def adjoint(self, obj, spin, lmax_in, lm_max):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin, lmax_in, lm_max)
            obj = buff
        return obj
    

    def set_field(self, simidx, it, component=None):
        for operator in self.operators:
            operator.set_field(simidx, it)


class ivf_operator:
    # This is a composite operator, consisting of the secondaries-operators.
    def __init__(self, operators):
        self.operators = operators
        self.lm_max = operators[0].lm_max if hasattr(operators[0], 'lm_max') else operators[1].lm_max # FIXME hoping these operators have what im looking for
    

    def act(self, obj, spin, lmax_in, lm_max):
        for operator in self.operators:
            buff = operator.act(obj, spin, lmax_in, lm_max)
            obj = buff
        return obj
    

    def adjoint(self, obj, spin, lmax_in, lm_max):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin, lmax_in, lm_max)
            obj = buff
        return obj
    

    def set_field(self, simidx, it, component=None):
        for operator in self.operators:
            operator.set_field(simidx, it)
    

class wf_operator:
    def __init__(self, operators):
        self.operators = operators
        # self.lm_max = operators[0].lm_max if hasattr(operators[0], 'lm_max') else operators[1].lm_max # FIXME hoping these operators have what im looking for
    

    def act(self, obj, spin=2, lmax_in=None, lm_max=None, adjoint=False, backwards=False, out_sht_mode=None, secondary=None):
        buff = None
        if secondary is None:
            secondary = [operator.ID for operator in self.operators]
        if not adjoint:
            for operator in self.operators:
                if operator.ID in secondary:
                    buff = operator.act(obj, spin, lmax_in=lmax_in, lm_max=lm_max, adjoint=False, backwards=False, out_sht_mode=out_sht_mode)
                    obj = buff
            return obj
        if adjoint:
            for operator in self.operators[::-1]:
                if operator.ID in secondary:
                    buff = operator.act(obj, spin, lmax_in=lmax_in, lm_max=lm_max, adjoint=True, backwards=True, out_sht_mode=out_sht_mode)
                    obj = buff
        return obj
    

    def set_field(self, simidx, it, secondary=None, component=None):
        if secondary is None:
            secondary = [operator.ID for operator in self.operators]
        for operator in self.operators:
            if operator.ID in secondary:
                if component is None or component not in operator.component:
                    component = operator.component
                operator.set_field(simidx, it, component)


class lensing(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.ID = 'lensing'
        self.field_fns = operator_desc["field_fns"]
        self.Lmin = operator_desc["Lmin"]
        self.lm_max = operator_desc["lm_max"]
        self.LM_max = operator_desc["LM_max"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.ffi = operator_desc["ffi"]

    # NOTE this is alm2alm
    def act(self, obj, spin=None, lmax_in=None, lm_max=None, adjoint=False, backwards=False, out_sht_mode=None):
        assert spin is not None, "spin not provided"
        lm_max = self.lm_max if lm_max is None else lm_max

        if self.perturbative: # Applies perturbative remapping
            return 
        else:
            spin = 2 if spin == None else spin
            # return self.ffi.gclm2lenmap(np.atleast_2d(obj), self.lm_max[1], spin, False)

            if adjoint and backwards and out_sht_mode == 'GRAD_ONLY':
                return self.ffi.lensgclm(np.atleast_2d(obj), lmax_in, spin, *lm_max, backwards=backwards, out_sht_mode=out_sht_mode)
            
            obj = np.atleast_2d(obj)
            obj = alm_copy(obj[0], None, *lm_max)
            return self.ffi.lensgclm(np.atleast_2d(obj), lmax_in, spin, *lm_max)
    

    def set_field(self, simidx, it, component=None):
        if component is None:
            comps_ = self.component
            for comp in comps_:
                self.set_field(simidx, it, comp)
        elif isinstance(component, list):
            comps_ = list(set(component) & set(self.component))
            for comp in comps_:
                self.set_field(simidx, it, comp)
        else:
            comps_ = [component]
            if self.field_cacher.is_cached(opj(self.field_fns[component].format(idx=simidx,it=it))):
                self.field[component] = self.klm2dlm(self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx,it=it)))[0])
            else:
                assert 0, f"cannot set field with it {it} and simidx {simidx}"
        d = np.array([self.field[comp].flatten() for comp in comps_], dtype=complex)
        if d.shape[0] == 1:
            d = [d[0],None]
        self.ffi = self.ffi.change_dlm(d, self.LM_max[1])


    def klm2dlm(self, klm):
        h2d = cli(0.5 * np.sqrt(np.arange(self.LM_max[0] + 1, dtype=float) * np.arange(1, self.LM_max[0] + 2, dtype=float)))
        return almxfl(klm, h2d, self.LM_max[1], False)


class birefringence(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.field_fns = operator_desc['field_fns']
        self.ID = 'birefringence'
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.LM_max = operator_desc["LM_max"]
        self.component = operator_desc["component"]
        self.ffi = operator_desc["ffi"]
        self.field = {component: None for component in self.component}


    # spin doesn't do anything here, but parameter is needed as joint operator passes it to all operators
    # NOTE this is alm2alm
    def act(self, obj, spin=None, lmax_in=None, lm_max=None, adjoint=False, backwards=False, out_sht_mode=None):
        lm_max = self.lm_max if lm_max is None else lm_max
        f = np.array([self.field[comp].flatten() for comp in self.component], dtype=complex)
        buff = alm_copy(f[0], None, *self.lm_max)
        buff_real = self.ffi.geom.alm2map(buff, *self.LM_max, 8, [-1., 1.])

        # Convert (Elm, Blm) to Q/U maps
        Q, U = self.ffi.geom.alm2map_spin(obj, 2, *self.lm_max, 8, [-1., 1.])
                
        # Apply birefringence rotation
        angle = 2 * buff_real
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        Q_rot = cos_a * Q - sin_a * U
        U_rot = sin_a * Q + cos_a * U

        # If adjoint, reverse the rotation
        if adjoint:
            Q_rot, U_rot = cos_a * Q + sin_a * U, -sin_a * Q + cos_a * U

        # Convert back to (Elm, Blm)
        Elm_rot, Blm_rot = self.ffi.geom.map2alm_spin(np.array([Q_rot, U_rot]), 2, *self.lm_max, 8)

        return np.array([Elm_rot, Blm_rot])


    def set_field(self, simidx, it, component=None):
        if component is None:
            for comp in self.component:
                self.set_field(simidx, it, comp)
        elif isinstance(component, list):
            for comp in list(set(component) & set(self.component)):
                self.set_field(simidx, it, comp)
        else:
            self.field[component] = self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx,it=it)))


class spin_raise:
    def __init__(self, operator_desc):
        self.lm_max = operator_desc["lm_max"]

    def act(self, obj, spin=None, lmax_in=None, lm_max=None, adjoint=False):
        # This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        assert adjoint == False, "adjoint not implemented"
        lm_max = self.lm_max if lm_max is None else lm_max
        # lmax = Alm.getlmax(obj.size, self.lm_max[1])
        # assert spin in [-2, 2], spin
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lm_max[0] + i1 + 1, dtype=float) * np.arange(i2, lm_max[0] + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = np.atleast_2d(almxfl(obj, fl, self.lm_max[1], False))
        return elm


    def adjoint(self, obj, spin=None, lm_max=None):
        assert 0, "implement if needed"
        return self.act(obj, adjoint=True, spin=spin)
    

    def set_field(self, simidx, it, component=None):
        pass
    

class beam:
    def __init__(self, operator_desc):
        self.beamwidth = operator_desc['beamwidth']
        self.lm_max = operator_desc['lm_max']
        self.beam = gauss_beam(df.a2r(self.beamwidth), lmax=self.lm_max[0])
        self.is_adjoint = False


    def act(self, obj, lm_max=None, adjoint=False):
        lm_max = self.lm_max if lm_max is None else lm_max
        if adjoint:
            return np.array([cli(almxfl(o, self.beam, lm_max[1], False)) for o in obj])
        return np.array([almxfl(o, self.beam, lm_max[1], False) for o in obj])


    def adjoint(self, lm_max=None):
        self.is_adjoint = True
        return self
    

    def __mul__(self, obj, other):
        return self.act(obj)
    
