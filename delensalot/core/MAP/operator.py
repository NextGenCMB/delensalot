from os.path import join as opj

import numpy as np

from delensalot.core import cachers
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy

template_lensingcomponents = ['p', 'w'] 
template_index_lensingcomponents = {val: i for i, val in enumerate(template_lensingcomponents)}

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
            return np.conj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj, spin=None, lm_max=None):
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, simidx, it, component=None):
        pass


class joint:
    def __init__(self, operators):
        self.operators = operators
    

    def act(self, obj, spin, lm_max_in, lm_max_out):
        for operator in self.operators:
            buff = operator.act(obj, spin, lm_max_in, lm_max_out)
            obj = buff
        return obj
    

    def adjoint(self, obj, spin, lm_max_in, lm_max_out):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin, lm_max_in, lm_max_out)
            obj = buff
        return obj
    

    def set_field(self, simidx, it, component=None):
        for operator in self.operators:
            operator.set_field(simidx, it)
    

class secondary_operator:
    def __init__(self, operators):
        self.operators = operators


    def act(self, obj, spin=2, lm_max_in=None, lm_max_out=None, adjoint=False, backwards=False, out_sht_mode=None, secondary=None):
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators if not adjoint else self.operators[::-1]

        for idx, operator in enumerate(operators):
            obj = operator.act(obj, spin, lm_max_in, lm_max_out, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode)
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
        self.LM_max = operator_desc["LM_max"]
        # self.Lmin = operator_desc["Lmin"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = operator_desc["field_fns"]
        self.ffi = operator_desc["ffi"]

        self.complist_sorted = sorted(self.component, key=lambda x: template_index_lensingcomponents.get(x, ''))

    # NOTE this is alm2alm
    def act(self, obj, spin=None, lm_max_in=None, lm_max_out=None, adjoint=False, backwards=False, out_sht_mode=None):
        assert spin is not None, "spin not provided"

        if self.perturbative: # Applies perturbative remapping
            return 
        else:
            spin = 2 if spin == None else spin
            # return self.ffi.gclm2lenmap(np.atleast_2d(obj), self.lm_max[1], spin, False)

            if adjoint and backwards and out_sht_mode == 'GRAD_ONLY':
                return self.ffi.lensgclm(np.atleast_2d(obj), lm_max_in[1], spin, *lm_max_out, backwards=backwards, out_sht_mode=out_sht_mode)
            
            obj = np.atleast_2d(obj)
            return self.ffi.lensgclm(np.atleast_2d(obj), lm_max_in[1], spin, *lm_max_out)
    

    def set_field(self, simidx, it, component=None):
        if component is None:
            component = self.component

        comps_ = [component] if isinstance(component, str) else sorted(
            set(component) & set(self.component), key=lambda x: template_index_lensingcomponents.get(x, ''))

        for comp in comps_:
            field_path = opj(self.field_fns[comp].format(idx=simidx, it=it))
            if self.field_cacher.is_cached(field_path):
                self.field[comp] = self.klm2dlm(self.field_cacher.load(field_path)[0])
            else: 
                assert 0, f"Cannot set field with it={it} and simidx={simidx}"

        d = np.array([self.field[comp].flatten() for comp in comps_], dtype=complex)
        if d.shape[0] == 1:
            d = [d[0], None] if comps_[0] == 'p' else [np.zeros_like(d[0], dtype=complex), d[0]]
        self.ffi = self.ffi.change_dlm(d, self.LM_max[1])


    def klm2dlm(self, klm):
        h2d = cli(0.5 * np.sqrt(np.arange(self.LM_max[0] + 1, dtype=float) * np.arange(1, self.LM_max[0] + 2, dtype=float)))
        Lmax = Alm.getlmax(klm.size, None)
        return almxfl(klm, h2d, Lmax, False)


class birefringence(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        
        self.ID = 'birefringence'
        self.LM_max = operator_desc["LM_max"]
        # self.Lmin = operator_desc["Lmin"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = operator_desc['field_fns']
        self.ffi = operator_desc["ffi"]


    # NOTE this is alm2alm
    def act(self, obj, spin=None, lm_max_in=None, lm_max_out=None, adjoint=False, backwards=False, out_sht_mode=None):
        lmax = Alm.getlmax(obj[0].size, None)

        # NOTE if no B component (e.g. for generating template), I set B to zero
        obj = np.atleast_2d(obj)
        if obj.shape[0] == 1:
            obj = [obj[0], np.zeros_like(obj[0])+np.zeros_like(obj[0])*1j] 
        Q, U = self.ffi.geom.alm2map_spin(obj, 2, lmax, lmax, 8)  
        angle = 2 * self.ffi.geom.alm2map(self.field[self.component[0]], *self.LM_max, 8)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        Q_rot = cos_a * Q - sin_a * U
        U_rot = sin_a * Q + cos_a * U

        if adjoint:
            Q_rot, U_rot = cos_a * Q + sin_a * U, -sin_a * Q + cos_a * U

        Elm_rot, Blm_rot = self.ffi.geom.map2alm_spin(np.array([Q_rot, U_rot]), 2, lmax, lmax, 8)

        return np.array([Elm_rot, Blm_rot])


    def set_field(self, simidx, it, component=None):
        if component is None:
            for comp in self.component:
                self.set_field(simidx, it, comp)
        elif isinstance(component, list):
            for comp in list(set(component) & set(self.component)):
                self.set_field(simidx, it, comp)
        else:
            self.field[component] = alm_copy(self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx,it=it))), None, *self.LM_max)


class spin_raise:
    def __init__(self, operator_desc):
        self.lm_max = operator_desc["lm_max"]

    def act(self, obj, spin=None, lm_max_in=None, lm_max_out=None, adjoint=False):
        # This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        assert adjoint == False, "adjoint not implemented"
        lm_max = lm_max_out
        # lmax = Alm.getlmax(obj.size, self.lm_max[1])
        # assert spin in [-2, 2], spin
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lm_max[0] + i1 + 1, dtype=float) * np.arange(i2, lm_max[0] + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = np.atleast_2d(almxfl(obj, fl, lm_max[1], False))
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


    def act(self, obj, lm_max_in=None, lm_max_out=None, adjoint=False):
        lm_max = self.lm_max if lm_max_out is None else lm_max_out
        if adjoint:
            return np.array([cli(almxfl(o, self.beam, lm_max[1], False)) for o in obj])
        return np.array([almxfl(o, self.beam, lm_max[1], False) for o in obj])


    def adjoint(self, lm_max=None):
        self.is_adjoint = True
        return self

    def __mul__(self, obj, other):
        return self.act(obj)
    

class noise:
    def __init__(self, operator_desc):
        self.noise = operator_desc['noise']
        self.n1e = self.n1elm * 0.5
        self.n1b = self.n1blm * 0.5

        self.n1elm = cli(_extend_cl(self.nlevp**2, self.lm_max_sky[0])) * (180 * 60 / np.pi) ** 2
        self.n1blm = cli(_extend_cl(self.nlevp**2, self.lm_max_sky[0])) * (180 * 60 / np.pi) ** 2

    def act(self, obj, lm_max_in=None, lm_max_out=None, adjoint=False):
        lm_max = self.lm_max if lm_max_out is None else lm_max_out
        if adjoint:
            return np.array([cli(almxfl(o, self.noise, lm_max[1], False)) for o in obj])
        return np.array([almxfl(o, self.noise, lm_max[1], False) for o in obj])


    def adjoint(self, lm_max=None):
        self.is_adjoint = True
        return self

    def __mul__(self, obj, other):
        return self.act(obj)