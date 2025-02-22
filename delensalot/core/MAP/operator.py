from os.path import join as opj

import numpy as np

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.core import cachers
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy
from delensalot.core.MAP import field


template_lensingcomponents = ['p', 'w'] 
template_index_lensingcomponents = {val: i for i, val in enumerate(template_lensingcomponents)}

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1
    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class base:
    def __init__(self, libdir, LM_max):
        zbounds = (-1,1)
        lenjob_geomlib = get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
        thtbounds = (np.arccos(zbounds[1]), np.arccos(zbounds[0]))
        lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
        self.ffi = deflection(lenjob_geomlib, np.zeros(shape=Alm.getsize(*LM_max), dtype=complex), LM_max[1], numthreads=8, verbosity=False, epsilon=1e-7)

        self.field_cacher = cachers.cacher_npy(libdir)


    def act(self, obj, lm_max=None, adjoint=False):
        assert 0, "subclass this"


    def set_field(self, simidx, it, component=None):
        assert 0, "subclass this"


class multiply:
    def __init__(self, descr):
        self.ID = 'multiply'
        self.factor = descr["factor"]
    

    def act(self, obj, spin=None, adjoint=False):
        if adjoint:
            return np.conj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, simidx, it, component=None):
        pass


class joint:
    def __init__(self, operators, out):
        self.operators = operators
        self.space_out = out
    

    def act(self, obj, spin):
        for operator in self.operators:
            if isinstance(operator, secondary_operator):
                obj = operator.act(obj, spin, out=self.space_out)
            else:
                obj = operator.act(obj, spin)
        return obj
    

    def adjoint(self, obj, spin):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin)
            obj = buff
        return obj
    

    def set_field(self, simidx, it, component=None):
        for operator in self.operators:
            operator.set_field(simidx, it)
    

class secondary_operator:
    def __init__(self, desc):
        self.ID = 'secondaries'
        self.operators = desc # ["operators"]


    def act(self, obj, spin=2, adjoint=False, backwards=False, out_sht_mode=None, secondary=None, out='alm'):
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators if not adjoint else self.operators[::-1]
        for idx, operator in enumerate(operators):
            if isinstance(operator, lensing):
                obj = operator.act(obj, spin, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode, out=out)
            else:
                obj = operator.act(obj, spin, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode)
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
        super().__init__(operator_desc["libdir"], operator_desc["LM_max"])
        
        self.ID = 'lensing'
        self.LM_max = operator_desc["LM_max"]
        self.lm_max_in = operator_desc["lm_max_in"]
        self.lm_max_out = operator_desc["lm_max_out"]
        # self.Lmin = operator_desc["Lmin"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = field.get_secondary_fns(self.component)

    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None, out='alm'):
        assert spin is not None, "spin not provided"

        if self.perturbative: # Applies perturbative remapping
            assert 0, "implement if needed" 
        else:
            if adjoint and backwards and out_sht_mode == 'GRAD_ONLY':
                return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max_in[1], spin, *self.lm_max_out, backwards=backwards, out_sht_mode=out_sht_mode)
            else:
                if out == 'map':
                    lmax = Alm.getlmax(obj.shape[-1], self.lm_max_out[1])
                    return self.ffi.gclm2lenmap(np.atleast_2d(obj), lmax, spin, False)
                elif out == 'alm':
                    obj = np.atleast_2d(obj)
                    lm_obj = Alm.getlmax(obj[0].size, None)
                    if lm_obj == self.lm_max_in[0]:
                        return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max_in[1], spin, *self.lm_max_out)
                    else:
                        return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max_out[1], spin, *self.lm_max_in)
    

    def set_field(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if component is None:
            component = self.component

        comps_ = [component] if isinstance(component, str) else sorted(
            set(component) & set(self.component), key=lambda x: template_index_lensingcomponents.get(x, ''))

        for comp in comps_:
            field_path = opj(self.field_fns[comp].format(idx=idx, idx2=idx2, it=it))
            if self.field_cacher.is_cached(field_path):
                self.field[comp] = self.klm2dlm(self.field_cacher.load(field_path)[0])
            else: 
                assert 0, f"Cannot set field with it={it} and idx={idx}_{idx2}"

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
        super().__init__(operator_desc["libdir"], operator_desc["LM_max"])
        
        self.ID = 'birefringence'
        self.LM_max = operator_desc["LM_max"]
        self.lm_max_in = operator_desc["lm_max_in"]
        self.lm_max_out = operator_desc["lm_max_out"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = field.get_secondary_fns(self.component)


    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None):
        obj = np.atleast_2d(obj)
        lmax = Alm.getlmax(obj[0].size, None)

        
        # NOTE if no B component (e.g. for generating template, or ivfres with Birefringence acting first), I set B to zero
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


    def set_field(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if component is None:
            for comp in self.component:
                self.set_field(idx, it, comp, idx2)
        elif isinstance(component, list):
            for comp in list(set(component) & set(self.component)):
                self.set_field(idx, it, comp, idx2)
        else:
            self.field[component] = alm_copy(self.field_cacher.load(opj(self.field_fns[component].format(idx=idx, idx2=idx2, it=it))), None, *self.LM_max)


class spin_raise:
    def __init__(self, lm_max):
        self.ID = 'spin_raise'
        self.lm_max = lm_max

    def act(self, obj, spin=None, adjoint=False):
        # This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        assert adjoint == False, "adjoint not implemented"
        # lmax = Alm.getlmax(obj.size, self.lm_max[1])
        # assert spin in [-2, 2], spin
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, self.lm_max[0] + i1 + 1, dtype=float) * np.arange(i2, self.lm_max[0] + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        print(obj.shape, fl.shape)
        elm = np.atleast_2d(almxfl(obj, fl, self.lm_max[1], False))
        return elm


    def adjoint(self, obj, spin=None):
        assert 0, "implement if needed"
        return self.act(obj, adjoint=True, spin=spin)
    

    def set_field(self, simidx, it, component=None):
        pass
    

class beam:
    def __init__(self, operator_desc):
        self.ID = 'beam'
        self.transferfunction = operator_desc['transferfunction']
        # self.tebl2idx = {'t': 0, 'e': 1, 'b': 2}
        self.tebl2idx = {'e': 0, 'b': 1}
        self.idx2tebl = {v: k for k, v in self.tebl2idx.items()}
        self.is_adjoint = False
        self.lm_max = operator_desc['lm_max']


    def act(self, obj, adjoint=False):
        # assert self.lm_max[0] == Alm.getlmax(obj[0].size, None), (self.lm_max[0], obj.shape, Alm.getlmax(obj[0].size, None))
        # FIXME change counting for T and MV
        if adjoint: 
            return np.array([cli(almxfl(o, self.transferfunction[self.idx2tebl[oi]], self.lm_max[0], False)) for oi, o in enumerate(obj)])
        return np.array([almxfl(o, self.transferfunction[self.idx2tebl[oi]], self.lm_max[0], False) for oi, o in enumerate(obj)])


    def adjoint(self):
        self.is_adjoint = True
        return self


    def __mul__(self, obj, other):
        return self.act(obj)
    

class inoise_operator:
    def __init__(self, nlev, lm_max):
        self.ID = 'inoise'
        self.nlev = nlev
        self.lm_max = lm_max
        self.n1eblm = [
            cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2,
            cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2
        ]


    def act(self, obj, adjoint=False):
        assert self.lm_max[0] == Alm.getlmax(obj[0].size, None), (self.lm_max[0], Alm.getlmax(obj[0].size, None))
        if adjoint:
            return np.array([cli(almxfl(o, self.n1eblm[oi], self.lm_max[1], False)) for oi, o in enumerate(obj)])
        return np.array([almxfl(o, self.n1eblm[oi], self.lm_max[1], False) for oi, o in enumerate(obj)])


    def adjoint(self):
        self.is_adjoint = True
        return self

    def __mul__(self, obj, other):
        return self.act(obj)