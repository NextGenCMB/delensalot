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


    def act(self, obj, adjoint=False):
        assert 0, "subclass this"


    def set_field(self, simidx, it, component=None):
        assert 0, "subclass this"


class multiply:
    def __init__(self, descr):
        self.factor = descr["factor"]
    

    def act(self, obj, spin=None, adjoint=False):
        if adjoint:
            return np.adj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, simidx, it, component=None):
        pass


class joint:
    def __init__(self, operators):
        self.operators = operators
    

    def act(self, obj, spin=None, adjoint=False):
        for operator in self.operators:
            buff = operator.act(obj, spin=spin)
            obj = buff
        return obj
    

    def adjoint(self, obj, spin=None):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin=spin)
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
    

    def act(self, obj, spin=None):
        for operator in self.operators:
            buff = operator.act(obj, spin=spin)
            obj = buff
        return obj
    

    def adjoint(self, obj, spin=None):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin=spin)
            obj = buff
        return obj
    

    def set_field(self, simidx, it, component=None):
        for operator in self.operators:
            operator.set_field(simidx, it)
    

class wf_operator:
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
    

    def set_field(self, simidx, it, component=None):
        # FIXME each operator has its own fields to update
        for operator in self.operators:
            operator.set_field(simidx, it)


class lensing(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.field_fns = operator_desc["field_fns"]
        self.Lmin = operator_desc["Lmin"]
        self.lm_max = operator_desc["lm_max"]
        self.LM_max = operator_desc["LM_max"]
        self.perturbative = operator_desc["perturbative"]
        self.components = operator_desc["components"]
        self.field = {component: None for component in self.components}
        self.ffi = operator_desc["ffi"]

    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False):
        assert adjoint == False, "adjoint not implemented"
        assert spin is not None, "spin not provided"
       
        if self.perturbative: # Applies perturbative remapping
            return 
            # get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            # geom, sht_tr = self.fq.ffi.geom, self.fq.ffi.sht_tr
            # d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
            # dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
            # del dp, dm, d1
            # elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
        else:  
            # NOTE this is used for B-lensing template, and in eq. 21 of the paper
            # ffi = self.fq.ffi.change_dlm([dlm, None], self.mmax_qlm)
            # elm, blm = ffi.lensgclm(np.array([elm_wf, np.zeros_like(elm_wf)]), self.mmax_filt, 2, lmaxb, mmaxb)
            
            # NOTE this is used e.g. in eq 22 of the paper.

            # TODO I don't want this alm_copy here
            obj = np.atleast_2d(obj)
            obj = alm_copy(obj[0], None, *self.lm_max)
            # return self.ffi.gclm2lenmap(np.atleast_2d(obj), self.lm_max[1], spin, False)
            return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max[1], 2, *self.lm_max)
    

    def adjoint(self, obj, spin=None):
        assert spin is not None, "spin not provided"
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, simidx, it, component=None):
        if component is None:
            for component in self.components:
                self.set_field(simidx, it, component)
            d = np.array([self.field[comp].flatten() for comp in self.components], dtype=complex)
            h2d = np.sqrt(np.arange(self.LM_max[0] + 1, dtype=float) * np.arange(1, self.LM_max[0] + 2, dtype=float))
            [almxfl(s, h2d, self.LM_max[1], True) for s in d]
            print(f'setting field for lensing operator with it {it}')
            if d.shape[0] == 1:
                d = [d[0],None]
            self.ffi = self.ffi.change_dlm(d, self.LM_max[1])
        else:
            if self.field_cacher.is_cached(opj(self.field_fns[component].format(idx=simidx,it=it))):
                self.field[component] = self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx,it=it)))
            else:
                assert 0, "cannot set field"


class birefringence(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.field_fns = operator_desc['field_fns']
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"]
        self.components = operator_desc["components"]
        self.field = {component: None for component in self.components}


    # spin doesn't do anything here, but parameter is needed as joint operator passes it to all operators
    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False):
        f = np.array([self.field[comp].flatten() for comp in self.components], dtype=complex)

        buff = alm_copy(f[0], None, *self.lm_max)
        if adjoint:
            return np.array([np.exp(2j*buff)*obj[0], np.exp(-2j*buff)*obj[1]])
        return np.array([np.exp(-2j*buff)*obj[0], np.exp(2j*buff)*obj[1]])
        # if adjoint:
        #     return np.array([np.exp(1j*f[0])*obj[0], np.exp(-1j*f[0])*obj[1]])
        # return np.array([np.exp(-1j*f[0])*obj[0], np.exp(1j*f[0])*obj[1]])
    
    
    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)


    def set_field(self, simidx, it, component=None):
        if component is None:
            for component in self.components:
                self.set_field(simidx, it, component)
        self.field[component] = self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx,it=it)))


class spin_raise:
    def __init__(self, operator_desc):
        self.lm_max = operator_desc["lm_max"]


    def act(self, obj, spin=None, adjoint=False):
        # This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        assert adjoint == False, "adjoint not implemented"
        assert spin in [-2, 2], spin

        lmax = Alm.getlmax(obj.size, self.lm_max[1])
        i1, i2 = (2, -1) if spin == -2 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = np.atleast_2d([almxfl(e_wf, fl, self.lm_max[1], False) for e_wf in obj])
        return elm


    def adjoint(self, obj, spin=None):
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


    def act(self, obj, adjoint=False):
        if adjoint:
            return np.array([cli(almxfl(o, self.beam, self.lm_max[1], False)) for o in obj])
        return np.array([almxfl(o, self.beam, self.lm_max[1], False) for o in obj])


    def adjoint(self):
        self.is_adjoint = True
        return self
    

    def __mul__(self, obj, other):
        return self.act(obj)
    
