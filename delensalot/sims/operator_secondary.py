from os.path import join as opj
import numpy as np
import healpy as hp

from lenspyx.remapping.deflection_028 import rtype, ctype
from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV
from delensalot.core import cachers
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy

class base:
    def __init__(self, libdir):
        if isinstance(libdir, str):
            self.field_cacher = cachers.cacher_npy(libdir)
        else:
            self.field_cacher = cachers.cacher_mem(libdir)


    def act(self, obj, adjoint=False):
        assert 0, "subclass this"


    def set_field(self, simidx, component=None):
        assert 0, "subclass this"


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
    

    def set_field(self, simidx, component=None):
        for operator in self.operators:
            operator.set_field(simidx)


class lensing(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.ID = 'lensing'
        
        if operator_desc["field_fns"] is DNaV:
            self.field_fns = {comp: "{idx}" for comp in operator_desc["components"]}
        else:
            self.field_fns = operator_desc["field_fns"]

        self.Lmin = operator_desc["Lmin"]
        self.lm_max = operator_desc["lm_max"]
        self.LM_max = operator_desc["LM_max"]
        self.perturbative = operator_desc["perturbative"]
        self.components = operator_desc["components"]
        self.geominfo = operator_desc["geominfo"]
        self.geomlib = get_geom(operator_desc['geominfo'])
        
        self.field = {component: None for component in self.components}
        self.ffi = deflection(self.geomlib, np.zeros(shape=hp.Alm.getsize(*self.LM_max)), self.LM_max[1], numthreads=operator_desc['tr'], verbosity=False, epsilon=operator_desc['epsilon'])



    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False):
        assert adjoint == False, "adjoint not implemented"
        assert spin is not None, "spin not provided"
       
        if self.perturbative: # Applies perturbative remapping
            return 
            get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            geom, sht_tr = self.fq.ffi.geom, self.fq.ffi.sht_tr
            d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
            dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
            del dp, dm, d1
            elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
        else:
            obj = np.atleast_2d(obj)
            # obj = alm_copy(obj[0], None, *self.lm_max)
            # return self.ffi.gclm2lenmap(np.atleast_2d(obj), self.lm_max[1], spin, False)
            return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max[1], spin, *self.lm_max)
    

    def adjoint(self, obj, spin=None):
        assert spin is not None, "spin not provided"
        return self.act(obj, spin=spin, adjoint=True)
    

    def set_field(self, field):
        
        if isinstance(field, list) or isinstance(field, tuple) or isinstance(field, np.ndarray):
            if len(field) != 1:
                field = np.atleast_2d(field)
            if len(field) == 1:
                field = [field[0], None]
            else:
                field = np.atleast_2d(field)
            
            assert len(field) == 2, "field must be a list of length 2"
        
        self.field = field
        self.ffi = self.ffi.change_dlm(field, self.LM_max[1])


class birefringence(base):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.ID = 'birefringence'
        self.field_fns = operator_desc['field_fns']
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"]
        self.components = operator_desc["components"]
        self.field = {component: None for component in self.components}


    # spin doesn't do anything here, but parameter is needed as joint operator passes it to all operators
    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False):
        if spin == 0:
            return obj
        
        f = np.array([self.field[comp].flatten() for comp in self.components], dtype=complex)

        buff = alm_copy(f[0], None, *self.lm_max)
        # buff = f[0]
        if adjoint:
            return np.array([np.exp(2j*buff)*obj[0], np.exp(-2j*buff)*obj[1]])
        return np.array([np.exp(-2j*buff)*obj[0], np.exp(2j*buff)*obj[1]])
        # if adjoint:
        #     return np.array([np.exp(1j*f[0])*obj[0], np.exp(-1j*f[0])*obj[1]])
        # return np.array([np.exp(-1j*f[0])*obj[0], np.exp(1j*f[0])*obj[1]])
    
    
    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)

    def set_field(self, field):
        self.field['f'] = field

    # def set_field(self, simidx, component=None):
    #     if component is None:
    #         for component in self.components.split("_"):
    #             self.set_field(simidx, component)
    #     self.field[component] = self.field_cacher.load(opj(self.field_fns[component].format(idx=simidx)))
    

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
    
