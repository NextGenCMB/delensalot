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


class lensing:
    def __init__(self, operator_desc):
        self.ID = 'lensing'
        self.Lmin = operator_desc["Lmin"]
        self.lm_max = operator_desc["lm_max"]
        self.LM_max = operator_desc["LM_max"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.geominfo = operator_desc["geominfo"]
        self.geomlib = get_geom(operator_desc['geominfo'])
        self.field = {component: None for component in self.component}
        self.ffi = deflection(self.geomlib, np.zeros(shape=hp.Alm.getsize(*self.LM_max)), self.LM_max[1], numthreads=operator_desc.get('tr', 8), verbosity=False, epsilon=operator_desc['epsilon'])


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


class birefringence:
    def __init__(self, operator_desc):
        self.ID = 'birefringence'
        self.Lmin = operator_desc["Lmin"],
        self.lm_max = operator_desc["lm_max"]
        self.LM_max = operator_desc["LM_max"]
        self.component = operator_desc["component"]
        self.geominfo = operator_desc["geominfo"]
        self.geomlib = get_geom(operator_desc['geominfo'])
        self.ffi = deflection(self.geomlib, np.zeros(shape=hp.Alm.getsize(*self.LM_max)), self.LM_max[1], numthreads=operator_desc.get('tr', 8), verbosity=False, epsilon=1)
        self.field = {component: None for component in self.component}


    # spin doesn't do anything here, but parameter is needed as joint operator passes it to all operators
    # NOTE this is alm2alm
    def act(self, obj, spin=None, adjoint=False):
        f = np.array([self.field[comp].flatten() for comp in self.component], dtype=complex)
        buff = alm_copy(f[0], None, *self.lm_max)
        buff_real = self.ffi.geom.alm2map(buff, lmax=self.LM_max[0], mmax=self.LM_max[1], nthreads=8)
        Q, U = self.ffi.geom.alm2map_spin(obj, spin=2, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=8)
                
        angle = 2 * buff_real
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        Q_rot = cos_a * Q - sin_a * U
        U_rot = sin_a * Q + cos_a * U

        if adjoint:
            Q_rot, U_rot = cos_a * Q + sin_a * U, -sin_a * Q + cos_a * U

        Elm_rot, Blm_rot = self.ffi.geom.map2alm_spin(np.array([Q_rot, U_rot]), spin=2, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
        return np.array([Elm_rot, Blm_rot])
        
    
    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)

    def set_field(self, field):
        self.field['f'] = field


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
    
