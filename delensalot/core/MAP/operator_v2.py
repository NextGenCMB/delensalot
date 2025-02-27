import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj

import numpy as np

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.core import cachers
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, almxfl_nd
from delensalot.core.MAP import field

from delensalot.utils import read_map


template_lensingcomponents = ['p', 'w'] 
template_index_lensingcomponents = {val: i for i, val in enumerate(template_lensingcomponents)}


def klm2dlm(klm, LM_max):
    h2d = cli(0.5 * np.sqrt(np.arange(LM_max[0] + 1, dtype=float) * np.arange(1, LM_max[0] + 2, dtype=float)))
    Lmax = Alm.getlmax(klm.size, None)
    return almxfl(klm, h2d, Lmax, False)


def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1
    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class Operator(object):
    def __init__(self, *args, **kwargs):
        zbounds = (-1,1)
        self.lenjob_geomlib = get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
        thtbounds = (np.arccos(zbounds[1]), np.arccos(zbounds[0]))
        self.lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)

        zbounds = (-1,1)
        self.data_geomlib = get_geom(('healpix',{'nside': 2048}))
        thtbounds = (np.arccos(zbounds[1]), np.arccos(zbounds[0]))
        self.data_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)

    def apply_inplace(self, vec_a, vec_b):
        assert 0, 'not implemented -- subclass this'

    def apply_adjoint_inplace(self, vec_a, vec_b):
        assert 0, 'not implemented -- subclass this'

    def apply_derivative(self, param, vec_a, vec_b):
        assert 0, 'I cant derive w.r.t. anything -- subclass this'

    def apply_adjoint_derivative(self, param, vec_a, vec_b):
        assert 0, 'I cant derive w.r.t. anything -- subclass this'

    def eval_speed(self, vec_a, vec_b):
        """Evaluate the execution speed of the operator and its adjoint
        """
        # TODO: this we can actually test here
        pass

    def test_adjointness(self):
        # TODO: this we can actually test here
        pass

    def allocate(self):
        """Intended usage is necessary preparations for e.g., cg-inversion
        """
        assert 0, 'implement this'

    def deallocate(self):
        """Clean-up all potentially allocated array to avoid unnecessary mess

            To be used e.g. at the end of a Wiener filter iterative search

        """
        assert 0, 'implement this'


class Lensing(Operator):
    def __init__(self, operator_desc):
        super().__init__()
        
        self.ID = 'lensing'
        self.LM_max = operator_desc["LM_max"]
        self.lm_max_in = operator_desc["lm_max_in"]
        self.lm_max_out = operator_desc["lm_max_out"]
        # self.Lmin = operator_desc["Lmin"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.spin = operator_desc["spin"]

        # NOTE expecting convergence
        # self.field = {comp: operator_desc['fieldlm'][comp] for comp in self.component}
        # d = np.array([self.field[comp].flatten() for comp in self.component], dtype=complex)
        # if d.shape[0] == 1:
        #     d = [d[0], None] if self.component[0] == 'p' else [np.zeros_like(d[0], dtype=complex), d[0]]
        # self.ffi = deflection(self.lenjob_geomlib, d, self.LM_max[1], numthreads=8, verbosity=False, epsilon=1e-7)
        

    @log_on_start(logging.DEBUG, "lensing: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "lensing done", logger=log)
    def apply_inplace(self, obj, map_out):
        if map_out:
            self.ffi.gclm2lenmap(np.atleast_2d(obj), self.lm_max_out[0], self.spin, False, map_out=map_out)
        else:
            obj = np.atleast_2d(obj)
            lm_obj = Alm.getlmax(obj.shape[1], None)
            # if lm_obj == self.lm_max_in[0]:
            return self.ffi.lensgclm(np.atleast_2d(obj), lm_obj, self.spin, *self.lm_max_out)
            # else:
            #     return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max_out[1], self.spin, *self.lm_max_in)
                    

    @log_on_start(logging.DEBUG, "lensing: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "lensing done", logger=log)
    def apply_adjoint_inplace(self, obj, obj_out=None):
        return self.ffi.lensgclm(np.atleast_2d(obj), self.lm_max_in[1], self.spin, *self.lm_max_out, backwards=True, out_sht_mode='GRAD_ONLY', gclm_out=obj_out)
        # return self.ffi.lenmap2gclm(np.atleast_2d(obj), self.lm_max_in[1], self.spin, *self.lm_max_out, backwards=True, out_sht_mode='GRAD_ONLY', alm_out=alm_out)


    def set_field(self, fieldlm):
        if fieldlm.shape[0] == 1:
            d = [fieldlm[0], None] if self.component[0] == 'p' else [np.zeros_like(fieldlm[0], dtype=complex), fieldlm[0]]
        else:
            d = fieldlm
        self.ffi = deflection(self.lenjob_geomlib, d[0], self.LM_max[1], dclm=d[1], numthreads=6, verbosity=False, epsilon=1e-7)


class Birefringence(Operator):
    def __init__(self, operator_desc):
        super().__init__()
        self.ID = 'birefringence'
        self.LM_max = operator_desc["LM_max"]
        self.lm_max = operator_desc["lm_max"]
        self.component = operator_desc["component"]

        # NOTE expecting convergence here


    @log_on_start(logging.DEBUG, "birefringence: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "birefringence done", logger=log)
    def apply_inplace(self, obj):
        obj = np.atleast_2d(obj)
        obj_shape_in_leading = obj.shape[0]
        
        # NOTE if no B component, I set B to zero
        if obj.shape[0] == 1:
            obj = [obj[0], np.zeros_like(obj[0])+np.zeros_like(obj[0])*1j] 
        Q, U = self.lenjob_geomlib.alm2map_spin(obj, 2, *self.lm_max, 6)  
        
        Q_rot = self.cos_a * Q - self.sin_a * U
        U_rot = self.sin_a * Q + self.cos_a * U

        EBlm_rot = self.lenjob_geomlib.map2alm_spin(np.array([Q_rot, U_rot]), 2, *self.lm_max, 6)

        obj = EBlm_rot[0] if obj_shape_in_leading == 1 else EBlm_rot
        return obj
  

    def apply_adjoint_inplace(self, obj):
        obj = np.atleast_2d(obj)
        obj_shape_in_leading = obj.shape[0]

        # NOTE if no B component, I set B to zero
        if obj.shape[0] == 1:
            obj = [obj[0], np.zeros_like(obj[0])+np.zeros_like(obj[0])*1j] 
        Q, U = self.lenjob_geomlib.alm2map_spin(obj, 2, *self.lm_max, 6)  

        Q_rot = self.cos_a * Q + self.sin_a * U
        U_rot = -self.sin_a * Q + self.cos_a * U
        EBlm_rot = self.lenjob_geomlib.map2alm_spin(np.array([Q_rot, U_rot]), 2, *self.lm_max, 6)

        obj = EBlm_rot[0] if obj_shape_in_leading == 1 else EBlm_rot
        return obj


    def set_field(self, fieldlm):
        self.angle = 2 * self.lenjob_geomlib.alm2map(fieldlm.squeeze(), *self.LM_max, 6)
        self.cos_a, self.sin_a = np.cos(self.angle), np.sin(self.angle)


class Multiply(Operator):
    def __init__(self, descr):
        super().__init__()
        self.ID = 'multiply'
        self.factor = descr["factor"]
    

    @log_on_start(logging.DEBUG, "multiply: {obj.shape}")
    @log_on_end(logging.DEBUG, "multiply done")
    def apply_inplace(self, obj, spin=None):
        obj *= self.factor
    

    @log_on_start(logging.DEBUG, "multiply adjoint: {obj.shape}")
    @log_on_end(logging.DEBUG, "multiply adjoint done")
    def apply_inplace_adjoint(self, obj, spin=None):
        obj *= np.conj(self.factor)


class SpinRaise(Operator):
    def __init__(self, lm_max):
        self.ID = 'spin_raise'
        self.lm_max = lm_max


    @log_on_start(logging.DEBUG, "spin_raise: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "spin_raise done", logger=log)
    def apply_inplace(self, obj, spin):
        # NOTE This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, self.lm_max[0] + i1 + 1, dtype=float) * np.arange(i2, self.lm_max[0] + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        almxfl(obj, fl, self.lm_max[1], True)


    def apply_adjoint_inplace(self, obj):
        assert False, "adjoint not implemented"
    

class Beam(Operator):
    def __init__(self, operator_desc):
        self.ID = 'beam'
        self.transferfunction = operator_desc['transferfunction']
        self.tebl2idx = {'e': 0, 'b': 1} # TODO self.tebl2idx = {'t': 0, 'e': 1, 'b': 2}
        self.idx2tebl = {v: k for k, v in self.tebl2idx.items()}
        self.lm_max = operator_desc['lm_max']


    @log_on_start(logging.DEBUG, "beam: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "beam done", logger=log)
    def apply_inplace(self, obj, obj_out=None):
        # assert self.lm_max[0] == Alm.getlmax(obj[0].size, None), (self.lm_max[0], obj.shape, Alm.getlmax(obj[0].size, None))
        # FIXME change counting for T and MV
        if Alm.getlmax(obj.shape[1], None) != self.lm_max[0]:
            obj_out = np.zeros((obj.shape[0], Alm.getsize(*self.lm_max)), dtype=complex)
        else:
            obj_out = obj
        for ob in obj:
            oi = self.tebl2idx['e']
            obj_out[oi] = alm_copy(ob, Alm.getlmax(obj.shape[1], None), *self.lm_max)
            almxfl(obj_out[oi], self.transferfunction[self.idx2tebl[oi]], self.lm_max[0], True)
            return obj_out


    def apply_adjoint_inplace(self, obj):
        for ob in obj:
            oi = self.tebl2idx['e']
            almxfl(ob, self.transferfunction[self.idx2tebl[oi]], self.lm_max[0], True)
        ob = cli(ob)
    

class Add(Operator):
    def __init__(self, operator_desc):
        super().__init__(operator_desc)
        self.ID = 'add'
    

    @log_on_start(logging.DEBUG, "add: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "add done", logger=log)
    def apply(self, obj1, obj2):
        if obj2.dtype in (np.complex64, np.complex128):
            obj1 += obj2
        elif obj2.dtype in (np.float32, np.float64):
            obj1 = self.data_geomlib.synthesis(obj1, 2, *self.lm_max, 6, apply_weights=False)
            obj1 += obj2
        return obj1
            

    @log_on_start(logging.DEBUG, "add adjoint: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "add adjoint done", logger=log)
    def apply_adjoint(self, obj):
        assert 0, 'not implemented'


class InverseNoiseVariance(Operator):
    def __init__(self, nlev, lm_max, niv_desc, transferfunction, libdir, spectrum_type=None, OBD=None, OBD_rescale=None, OBD_libdir=None, mask=None, sky_coverage=None, rhits_normalised=None):
        super().__init__(libdir, lm_max)
        self.ID = 'inoise'

        self.nlev = nlev
        self.lm_max = lm_max
        self.niv_desc = [niv_desc['P']]
        self.transferfunction = transferfunction
        spectrum_type = spectrum_type
        OBD = OBD
        self.mask = mask
        self.sky_coverage = sky_coverage
        rhits_normalised = rhits_normalised
        self.n1eblm = [
            cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2,
            cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2
        ]


    @log_on_start(logging.DEBUG, "InverseNoiseVariance: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "InverseNoiseVariance done", logger=log)
    def apply(self, obj):
        if obj.dtype in (np.complex64, np.complex128):
            for oi, o in enumerate(obj):
                almxfl(o, self.n1eblm[oi], self.lm_max[1], True)
            return obj
        elif obj.dtype in (np.float32, np.float64):
            if len(self.niv_desc) == 1:  #  QQ = UU
                obj *= self.niv_desc[0]
            elif len(self.niv_desc) == 3:  # QQ, QU, UU
                assert self.template is None
                qmap, umap = obj
                qmap_copy = qmap.copy()
                qmap *= self.niv_desc[0]
                qmap += self.niv_desc[1] * umap
                umap *= self.niv_desc[2]
                umap += self.niv_desc[1] * qmap_copy
                del qmap_copy
            else:
                assert 0
            eblm = self.data_geomlib.adjoint_synthesis(obj, 2, *self.lm_max, 6, apply_weights=False)
            return eblm


    def apply_adjoint(self, obj):
        return np.array([cli(almxfl(o, self.n1eblm[oi], self.lm_max[1], False)) for oi, o in enumerate(obj)])
    

    def get_febl(self, transferfunction):
        if self.sky_coverage == 'full':
            ret = _extend_cl(transferfunction['e']*2, len(self.n1eblm[0])-1) * self.n1eblm[0]
            return ret, None

        if len(self.niv_desc) == 1:
            nlev_febl = 10800. / np.sqrt(np.sum(read_map(self.niv_desc[0])) / (4.0 * np.pi)) / np.pi
        elif len(self.niv_desc) == 3:
            nlev_febl = 10800. / np.sqrt(
                (0.5 * np.sum(read_map(self.niv_desc[0])) + np.sum(read_map(self.niv_desc[2]))) / (4.0 * np.pi)) / np.pi
        else:
            assert 0
        self._nlevp = nlev_febl
        niv_cl_e = transferfunction['e'] ** 2  / (self._nlevp/ 180. / 60. * np.pi) ** 2
        niv_cl_b = transferfunction['b'] ** 2  / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return niv_cl_e, niv_cl_b.copy()


class Compound(Operator):
    def __init__(self, operators, spin):
        super().__init__()
        self.operators = operators
        self.spin = spin
    

    @log_on_start(logging.DEBUG, "Compound apply_inplace: {obj.shape}", logger=log)  
    @log_on_end(logging.DEBUG, "Compound apply_inplace done", logger=log)  
    def apply_inplace(self, obj, spin=None):
        for operator in self.operators:
            if isinstance(operator, Secondary):
                obj = operator.apply_inplace(obj)
            else:
                operator.apply_inplace(obj, spin)
        return self.lenjob_geomlib.synthesis(obj, self.spin, *self.operators[-1].operators[-1].lm_max_in, 6)
    

    def apply_adjoint_inplace(self, obj):
        assert 0, 'not implemented'
        for operator in self.operators[::-1]:
            buff = operator.apply_adjoint.inplace(obj)
            obj = buff
        return obj
    

    def set_field(self, field):
        for operator in self.operators:
            operator.set_field(field[operator.ID])
    

class Secondary(Operator):
    def __init__(self, desc):
        self.ID = 'secondaries'
        self.operators = desc # ["operators"]


    @log_on_start(logging.DEBUG, "secondary: {obj.shape}", logger=log)  
    @log_on_end(logging.DEBUG, "secondary done", logger=log)  
    def apply_inplace(self, obj, secondary=None, obj_out=None):
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators
        for idx, operator in enumerate(operators):
            # print(operator.ID)
            if operator.ID in secondary:
                if isinstance(operator, Lensing):
                    obj = operator.apply_inplace(obj, obj_out)
                else:
                    obj = operator.apply_inplace(obj)
        return obj
    

    @log_on_start(logging.DEBUG, "secondary adjoint: {obj.shape}", logger=log)  
    @log_on_end(logging.DEBUG, "secondary adjoint done", logger=log)  
    def apply_adjoint_inplace(self, obj, secondary=None):
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators[::-1]
        for idx, operator in enumerate(operators):
            # print(operator.ID)
            if operator.ID in secondary:
                if isinstance(operator, Lensing):
                    obj = operator.apply_adjoint_inplace(obj)
                else:
                    obj = operator.apply_adjoint_inplace(obj)
        return obj
    

    def set_field(self, field):
        for operator in self.operators:
            operator.set_field(field[operator.ID])