import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom
from lenspyx.remapping.deflection_028 import rtype

from delensalot.core import cachers
from delensalot.core.MAP import field

from delensalot.utils import cli, read_map
from delensalot.utility import utils_qe
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1
    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class Operator:
    def __init__(self, libdir):
        zbounds = (-1,1)
        self.lenjob_geomlib = get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
        thtbounds = (np.arccos(zbounds[1]), np.arccos(zbounds[0]))
        self.lenjob_geomlib # .restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
        self.field_cacher = cachers.cacher_npy(libdir)


    def act(self, obj, lm_max=None, adjoint=False):
        assert 0, "subclass this"


class Multiply:
    def __init__(self, descr):
        
        self.ID = 'multiply'
        self.factor = descr["factor"]
    

    @log_on_start(logging.DEBUG, "multiply", logger=log)
    @log_on_end(logging.DEBUG, "multiply done", logger=log)
    def act(self, obj, spin=None, adjoint=False):
        if adjoint:
            return np.conj(self.factor)*obj
        else:
            return self.factor*obj
    

    def adjoint(self, obj, spin=None):
        return self.act(obj, spin=spin, adjoint=True)


class Compound:
    def __init__(self, operators, out, sht_tr):
        
        self.operators = operators
        self.space_out = out
        self.sht_tr = sht_tr
    

    @log_on_start(logging.DEBUG, "joint", logger=log)  
    @log_on_end(logging.DEBUG, "joint done", logger=log)  
    def act(self, obj, spin):
        assert len(obj) == 3, "obj must be a 3 element array"
        for operator in self.operators:
            if isinstance(operator, Secondary):
                obj = operator.act(obj, spin=spin, out=self.space_out)
            else:
                operator.act(obj, spin)

        if self.space_out == 'map' and obj[0].dtype in [np.complex64, np.complex128]:
            # NOTE this is a hack to catch a birefringence only case and return map
            # NOTE I should rather move the out space here completely
            # FIXME this needs changing
            return self.operators[-1].operators[-1].lenjob_geomlib.synthesis(obj, 2, *self.operators[-1].operators[-1].lm_max, self.sht_tr)
        
        return obj
    

    def adjoint(self, obj, spin):
        for operator in self.operators[::-1]:
            buff = operator.adjoint.act(obj, spin)
            obj = buff
        return obj
    

class Secondary:
    def __init__(self, desc):
        self.ID = 'secondaries'
        self.operators = desc # ["operators"]


    @log_on_start(logging.DEBUG, "secondary", logger=log)  
    @log_on_end(logging.DEBUG, "secondary done", logger=log)  
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None, secondary=None, nomagn=None, out='alm', order='normal'):
        assert order in ['normal', 'reversed'], "order must be 'normal' or 'reversed'. Reversed is used for e.g. template generation"
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators if not adjoint else self.operators[::-1]
        operators = operators if order == 'normal' else operators[::-1]
        for idx, operator in enumerate(operators):
            if operator.ID in secondary:
                if isinstance(operator, Lensing):
                    obj = operator.act(obj, spin=spin, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode, nomagn=nomagn, out=out)
                else:
                    obj = operator.act(obj, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode)
        return obj


    def set_field(self, field):
        for operator in self.operators:
            operator.set_field(field[operator.ID])


    def get_field(self):
        return {operator.ID: operator.get_field() for operator in self.operators}


    def update_lm_max(self, lm_max_in, lm_max_out):
        in_prev, out_prev = self.operators[0].lm_max_in, self.operators[0].lm_max_out
        for operator in self.operators:
            operator.lm_max_in = lm_max_in
            operator.lm_max_out = lm_max_out
        return in_prev, out_prev


class Lensing(Operator):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.ID = 'lensing'
        self.data_key = operator_desc["data_key"]

        self.LM_max = operator_desc["LM_max"]
        self.lm_max_in = operator_desc["lm_max_in"]
        self.lm_max_out = operator_desc["lm_max_out"]
        # self.Lmin = operator_desc["Lmin"]
        self.perturbative = operator_desc["perturbative"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = field.get_secondary_fns(self.component)

        self.sht_tr = operator_desc["sht_tr"]
        self.ffi = deflection(self.lenjob_geomlib, np.zeros(shape=Alm.getsize(*self.LM_max), dtype=complex), self.LM_max[1], numthreads=self.sht_tr, verbosity=False, epsilon=1e-10)


    @log_on_start(logging.DEBUG, "lensing", logger=log)
    # @log_on_end(logging.DEBUG, "lensing done", logger=log)
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None, nomagn=None, out='alm'):
        lmax = Alm.getlmax(np.max([len(o) for o in obj]), None)
        if self.perturbative: # Applies perturbative remapping
            # get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            # geom, sht_tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
            # d1_c = np.empty((geom.npix(),), dtype=elm_wf.dtype)
            # d1_r = d1_c.view(rtype[d1_c.dtype]).reshape((d1_c.size, 2)).T  # real view onto complex array
            # geom.synthesis(dlm, 1, self.lmax_qlm, self.mmax_qlm, sht_tr, map=d1_r, mode='GRAD_ONLY')
            # dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            # dlens_c = -0.5 * ((d1_c.conj()) * dp + d1_c * dm)
            # dlens_r = dlens_c.view(rtype[dlens_c.dtype]).reshape((dlens_c.size, 2)).T  # real view onto complex array
            # del dp, dm, d1_c
            # blm = geom.adjoint_synthesis(dlens_r, 2, lmaxb, mmaxb, sht_tr)[1]
            # return blm

            get_alm = lambda a: obj[1] if a == 'e' else np.zeros_like(obj[1])
            geom, sht_tr = self.ffi.geom, self.ffi.sht_tr
            d1_c = np.empty((geom.npix(),), dtype=obj[1].dtype)
            d1_r = d1_c.view(rtype[d1_c.dtype]).reshape((d1_c.size, 2)).T  # real view onto complex array
            self.ffi.geom.synthesis(self.ffi.dlm, 1, self.LM_max[0], self.LM_max[1], sht_tr, map=d1_r, mode='GRAD_ONLY')

            dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lm_max_in[0])])(get_alm, geom, sht_tr)
            dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lm_max_in[0])])(get_alm, geom, sht_tr)
            dlens_c = -0.5 * ((d1_c.conj()) * dp + d1_c * dm)
            dlens_r = dlens_c.view(rtype[dlens_c.dtype]).reshape((dlens_c.size, 2)).T  # real view onto complex array
            del dp, dm, d1_c
            eblm = self.ffi.geom.adjoint_synthesis(dlens_r, 2, 500, 500, sht_tr)
            tlm = np.zeros_like(eblm[0])
            return np.array([tlm, *eblm])
        else:
            if adjoint and backwards:
                tlm = np.atleast_2d(self.ffi.lensgclm(obj[0], self.lm_max_in[1], 0, *self.lm_max_out, backwards=backwards, out_sht_mode='STANDARD')) if self.data_key in ['tt', 'tp'] else np.zeros(shape=(Alm.getsize(*self.lm_max_out)),dtype=complex)
                out_sht_mode = out_sht_mode or 'GRAD_ONLY'
                nomagn = nomagn or False
                shaptefirstdim = 1 if out_sht_mode == 'GRAD_ONLY' else 2
                eblm = np.atleast_2d(self.ffi.lensgclm(np.atleast_2d(obj[1:]), self.lm_max_in[1], 2, *self.lm_max_out, backwards=backwards, out_sht_mode=out_sht_mode, nomagn=nomagn)) if self.data_key in ['p', 'ee', 'eb', 'bb', 'tp'] else np.zeros(shape=(shaptefirstdim, Alm.getsize(*self.lm_max_out)),dtype=complex)
                return np.array([tlm.squeeze(), *eblm, np.zeros_like(tlm.squeeze())]) if out_sht_mode == 'GRAD_ONLY' else np.array([tlm.squeeze(), *eblm])
            else:
                if out == 'map':
                    tmap = self.ffi.gclm2lenmap(np.atleast_2d(obj[0]), lmax, spin, False) if self.data_key in ['tt', 'tp'] else np.zeros(shape=(2,self.ffi.geom.npix()))
                    ebmap = self.ffi.gclm2lenmap(np.atleast_2d(obj[1:]), lmax, spin, False) if self.data_key in ['p', 'ee', 'eb', 'bb', 'tp'] else np.zeros(shape=(2,self.ffi.geom.npix()))
                    return tmap+ebmap
               
                elif out == 'alm':
                    if lmax == self.lm_max_in[0]:
                        tlm = self.ffi.lensgclm(np.atleast_2d(obj[0]), self.lm_max_in[1], 0, *self.lm_max_out) if self.data_key in ['tt', 'tp'] else np.zeros(shape=(Alm.getsize(*self.lm_max_out)),dtype=complex)
                        eblm = self.ffi.lensgclm(np.atleast_2d(obj[1:]), self.lm_max_in[1], 2, *self.lm_max_out)  if self.data_key in ['p', 'ee', 'eb', 'bb', 'tp'] else np.zeros(shape=(2,Alm.getsize(*self.lm_max_out)),dtype=complex)
                        return np.array([tlm, *eblm])
                    else:
                        tlm = self.ffi.lensgclm(np.atleast_2d(obj[0]), self.lm_max_out[1], 0, *self.lm_max_in) if self.data_key in ['tt', 'tp'] else np.zeros(shape=(Alm.getsize(*self.lm_max_in)),dtype=complex)
                        eblm = self.ffi.lensgclm(np.atleast_2d(obj[1:]), self.lm_max_out[1], 2, *self.lm_max_in)  if self.data_key in ['p', 'ee', 'eb', 'bb', 'tp'] else np.zeros(shape=(2,Alm.getsize(*self.lm_max_in)),dtype=complex)
                        return np.array([tlm, *eblm])


    def set_field(self, fieldlm):
        if isinstance(fieldlm, list):
            if len(fieldlm) == 1:
                d = [fieldlm[0], None] if self.component[0] == 'p' else [np.zeros_like(fieldlm[0], dtype=complex), fieldlm[0]]
            elif len(fieldlm) == 2:
                d = fieldlm
        elif fieldlm.shape[0] == 1:
            d = [fieldlm[0], None] if self.component[0] == 'p' else [np.zeros_like(fieldlm[0], dtype=complex), fieldlm[0]]
        else:
            d = fieldlm
        # TODO fix hardcoded epsilon
        self.ffi = deflection(self.lenjob_geomlib, d[0], self.LM_max[1], dclm=d[1], numthreads=self.sht_tr, verbosity=False, epsilon=1e-10)


    def get_field(self):
        return [self.ffi.dlm, self.ffi.dclm]


    def klm2dlm(self, klm):
        h2d = cli(0.5 * np.sqrt(np.arange(self.LM_max[0] + 1, dtype=float) * np.arange(1, self.LM_max[0] + 2, dtype=float)))
        Lmax = Alm.getlmax(klm.size, None)
        return almxfl(klm, h2d, Lmax, False)


class Birefringence(Operator):
    def __init__(self, operator_desc):
        super().__init__(operator_desc["libdir"])
        self.ID = 'birefringence'
        self.LM_max = operator_desc["LM_max"]
        self.lm_max = operator_desc["lm_max"]
        self.lm_max_out = operator_desc["lm_max"]
        self.component = operator_desc["component"]
        self.field = {component: None for component in self.component}
        self.field_fns = field.get_secondary_fns(self.component)

        self.perturbative = operator_desc["perturbative"]
        self.sht_tr = operator_desc["sht_tr"]

    @log_on_start(logging.DEBUG, "birefringence", logger=log)
    # @log_on_end(logging.DEBUG, "birefringence done", logger=log)
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None):
        assert obj.shape[0] == 3, "obj must have 3 components"
        lmax = Alm.getlmax(obj[0].size, None)
        Q, U = self.lenjob_geomlib.alm2map_spin(obj[1:], 2, lmax, lmax, self.sht_tr)
        if self.perturbative:
            if adjoint:
                Q_rot = Q + self.angle * U
                U_rot = U - self.angle * Q
            else:
                Q_rot = Q - self.angle * U
                U_rot = U + self.angle * Q
        else:
            if adjoint:
                Q_rot = self.cos_a * Q + self.sin_a * U
                U_rot = self.cos_a * U - self.sin_a * Q
            else:
                Q_rot = self.cos_a * Q - self.sin_a * U
                U_rot = self.cos_a * U + self.sin_a * Q

        Elm_rot, Blm_rot = self.lenjob_geomlib.map2alm_spin(np.array([Q_rot, U_rot]), 2, lmax, lmax, self.sht_tr)
        if out_sht_mode == 'GRAD_ONLY':
            return np.atleast_2d(Elm_rot)
        return np.array([obj[0], Elm_rot, Blm_rot])


    def set_field(self, fieldlm):
        self.angle = 2 * self.lenjob_geomlib.alm2map(fieldlm.squeeze(), *self.LM_max, self.sht_tr)
        self.cos_a, self.sin_a = np.cos(self.angle), np.sin(self.angle)
        self.field = fieldlm


    def get_field(self):
        return self.field


class SpinRaise:
    def __init__(self, lm_max):
        self.ID = 'spin_raise'
        self.lm_max = lm_max


    @log_on_start(logging.DEBUG, "spin_raise", logger=log)
    # @log_on_end(logging.DEBUG, "spin_raise done", logger=log)
    def act(self, obj, spin=None, adjoint=False):
        # This is the property d _sY = -np.sqrt((l+s+1)(l-s+1)) _(s+1)Y
        assert adjoint == False, "adjoint not implemented"
        if spin == 1:
            fl = -np.sqrt(np.arange(self.lm_max[0] + 1) * np.arange(1, self.lm_max[0] + 2))
            almxfl(obj[0], fl, self.lm_max[1], True)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, self.lm_max[0] + i1 + 1, dtype=float) * np.arange(i2, self.lm_max[0] + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        almxfl(obj[1], fl, self.lm_max[1], True)


    def adjoint(self, obj, spin=None):
        assert 0, "implement if needed"
        return self.act(obj, adjoint=True, spin=spin)


class Beam:
    def __init__(self, operator_desc):
        self.ID = 'beam'
        self.data_key = operator_desc['data_key']
        self.lm_max = operator_desc['lm_max']
        self.transferfunction = [
            operator_desc['transferfunction']['t'][:self.lm_max[0]+1] if self.data_key in ['tp', 'tt'] else np.zeros(shape=self.lm_max[0]+1),
            operator_desc['transferfunction']['e'][:self.lm_max[0]+1] if self.data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=self.lm_max[0]+1),
            operator_desc['transferfunction']['b'][:self.lm_max[0]+1] if self.data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=self.lm_max[0]+1),]
        self.tebl2idx = {'t':0, 'e': 1, 'b': 2}
        self.idx2tebl = {v: k for k, v in self.tebl2idx.items()}
        self.is_adjoint = False


    @log_on_start(logging.DEBUG, "beam", logger=log)
    # @log_on_end(logging.DEBUG, "beam done", logger=log)
    def act(self, obj, adjoint=False, factor_p=1):
        assert len(obj) == 3, "obj must have 3 components"
        factor = lambda oi: factor_p if oi > 0 else 1.
        ellmax_ = Alm.getlmax(np.max([len(o) for o in obj]), None)
        if ellmax_ > self.lm_max[0]:
            log.warning(f"Beam operator: ellmax of input {ellmax_} is larger than lm_max of operator {self.lm_max[0]}. Extending transfer function to ellmax {ellmax_}.")
        trsf_ = [_extend_cl(self.transferfunction[oi], ellmax_) for oi in range(3)] if ellmax_ > self.lm_max[0] else self.transferfunction
        trsf_ = [cli(v) for v in trsf_] if adjoint else trsf_
        val = np.array([almxfl(o, trsf_[oi]*factor(oi), len(trsf_[oi])-1, False) for oi, o in enumerate(obj)])
        return val



    def adjoint(self):
        self.is_adjoint = True
        return self


    def __mul__(self, obj, other):
        return self.act(obj)
    

class InverseNoiseVariance(Operator):
    def __init__(self, nlev, lm_max, niv_desc, geom_lib, geominfo, transferfunction, libdir, sht_tr, spectrum_type=None, OBD=None, obd_rescale=None, obd_libdir=None, filtering_type=None, data_key=None):
        super().__init__(libdir)
        self.ID = 'inoise'
        self.data_key = data_key
        self.geom_lib = geom_lib
        self.geominfo = geominfo
        self.nlev = nlev
        self.colored_noise = isinstance(nlev['T'], np.ndarray) or isinstance(nlev['P'], np.ndarray)
        self.lm_max = lm_max
        nivkeys_sorted = ['t', 'e', 'b']
        self.niv = [read_map(niv_desc[key]) for key in nivkeys_sorted] # NOTE niv is always TT, QQ, UU
        self.transferfunction = transferfunction
        spectrum_type = spectrum_type
        OBD = OBD

        self.sht_tr = sht_tr
        self.filtering_type = filtering_type
        self.n1tebl = [
            cli(_extend_cl(self.nlev['T']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['tp', 'tt'] else np.zeros(shape=lm_max[0]+1),
            1.0*cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=lm_max[0]+1),
            1.0*cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=lm_max[0]+1)]
        self.template = None

    @log_on_start(logging.DEBUG, "InverseNoiseVariance", logger=log)
    # @log_on_end(logging.DEBUG, "InverseNoiseVariance done", logger=log)
    def act(self, obj, adjoint=False):
        # TODO "operatorise" this function. If OBD activated, and spectrum_type is non-white, more opertations are needed in here
        if obj.dtype in (np.complex64, np.complex128): # NOTE this is full sky isotropic run (we run things on alms)
            if adjoint:
                return np.array([cli(almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False)) for oi, o in enumerate(obj)])
            return np.array([almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False) for oi, o in enumerate(obj)])
        else:
            obj[0] *= self.niv[0]
            obj[1:] *= self.niv[1]
            
            if False: # TODO if noise inverse variance maps are TT,QQ,UU,QU, need to catch it here
                assert 0, "implement if needed"
                assert self.template is None
                qmap, umap = obj[1], obj[2]
                qmap_copy = qmap.copy()
                qmap *= self.niv[1]
                qmap += self.niv[2] * umap
                umap *= self.niv[2]
                umap += self.niv[1] * qmap_copy
                del qmap_copy

            tlm = self.geom_lib.adjoint_synthesis(obj[0], 0, *self.lm_max, self.sht_tr, apply_weights=False)
            eblm = self.geom_lib.adjoint_synthesis(obj[1:], 2, *self.lm_max, self.sht_tr, apply_weights=False)
            return np.array([*tlm, *eblm])
        

    def apply_combined(self, obj, adjoint=False):
        """
        Apply approx N^{-1} using sqrt-weighted harmonic filtering:
        out = W^{1/2} y^{-1} (1/N_ell) y W^{1/2} qumap
        """
        if obj.dtype in (np.complex64, np.complex128): # NOTE this is full sky isotropic run (we run things on alms)
            if adjoint:
                return np.array([cli(almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False)) for oi, o in enumerate(obj)])
            return np.array([almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False) for oi, o in enumerate(obj)])
        else: # NOTE this is anisotropic run (we run things on maps)
            if self.colored_noise is False:
                tlm = self.geom_lib.adjoint_synthesis(obj[0], 0, *self.lm_max, self.sht_tr, apply_weights=False)
                eblm = self.geom_lib.adjoint_synthesis(obj[1:], 2, *self.lm_max, self.sht_tr, apply_weights=False)

            elif self.colored_noise:
                obj[0] *= np.sqrt(self.niv[0])
                obj[1:] *= np.sqrt(self.niv[1])
                # spin-2 SHT -> multiply alms by invN -> inverse SHT
                almT_f = self.almxfl(tlm, (1.0 / (np.pi / (180.0 * 60.0) * self.nlev['T']))**2)
                almE_f = self.almxfl(eblm[0], (1.0 / (np.pi / (180.0 * 60.0) * self.nlev['P']))**2)
                almB_f = self.almxfl(eblm[1], (1.0 / (np.pi / (180.0 * 60.0) * self.nlev['P']))**2)

                obj[0] = self.geom_lib.synthesis(almT_f, 0, *self.lm_max, self.sht_tr, apply_weights=False)
                obj[1:] = self.geom_lib.synthesis(np.array([almE_f, almB_f]), 2, *self.lm_max, self.sht_tr, apply_weights=False)
                # post-weight
                obj[0] *= np.sqrt(self.niv[0])
                obj[1:] *= np.sqrt(self.niv[1])

                tlm = self.geom_lib.adjoint_synthesis(obj[0], 0, *self.lm_max, self.sht_tr, apply_weights=False)
                eblm = self.geom_lib.adjoint_synthesis(obj[1:], 2, *self.lm_max, self.sht_tr, apply_weights=False)
            return np.array([*tlm, *eblm])


    def adjoint(self):
        self.is_adjoint = True
        return self


    def get_ftebl(self, transferfunction):
        if self.filtering_type == 'isotropic':
            ret_t = _extend_cl(transferfunction[0]**2, len(self.n1tebl[0])-1) * self.n1tebl[0]
            ret_e = _extend_cl(transferfunction[1]**2, len(self.n1tebl[1])-1) * self.n1tebl[1]
            ret_b = _extend_cl(transferfunction[2]**2, len(self.n1tebl[2])-1) * self.n1tebl[2]
            return [ret_t, ret_e, ret_b]

        nlev_ftl = 10800. / np.sqrt(np.sum(read_map(self.niv[0])) / (4.0 * np.pi)) / np.pi
        # NOTE analog to main branch, I only take niv[1] here assuming QQ = UU. Otherwise I need to take into account QU cross as well
        nlev_febl = 10800. / np.sqrt((np.sum(read_map(self.niv[1]))) / (4.0 * np.pi)) / np.pi
        # nlev_febl = 10800. / np.sqrt((0.5 * np.sum(read_map(self.niv[1])) + 0.5 * np.sum(read_map(self.niv[2]))) / (4.0 * np.pi)) / np.pi
        log.debug('Using nlevp %.2f amin'%nlev_febl)
        niv_cl_t = transferfunction[0] ** 2 / (nlev_ftl/ 180. / 60. * np.pi) ** 2
        niv_cl_e = transferfunction[1] ** 2 / (nlev_febl/ 180. / 60. * np.pi) ** 2
        niv_cl_b = transferfunction[2] ** 2 / (nlev_febl/ 180. / 60. * np.pi) ** 2
        return [niv_cl_t, niv_cl_e , niv_cl_b]
    

class Add:
    def __init__(self, operator_desc):
        # super().__init__(operator_desc)
        self.ID = 'add'
    

    @log_on_start(logging.DEBUG, "add: {obj1.shape}", logger=log)
    # @log_on_end(logging.DEBUG, "add done", logger=log)
    def apply(self, obj1, obj2):
        if obj2.dtype in (np.complex64, np.complex128):
            obj1 += obj2
        elif obj2.dtype in (np.float32, np.float64):
            obj1 = self.data_geomlib.synthesis(obj1, 2, *self.lm_max, self.sht_tr, apply_weights=False)
            obj1 += obj2
        return obj1
            

    @log_on_start(logging.DEBUG, "add adjoint: {obj.shape}", logger=log)
    @log_on_end(logging.DEBUG, "add adjoint done", logger=log)
    def apply_adjoint(self, obj):
        assert 0, 'not implemented'