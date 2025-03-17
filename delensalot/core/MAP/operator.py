import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.core import cachers
from delensalot.core.MAP import field

from delensalot.utils import cli, read_map
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


    def set_field(self, it, component=None):
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
    

    def set_field(self, it, component=None):
        pass


class Compound:
    def __init__(self, operators, out, sht_tr):
        
        self.operators = operators
        self.space_out = out
        self.sht_tr = sht_tr
    

    @log_on_start(logging.DEBUG, "joint", logger=log)  
    @log_on_end(logging.DEBUG, "joint done", logger=log)  
    def act(self, obj, spin):
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
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None, secondary=None, out='alm'):
        secondary = secondary or [op.ID for op in self.operators]
        operators = self.operators if not adjoint else self.operators[::-1]
        for idx, operator in enumerate(operators):
            if operator.ID in secondary:
                if isinstance(operator, Lensing):
                    obj = operator.act(obj, spin=spin, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode, out=out)
                else:
                    obj = operator.act(obj, adjoint=adjoint, backwards=adjoint, out_sht_mode=out_sht_mode)
        return obj


    def set_field(self, field):
        for operator in self.operators:
            operator.set_field(field[operator.ID])

    
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
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None, out='alm'):
        lmax = Alm.getlmax(np.max([len(o) for o in obj]), None)
        if self.perturbative: # Applies perturbative remapping
            assert 0, "implement if needed" 
        else:
            if adjoint and backwards:
                tlm = np.atleast_2d(self.ffi.lensgclm(obj[0], self.lm_max_in[1], 0, *self.lm_max_out, backwards=backwards, out_sht_mode='STANDARD')) if self.data_key in ['tt', 'tp'] else np.zeros(shape=(Alm.getsize(*self.lm_max_out)),dtype=complex)
                eblm = np.atleast_2d(self.ffi.lensgclm(np.atleast_2d(obj[1:]), self.lm_max_in[1], 2, *self.lm_max_out, backwards=backwards, out_sht_mode="GRAD_ONLY")) if self.data_key in ['p', 'ee', 'eb', 'bb', 'tp'] else np.zeros(shape=(1, Alm.getsize(*self.lm_max_out)),dtype=complex)
                return np.array([tlm.squeeze(), *eblm, np.zeros_like(tlm.squeeze())])
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
        if fieldlm.shape[0] == 1:
            d = [fieldlm[0], None] if self.component[0] == 'p' else [np.zeros_like(fieldlm[0], dtype=complex), fieldlm[0]]
        else:
            d = fieldlm
        self.ffi = deflection(self.lenjob_geomlib, d[0], self.LM_max[1], dclm=d[1], numthreads=self.sht_tr, verbosity=False, epsilon=1e-10)


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

        self.sht_tr = operator_desc["sht_tr"]

    @log_on_start(logging.DEBUG, "birefringence", logger=log)
    # @log_on_end(logging.DEBUG, "birefringence done", logger=log)
    def act(self, obj, spin=None, adjoint=False, backwards=False, out_sht_mode=None):
        assert obj.shape[0] == 3, "obj must have 3 components"
        lmax = Alm.getlmax(obj[0].size, None)

        # NOTE if no B component, I set B to zero
        # if obj.shape[0] == 1:
        #     obj = [obj[0], np.zeros_like(obj[0])+np.zeros_like(obj[0])*1j] 
        Q, U = self.lenjob_geomlib.alm2map_spin(obj[1:], 2, lmax, lmax, self.sht_tr)

        Q_rot = self.cos_a * Q - self.sin_a * U
        U_rot = self.sin_a * Q + self.cos_a * U

        if adjoint:
            Q_rot, U_rot = self.cos_a * Q + self.sin_a * U, -self.sin_a * Q + self.cos_a * U

        Elm_rot, Blm_rot = self.lenjob_geomlib.map2alm_spin(np.array([Q_rot, U_rot]), 2, lmax, lmax, self.sht_tr)

        if out_sht_mode == 'GRAD_ONLY':
            return np.atleast_2d(Elm_rot)
        return np.array([obj[0], Elm_rot, Blm_rot])


    def set_field(self, fieldlm):
        self.angle = 2 * self.lenjob_geomlib.alm2map(fieldlm.squeeze(), *self.LM_max, self.sht_tr)
        self.cos_a, self.sin_a = np.cos(self.angle), np.sin(self.angle)


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
    

    def set_field(self, idx, it, component=None):
        pass
    

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
    def act(self, obj, adjoint=False):
        assert len(obj) == 3, "obj must have 3 components"
        val = np.array([almxfl(o, self.transferfunction[oi], len(self.transferfunction[oi])-1, False) for oi, o in enumerate(obj)])
        return cli(val) if adjoint else val


    def adjoint(self):
        self.is_adjoint = True
        return self


    def __mul__(self, obj, other):
        return self.act(obj)
    

class InverseNoiseVariance(Operator):
    def __init__(self, nlev, lm_max, niv_desc, geom_lib, geominfo, transferfunction, libdir, spectrum_type=None, OBD=None, obd_rescale=None, obd_libdir=None, sky_coverage=None, filtering_spatial_type=None, data_key=None, sht_tr=None):
        super().__init__(libdir)
        self.ID = 'inoise'
        self.data_key = data_key
        self.geom_lib = geom_lib
        self.geominfo = geominfo
        self.nlev = nlev
        self.lm_max = lm_max
        nivkeys_sorted = ['t', 'e', 'b']
        self.niv = [read_map(niv_desc[key]) for key in nivkeys_sorted]
        self.transferfunction = transferfunction
        spectrum_type = spectrum_type
        OBD = OBD

        self.sht_tr = sht_tr
        self.sky_coverage = sky_coverage
        self.n1tebl = [
            cli(_extend_cl(self.nlev['T']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['tp', 'tt'] else np.zeros(shape=lm_max[0]+1),
            0.5*cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=lm_max[0]+1),
            0.5*cli(_extend_cl(self.nlev['P']**2, lm_max[0])) * (180 * 60 / np.pi) ** 2 if data_key in ['p', 'ee', 'eb', 'tp'] else np.zeros(shape=lm_max[0]+1)]
        self.template = None

    @log_on_start(logging.DEBUG, "InverseNoiseVariance", logger=log)
    # @log_on_end(logging.DEBUG, "InverseNoiseVariance done", logger=log)
    def act(self, obj, adjoint=False):
        if adjoint:
            return np.array([cli(almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False)) for oi, o in enumerate(obj)])
        return np.array([almxfl(o, self.n1tebl[oi], len(self.n1tebl[oi])-1, False) for oi, o in enumerate(obj)])


    def adjoint(self):
        self.is_adjoint = True
        return self
    

    def apply_map(self, tqumap):
        # NOTE niv order is assumed TT,QQ,UU,QU
        """Applies pixel inverse-noise variance maps
        """
        assert len(tqumap) == 3

        tqumap[0] *= self.niv[0]
        assert self.template is None
        qmap, umap = tqumap[1], tqumap[2]
        qmap_copy = qmap.copy()
        qmap *= self.niv[1]
        qmap += self.niv[-1] * umap
        umap *= self.niv[2]
        umap += self.niv[-1] * qmap_copy
        del qmap_copy

        tlm = self.geom_lib.adjoint_synthesis(tqumap[0], 0, *self.lm_max, self.sht_tr, apply_weights=False)
        eblm = self.geom_lib.adjoint_synthesis([qmap, umap], 2, *self.lm_max, self.sht_tr, apply_weights=False)
        return np.array([*tlm, *eblm])


    def get_ftel(self, transferfunction):
        if self.sky_coverage == 'full':
            ret_t = _extend_cl(transferfunction[0]*2, len(self.n1tebl[0])-1) * self.n1tebl[0]
            ret_e = _extend_cl(transferfunction[1]*2, len(self.n1tebl[1])-1) * self.n1tebl[1]
            ret_b = _extend_cl(transferfunction[2]*2, len(self.n1tebl[2])-1) * self.n1tebl[2]
            return [ret_t, ret_e, ret_b]

        nlev_ftl = 10800. / np.sqrt(np.sum(read_map(self.niv[0])) / (4.0 * np.pi)) / np.pi
        nlev_febl = 10800. / np.sqrt((0.5 * np.sum(read_map(self.niv[1])) + np.sum(read_map(self.niv[2]))) / (4.0 * np.pi)) / np.pi
        log.info('Using nlevp %.2f amin'%nlev_febl)
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