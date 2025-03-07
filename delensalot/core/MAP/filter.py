import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np

from scipy.interpolate import UnivariateSpline as spl

from delensalot.core.MAP import cg, field, operator, operator_3d
from delensalot.core.MAP.context import ComputationContext

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd
from delensalot.utils import cli

CMBfields_sorted = ['tt', 'ee', 'bb']

filterfield_desc = lambda ID, libdir: {
    "ID": ID,
    "libdir": opj(libdir),
    "fns": f"{ID}_idx{{idx}}_{{idx2}}_it{{it}}",}


class Filter_3d:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.sec_operator: operator_3d.Secondary = filter_desc['sec_operator']
        self.beam_operator: operator_3d.Beam = filter_desc['beam_operator']
        self.inv_operator: operator_3d.InverseNoiseVariance = filter_desc['inv_operator']

        self.add_operator: operator_3d.Add = filter_desc['add_operator']
        
        self.chain_descr = filter_desc['chain_descr']
        
        self.cls_filt = filter_desc['cls_filt']
        self.cls_filt_bool = [None, None, None]
        for keyi, key in enumerate(CMBfields_sorted):
            if key in self.cls_filt:
                self.cls_filt_bool[keyi] = filter_desc['cls_filt'][key]>0.
            else:
                self.cls_filt_bool[keyi] = np.array([False for _ in range(len(list(filter_desc['cls_filt'].values())[0]))])
        # self.cls_filt_bool = [filter_desc['cls_filt'][key]>0. for key in CMBfields_sorted if key in filter_desc['cls_filt']]
        self.icls_filt = self.invert_cls_filt(self.cls_filt)
        
        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            if it>1:
                cg_sol_curr = self.wf_field.get_field(it=it-1)
            else:
                cg_sol_curr = np.zeros(shape=(3,Alm.getsize(4200,4200)),dtype=complex)
                if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                    cg_sol_curr[0:2] = self.wf_field.get_field(it=it-1)
                elif 'tt' in self.cls_filt:
                    cg_sol_curr[0] = self.wf_field.get_field(it=it-1)
                elif 'ee' in self.cls_filt:
                    cg_sol_curr[1] = self.wf_field.get_field(it=it-1)
            print(cg_sol_curr)
            teb_prep_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            print(teb_prep_alm)
            mchain = cg.ConjugateGradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, teb_prep_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, data):
        # NOTE data can be alms or map
        """cg preoperation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        assert len(data) == 3, len(data)
        space = next(('alm' if data[i].dtype in [np.complex64, np.complex128] else 'map') for i in range(3) if np.any(data[i]))

        # TODO can remove this once operator has a single act()
        if space == 'alm':
            teblmc = self.inv_operator.act(data, adjoint=False)
        elif space == 'map':
            teblmc = self.inv_operator.apply_map(data)
        assert len(teblmc) == 3, teblmc.shape
        
        teblmc = self.beam_operator.act(teblmc, adjoint=False)
        assert len(teblmc) == 3, len(teblmc)
        # NOTE spin 0 is standard, spin 2 is GRAD_only. For convenience, I'll make it return a 3 tuple
        teblm = self.sec_operator.act(teblmc, adjoint=True, backwards=True) # NOTE lm_sky -> lm_pri
        assert len(teblm) == 3, len(teblm)

        teblm = almxfl_nd(teblm, self.cls_filt_bool, None, False)
        assert len(teblm) == 3, len(teblm)
        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            teblm[2] = np.zeros_like(teblm[1],dtype=complex)
        elif 'tt' in self.cls_filt:
            teblm[1] = np.zeros_like(teblm[0],dtype=complex)
            teblm[2] = np.zeros_like(teblm[0],dtype=complex)
        elif 'ee' in self.cls_filt:
            teblm[0] = np.zeros_like(teblm[1],dtype=complex)
            teblm[2] = np.zeros_like(teblm[1],dtype=complex)
        return np.array(teblm)


    @log_on_start(logging.DEBUG, " ---- fwd_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op(self, tebwflm):
        tebwflm = [teb for teb in tebwflm]
        # NOTE if bb interesting, can be implemented here. Currently, bb is just zero, only shape is kept
        assert len(tebwflm) == 3, len(tebwflm)
        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            pass
        elif 'tt' not in self.cls_filt:
            tebwflm[0] = np.array([], dtype=complex)
        elif 'ee' not in self.cls_filt:
            tebwflm[1] = np.array([], dtype=complex)
        """ acts on elm, which is a lm_max_pri map
        """
        nlm = [np.copy(wflm) for wflm in tebwflm]
        teblm = self.sec_operator.act(nlm, adjoint=False, backwards=False) # # NOTE lm_max_pri -> lm_max_sky
        assert len(teblm) == 3, len(teblm)
        teblm = self.beam_operator.act(teblm, adjoint=False)
        assert len(teblm) == 3, len(teblm)

        if self.inv_operator.sky_coverage == 'full':
            teblm = self.inv_operator.act(teblm, adjoint=False)
        else:
            lm_max = self.inv_operator.lm_max
            imap = self.inv_operator.geom_lib.synthesis(teblm[0], 0, *lm_max, 6)
            qumap = self.inv_operator.geom_lib.synthesis(teblm[1:], 2, *lm_max, 6)
            teblm = self.inv_operator.apply_map([imap, qumap])

        teblm = self.beam_operator.act(teblm, adjoint=False)
        teblm = self.sec_operator.act(teblm, adjoint=True, backwards=True) # lm_sky -> lm_pri
        nlm = teblm
        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            nlm[0] += almxfl(tebwflm[0], self.icls[:, 0, 0], len(self.cls_filt_bool[0])-1, False)
            nlm[0] += almxfl(tebwflm[1], self.icls[:, 0, 1], len(self.cls_filt_bool[0])-1, False)
            nlm[1] += almxfl(tebwflm[1], self.icls[:, 1, 1], len(self.cls_filt_bool[0])-1, False)
            nlm[1] += almxfl(tebwflm[0], self.icls[:, 1, 0], len(self.cls_filt_bool[0])-1, False)
            nlm[2] = np.zeros_like(nlm[1],dtype=complex)
            almxfl(nlm[0], self.cls_filt['tt'] > 0, len(self.cls_filt_bool[0]), True)
            almxfl(nlm[1], self.cls_filt['ee'] > 0, len(self.cls_filt_bool[0]), True)
        elif 'tt' in self.cls_filt:
            nlm[0] += almxfl(tebwflm[0], self.icls[:, 0, 0], len(self.cls_filt_bool[0])-1, False)
            almxfl(nlm[0], self.cls_filt['tt'] > 0, len(self.cls_filt_bool[0]), True)
            nlm[1] = np.zeros_like(nlm[0],dtype=complex)
            nlm[2] = np.zeros_like(nlm[0],dtype=complex)
        elif 'ee' in self.cls_filt:
            nlm[1] += almxfl(tebwflm[1], self.icls[:, 0, 0], len(self.cls_filt_bool[0])-1, False)
            almxfl(nlm[1], self.cls_filt['ee'] > 0, len(self.cls_filt_bool[0]), True)
            nlm[0] = np.zeros_like(nlm[1],dtype=complex)
            nlm[2] = np.zeros_like(nlm[1],dtype=complex)
        return np.array(nlm)


    @log_on_start(logging.DEBUG, " ---- precon_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- precon_op", logger=log)  
    def precon_op(self, teblm):
        # NOTE solm is  (t,e)
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        # FIXME need to access lmax_sol safely
        lmax_sol = 4200
        mmax_sol = 4200
        ninv_ftebl = self.inv_operator.get_ftel(self.beam_operator.transferfunction)
        if np.any(ninv_ftebl[0]) and len(ninv_ftebl[0]) - 1 < lmax_sol:  # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_ftl = ninv_ftebl[0]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s" % (len(ninv_ftl)-1, lmax_sol))
            assert np.all(ninv_ftl >= 0)
            nz = np.where(ninv_ftl > 0)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        if np.any(ninv_ftebl[1]) and len(ninv_ftebl[1]) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_fel = ninv_ftebl[1]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            self.icls[:,0,0] += ninv_ftl[:lmax_sol + 1]
            self.icls[:,1,1] += ninv_fel[:lmax_sol + 1]
            tebout = np.empty(shape=(3,teblm[0].size), dtype=complex)
        elif 'tt' in self.cls_filt:
            self.icls[:,0,0] += ninv_ftl[:lmax_sol + 1]
            tebout = np.empty(shape=(3,teblm[0].size), dtype=complex)
        elif 'ee' in self.cls_filt:
            self.icls[:,0,0] += ninv_fel[:lmax_sol + 1]
            tebout = np.empty(shape=(3,teblm[1].size), dtype=complex)
        flmat = np.linalg.pinv(self.icls)

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], mmax_sol, False) + almxfl(teblm[1], flmat[:, 0, 1], mmax_sol, False)
            tebout[1] = almxfl(teblm[0], flmat[:, 1, 0], mmax_sol, False) + almxfl(teblm[1], flmat[:, 1, 1], mmax_sol, False)
        elif 'tt' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], mmax_sol, False)
        elif 'ee' in self.cls_filt:
            tebout[1] = almxfl(teblm[1], flmat[:, 0, 0], mmax_sol, False)
        return tebout
    

    @log_on_start(logging.DEBUG, " ---- get_ivfreslm: {it}", logger=log)
    def get_ivfreslm(self, it, data=None, elm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert elm_wf is not None and data is not None
            ivfreslm = self.sec_operator.act(elm_wf)
            assert len(ivfreslm) == 3, ivfreslm.shape
            if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                ivfreslm[2] = np.array([],dtype=complex)
            elif 'tt' in self.cls_filt:
                ivfreslm[1] = np.array([],dtype=complex)
                ivfreslm[2] = np.array([],dtype=complex)
            elif 'ee' in self.cls_filt:
                ivfreslm[0] = np.array([],dtype=complex)
                ivfreslm[2] = np.array([],dtype=complex)
            ivfreslm = [-1*val for val in self.beam_operator.act(ivfreslm)]
            if data[0].dtype in [np.complex64, np.complex128]:
                ivfreslm = [(ivf+d)*0.5 for ivf,d in zip(ivfreslm,data)]
                ivfreslm = self.inv_operator.act(ivfreslm, adjoint=False)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = []
                ivfresmap.append(self.inv_operator.geom_lib.synthesis(ivfreslm[0], 0, *lm_max, 6))
                ivfresmap.extend(*self.inv_operator.geom_lib.synthesis(ivfreslm[1:], 2, *lm_max, 6))
                ivfresmap = self.inv_operator.geom_lib.synthesis(ivfreslm, 2, *lm_max, 6)
                ivfresmap = [ivf+d for ivf,d in zip(ivfreslm,data)]
                ivfreslm = self.inv_operator.apply_map(ivfresmap)

            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'tt' in self.cls_filt:
                ivfreslm[1] = np.zeros_like(ivfreslm[0],dtype=complex)
                ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'ee' in self.cls_filt:
                ivfreslm[0] = np.zeros_like(ivfreslm[1],dtype=complex)
                ivfreslm[2] = np.zeros_like(ivfreslm[1],dtype=complex)
            self.ivf_field.cache(ivfreslm, it=it)
        return self.ivf_field.get_field(it=it)


    def invert_cls_filt(self, cls_filt):
        if 'tt' in cls_filt and 'ee' in cls_filt:
            Si = np.empty((cls_filt['tt'].size, 2, 2), dtype=float)
            Si[:, 0, 0] = cls_filt['tt']
            Si[:, 0, 1] = cls_filt['te']
            Si[:, 1, 0] = cls_filt['te']
            Si[:, 1, 1] = cls_filt['ee']
        elif 'tt' in cls_filt:
            Si = np.empty((cls_filt['tt'].size, 1, 1), dtype=float)
            Si[:, 0, 0] = cls_filt['tt']
        elif 'ee' in cls_filt:
            Si = np.empty((cls_filt['ee'].size, 1, 1), dtype=float)
            Si[:, 0, 0] = cls_filt['ee']
        self.icls = np.linalg.pinv(Si)


    def update_operator(self, field):
        self.sec_operator.set_field(field)


    def get_template(self, it, secondary=None, component=None):
        estCMB = self.get_wflm(it=it)

        for operator in self.sec_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.sec_operator.act(estCMB, secondary=secondary)  


class Filter:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.sec_operator: operator.Secondary = filter_desc['sec_operator']
        self.beam_operator: operator.Beam = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']

        self.add_operator: operator.Add = filter_desc['add_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(it=it-1) #* (0+1j*0)
            tpn_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = cg.ConjugateGradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep: {qumap.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, qumap):
        """cg preoperation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        if qumap.dtype in [np.complex64, np.complex128]:
            eblmc = self.inv_operator.act(qumap, adjoint=False)
            assert eblmc.shape[0] == 2, eblmc.shape
        else:
            eblmc = self.inv_operator.apply_map(qumap)
            assert eblmc.shape[0] == 2, eblmc.shape
        eblmc = self.beam_operator.act(eblmc, adjoint=False)
        elm = self.sec_operator.act(eblmc, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY') # NOTE lm_sky -> lm_pri
        assert elm.shape[0] == 1, elm.shape
        elm = almxfl_nd(elm, self.cls_filt['ee'] > 0., None, False)
        return elm[0] if elm.ndim == 2 else elm
    


    @log_on_start(logging.DEBUG, " ---- fwd_op: {ewflm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op(self, ewflm):
        """ acts on elm, which is a lm_max_pri map
        """
        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(ewflm)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = nlm.reshape((1, nlm.size))
        eblm = self.sec_operator.act(elm_2d, spin=2, adjoint=False, backwards=False) # # NOTE lm_max_pri -> lm_max_sky
        assert eblm.shape[0] == 2, eblm.shape
        eblm = self.beam_operator.act(eblm, adjoint=False)
        assert eblm.shape[0] == 2, eblm.shape

        # FIXME this confuses me, who turns eblm into maps when masked?
        if eblm.dtype in [np.complex64, np.complex128]:
            eblm = self.inv_operator.act(eblm, adjoint=False)
        else:
            lm_max = self.inv_operator.lm_max
            qumap = self.inv_operator.geom_lib.synthesis(eblm, 2, *lm_max, 6)
            eblm = self.inv_operator.apply_map(qumap)
        
        eblm = self.beam_operator.act(eblm, adjoint=False)
        elm_2d = np.atleast_2d(self.sec_operator.act(eblm, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')) # lm_sky -> lm_pri
        nlm = elm_2d.squeeze()
        nlm += almxfl(ewflm, iclee * (iclee > 0.0), Alm.getlmax(ewflm.size, None), False)
        return nlm


    @log_on_start(logging.DEBUG, " ---- precon_op: {elm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- precon_op", logger=log)  
    def precon_op(self, elm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(elm.size, None), (Alm.getlmax(elm[0].size, None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel, ninv_fbl = self.inv_operator.get_febl(self.beam_operator.transferfunction)
        if len(ninv_fel) - 1 < Alm.getlmax(elm.size, None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(elm.size, None) + 1, dtype=float)))

        lmax = Alm.getlmax(elm.size, None)
        flmat = cli(self.cls_filt['ee'][:lmax+1]) + ninv_fel[:]
        flmat = cli(flmat) * (self.cls_filt['ee'][:lmax+1] > 0.)
        return almxfl(elm, flmat, lmax, False)
    

    @log_on_start(logging.DEBUG, " ---- get_ivfreslm: {it}", logger=log)
    def get_ivfreslm(self, it, data=None, elm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert elm_wf is not None and data is not None
            ivfreslm = self.sec_operator.act(elm_wf, spin=2)
            assert ivfreslm.shape[0] == 2, ivfreslm.shape
            ivfreslm = -1*self.beam_operator.act(ivfreslm)

            if data.dtype in [np.complex64, np.complex128]:
                ivfreslm += data
                ivfreslm = self.inv_operator.act(0.5*ivfreslm, adjoint=False)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = self.inv_operator.geom_lib.synthesis(ivfreslm, 2, *lm_max, 6)
                ivfresmap += data
                ivfreslm = self.inv_operator.apply_map(ivfresmap)

            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            self.ivf_field.cache(ivfreslm, it=it)
        return self.ivf_field.get_field(it=it)


    def update_operator(self, field):
        self.sec_operator.set_field(field)


    def get_template(self, it, secondary=None, component=None):
        estCMB = self.get_wflm(it=it)

        for operator in self.sec_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.sec_operator.act(estCMB, secondary=secondary)
    


class IVF_v2:
    def __init__(self, filter_desc): 
        self.libdir = filter_desc['libdir']
        self.ivf_operator = filter_desc['ivf_operator']
        self.beam_operator = filter_desc['beam_operator']
        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']
        self.add_operator: operator.Add = filter_desc['add_operator']


    def get_ivfreslm(self, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert eblm_wf is not None and data is not None
            ivfreslm = self.ivf_operator.apply_inplace(eblm_wf)
            ivfreslm = -1*self.beam_operator.apply_inplace(ivfreslm)

            if data.dtype in [np.complex64, np.complex128]:
                ivfreslm += data
                ivfreslm = self.inv_operator.apply(0.5*ivfreslm)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = self.inv_operator.data_geomlib.synthesis(ivfreslm, 2, *lm_max, 6)
                ivfresmap += data
                ivfreslm = self.inv_operator.apply(ivfresmap)

            ivfreslm = self.beam_operator.apply_inplace(ivfreslm)
            self.ivf_field.cache(ivfreslm, it=it)
        return self.ivf_field.get_field(it=it)


    def update_operator(self, field):
        self.ivf_operator.set_field(field)


class WF_v2:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.wf_operator: operator.Secondary = filter_desc['wf_operator']
        self.beam_operator: operator.Beam = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(it=it-1)# * (0+1j*0)
            tpn_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = cg.ConjugateGradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep: {data_sky.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, data_sky): # sky -> pri
        """cg-inversion pre-operation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        # NOTE if data_sky is complex, I can continue with data_sky, but not doing this at this time
        eblm_sky = self.inv_operator.apply(data_sky)
        self.beam_operator.apply_inplace(eblm_sky) #  eblm_sky or data_sky                             
        elm_pri = self.wf_operator.apply_adjoint_inplace(eblm_sky) # eblm_sky or data_sky NOTE lm_sky -> lm_pri
        almxfl(elm_pri, self.cls_filt['ee'] > 0., None, True)
        return elm_pri
    

    @log_on_start(logging.DEBUG, " ---- fwd_op: {ewflm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op(self, ewflm): # pri -> pri
        """ acts on elm, which is a lm_max_pri map
        """
        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(ewflm).reshape((1, ewflm.size))
        nlm = self.wf_operator.apply_inplace(nlm) # NOTE                lm_pri -> lm_pri    alm -> alm  alm -> alm
        elm_sky = self.beam_operator.apply_inplace(nlm) #               lm_pri -> lm_sky    alm -> alm  alm -> alm
        self.inv_operator.apply(elm_sky) #                              lm_sky -> lm_sky    alm -> alm  alm -> alm
        self.beam_operator.apply_inplace(elm_sky) #                     lm_sky -> lm_sky    alm -> alm  alm -> alm

        elm_pri = self.wf_operator.apply_adjoint_inplace(elm_sky) #     lm_sky -> lm_pri
        ret = elm_pri + almxfl(ewflm, iclee * (iclee > 0.0), Alm.getlmax(ewflm.size, None), False)
        # hp.mollview(hp.alm2map(elm_pri, nside=256))
        # plt.show()
        return ret


    @log_on_start(logging.DEBUG, " ---- precon_op: {elm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- precon_op", logger=log)  
    def precon_op(self, elm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(elm.size, None), (Alm.getlmax(elm[0].size, None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel, ninv_fbl = self.inv_operator.get_febl(self.beam_operator.transferfunction)
        if len(ninv_fel) - 1 < Alm.getlmax(elm.size, None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(elm.size, None) + 1, dtype=float)))

        lmax = Alm.getlmax(elm.size, None)
        flmat = cli(self.cls_filt['ee'][:lmax+1]) + ninv_fel[:]
        flmat = cli(flmat) * (self.cls_filt['ee'][:lmax+1] > 0.)
        return almxfl(elm, flmat, lmax, False)


    def get_template(self, it, secondary, component, lm_max_in=None, lm_max_out=None):
        ctx, _ = ComputationContext()  # Get the singleton instance
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.wf_operator.set_field(idx=idx, it=it, secondary=secondary, component=component, idx2=idx2)
        estCMB = self.get_wflm(it=it)

        for operator in self.wf_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.wf_operator.act(estCMB, secondary=secondary)
    
