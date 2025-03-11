import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from delensalot.core.MAP import cg, field, operator

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd

CMBfields_sorted = ['tt', 'ee', 'bb']

filterfield_desc = lambda ID, libdir: {
    "ID": ID,
    "libdir": opj(libdir),
    "fns": f"{ID}_idx{{idx}}_{{idx2}}_it{{it}}",}


class Filter_3d:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.sec_operator: operator.Secondary = filter_desc['sec_operator']
        self.beam_operator: operator.Beam = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']
        self.add_operator: operator.Add = filter_desc['add_operator']
        
        self.chain_descr = filter_desc['chain_descr']
        
        self.cls_filt = filter_desc['cls_filt']
        lenclsfilt =  np.array([False for _ in range(len(list(filter_desc['cls_filt'].values())[0]))])

        self.cls_filt_bool = np.array([filter_desc['cls_filt'][key]>0 if key in self.cls_filt else lenclsfilt for keyi, key in enumerate(CMBfields_sorted)])
        self.icls = self.invert_cls_filt(self.cls_filt)
        
        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        lm_max_pri = self.sec_operator.operators[-1].lm_max_out
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            if it>1:
                cg_sol_curr = self.wf_field.get_field(it=it-1)
            else:
                cg_sol_curr = np.zeros(shape=(3,Alm.getsize(*lm_max_pri)),dtype=complex)
                if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                    cg_sol_curr[0:2] = self.wf_field.get_field(it=it-1)
                elif 'tt' in self.cls_filt:
                    cg_sol_curr[0] = self.wf_field.get_field(it=it-1)
                elif 'ee' in self.cls_filt:
                    cg_sol_curr[1] = self.wf_field.get_field(it=it-1)
            teb_prep_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = cg.ConjugateGradient(self.preconditioner_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, teb_prep_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, data):
        # NOTE data can be alms or map
        """cg preoperation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        assert data.shape[0] == 3, len(data)
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
        """ acts on elm, which is a lm_max_pri map
        """
        # NOTE if bb interesting, can be implemented here. Currently, bb is just zero, only shape is kept
        assert tebwflm.shape[0] == 3, len(tebwflm)

        nlm = np.copy(tebwflm)
        teblm = self.sec_operator.act(nlm, adjoint=False, backwards=False) # # NOTE lm_max_pri -> lm_max_sky
        assert len(teblm) == 3, len(teblm)
        teblm = self.beam_operator.act(teblm, adjoint=False)
        assert len(teblm) == 3, len(teblm)

        if self.inv_operator.sky_coverage == 'full':
            teblm = self.inv_operator.act(teblm, adjoint=False)
        else:
            lm_max = self.inv_operator.lm_max
            imap = self.inv_operator.geom_lib.synthesis(teblm[0], 0, *lm_max, self.sht_tr)
            qumap = self.inv_operator.geom_lib.synthesis(teblm[1:], 2, *lm_max, self.sht_tr)
            teblm = self.inv_operator.apply_map(np.array([*imap, *qumap]))

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
        return nlm


    @log_on_start(logging.DEBUG, " ---- preconditioner_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- preconditioner_op", logger=log)  
    def preconditioner_op(self, teblm):
        lmax_ = Alm.getlmax(teblm[1].size, None)

        ninv_ftebl = self.inv_operator.get_ftel(self.beam_operator.transferfunction)
        if np.any(ninv_ftebl[0]) and len(ninv_ftebl[0]) - 1 < lmax_:  # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_ftl = ninv_ftebl[0]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s" % (len(ninv_ftl)-1, lmax_))
            nz = np.where(ninv_ftl > 0)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_ + 1, dtype=float)))
        if np.any(ninv_ftebl[1]) and len(ninv_ftebl[1]) - 1 < lmax_: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_fel = ninv_ftebl[1]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_))
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_+1, dtype=float)))

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            lmax_sky_ = self.cls_filt['tt'].size
            Si = np.empty((lmax_ + 1,2,2), dtype=float)
            Si[:lmax_sky_+1,0,0] = self.icls[:lmax_sky_+1,0,0]
            Si[:lmax_sky_+1,1,1] = self.icls[:lmax_sky_+1,1,1]
            Si[:lmax_sky_+1,0,1] = self.icls[:lmax_sky_+1,0,1]
            Si[:lmax_sky_+1,1,0] = self.icls[:lmax_sky_+1,1,0]
            Si[:,0,0] += ninv_ftl[:lmax_+1]
            Si[:,1,1] += ninv_fel[:lmax_+1]
            tebout = np.empty(shape=(3,teblm[0].size), dtype=complex)
        elif 'tt' in self.cls_filt:
            lmax_sky_ = self.cls_filt['tt'].size
            Si = np.empty((lmax_ + 1,1,1), dtype=float)
            Si[:lmax_sky_+1,0,0] = self.icls[:lmax_sky_+1,0,0]
            Si[:lmax_sky_+1,0,0] += ninv_ftl[:lmax_+1]
            tebout = np.empty(shape=(3,teblm[0].size), dtype=complex)
        elif 'ee' in self.cls_filt:
            lmax_sky_ = self.cls_filt['ee'].size
            Si = np.empty((lmax_ + 1,1,1), dtype=float)
            Si[:lmax_sky_+1,0,0] = self.icls[:lmax_sky_+1,0,0]
            Si[:lmax_sky_+1,0,0] += ninv_fel[:lmax_+1]
            tebout = np.empty(shape=(3,teblm[1].size), dtype=complex)
        flmat = np.linalg.pinv(Si)

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], lmax_, False) + almxfl(teblm[1], flmat[:, 0, 1], lmax_, False)
            tebout[1] = almxfl(teblm[0], flmat[:, 1, 0], lmax_, False) + almxfl(teblm[1], flmat[:, 1, 1], lmax_, False)
        elif 'tt' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], lmax_, False)
        elif 'ee' in self.cls_filt:
            tebout[1] = almxfl(teblm[1], flmat[:, 0, 0], lmax_, False)
        return tebout
    

    @log_on_start(logging.DEBUG, " ---- get_ivfreslm: {it}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- get_ivfreslm", logger=log)
    def get_ivfreslm(self, it, data=None, elm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert elm_wf is not None and data is not None
            ivfreslm = self.sec_operator.act(elm_wf)
            assert ivfreslm.shape[0] == 3, ivfreslm.shape
            ivfreslm = -1*self.beam_operator.act(ivfreslm)
            
            if data[0].dtype in [np.complex64, np.complex128]:
                ivfreslm += data
                ivfreslm = self.inv_operator.act(ivfreslm, adjoint=False)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = []
                ivfresmap.append(self.inv_operator.geom_lib.synthesis(ivfreslm[0], 0, *lm_max, self.sht_tr))
                buff = self.inv_operator.geom_lib.synthesis(ivfreslm[1:], 2, *lm_max, self.sht_tr)
                ivfresmap.append(buff[0])
                ivfresmap.append(buff[1])
                ivfresmap = [ivf+d for ivf,d in zip(ivfresmap,data)]
                ivfreslm = self.inv_operator.apply_map(ivfresmap)

            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'tt' in self.cls_filt:
                ivfreslm[1] = np.zeros_like(ivfreslm[0],dtype=complex)
                ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'ee' in self.cls_filt:
                ivfreslm[0] = np.zeros_like(ivfreslm[1],dtype=complex)
                # ivfreslm[2] = np.zeros_like(ivfreslm[1],dtype=complex)
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
        return np.linalg.pinv(Si)


    def update_operator(self, field):
        self.sec_operator.set_field(field)


    def get_template(self, it, secondary=None, component=None):
        estCMB = self.get_wflm(it=it)

        for operator in self.sec_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.sec_operator.act(estCMB, secondary=secondary)