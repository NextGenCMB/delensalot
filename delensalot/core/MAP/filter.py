import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from delensalot.core.MAP import cg, field, operator
from delensalot.config.config_manager import get_config
from delensalot.utils import cli

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd

CMBfields_sorted = ['tt', 'ee', 'bb']

filterfield_desc = lambda ID, libdir: {
    "ID": ID,
    "libdir": opj(libdir),
    "fns": f"{ID}_idx{{idx}}_{{idx2}}_it{{it}}",
    # "cacher_type": 'npy' if ID == 'wf' else 'NoCache'
    "cacher_type": 'npy' if ID == 'wf' else 'npy'
}

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1
    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret

def zeroed_copy(d):
    """Return a deep copy of dict with same structure, but all arrays replaced by zeros of same shape."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.zeros_like(v, dtype=complex)
        elif isinstance(v, list):
            out[k] = [np.zeros_like(x, dtype=complex) if isinstance(x, np.ndarray) else x for x in v]
        elif isinstance(v, dict):
            out[k] = zeroed_copy(v)  # recursive
        else:
            out[k] = v  # leave untouched if not array/list
    return out

class Filter_3d:
    def __init__(self, filter_desc):
        config = get_config()
        self.libdir = filter_desc['libdir']
        self.sec_operator: operator.Secondary = filter_desc['sec_operator']
        self.beam_operator: operator.Beam = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']
        self.add_operator: operator.Add = filter_desc['add_operator']
        
        self.filtering_type = filter_desc['filtering_type']
        self.sky_coverage = filter_desc['sky_coverage']
        self.chain_descr = filter_desc['chain_descr']
        
        self.cls_filt = filter_desc['cls_filt']
        lenclsfilt =  np.array([False for _ in range(len(list(filter_desc['cls_filt'].values())[0]))])

        self.cls_filt_bool =np.array([_extend_cl(filter_desc['cls_filt'][key], config.lm_max_pri[0])>0 if key in self.cls_filt else _extend_cl(lenclsfilt, config.lm_max_pri[0]) for keyi, key in enumerate(CMBfields_sorted)])
        self.icls = self.invert_cls_filt(self.cls_filt)
        self.sht_tr = filter_desc['sht_tr']
        
        self.ivfres_field = field.Filter(filterfield_desc('ivfres', self.libdir))
        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))

        self.mchain = cg.ConjugateGradient(self.preconditioner_op, self.chain_descr, self.cls_filt)
        self.nobire = False
        self.nocurl = False
        self.shtmode = 'STANDARD'
        # self.shtmode = 'GRAD_ONLY' # NOTE grad only comes with dangers... if B relevant in intermediate steps, can spoil result..


    def get_wflm(self, it, data=None):
        config = get_config()
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            if it>1:
                cg_sol_curr = self.wf_field.get_field(it=it-1)
            else:
                cg_sol_curr = np.zeros(shape=(3,Alm.getsize(*config.lm_max_pri)),dtype=complex)
                if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                    cg_sol_curr[0:2] = self.wf_field.get_field(it=it-1)
                elif 'tt' in self.cls_filt:
                    cg_sol_curr[0] = self.wf_field.get_field(it=it-1)
                elif 'ee' in self.cls_filt:
                    cg_sol_curr[1] = self.wf_field.get_field(it=it-1)
            if self.filtering_type == 'isotropic':
                if data[0].dtype in [np.float32, np.float64]:
                    delTEB = np.zeros(shape=(3,Alm.getsize(*config.lm_max_sky)),dtype=complex)
                    delTEB[0] = self.inv_operator.geom_lib.adjoint_synthesis(data[0], 0, *config.lm_max_sky, self.sht_tr)[0]
                    delTEB[1:] = self.inv_operator.geom_lib.adjoint_synthesis(data[1:], 2, *config.lm_max_sky, self.sht_tr)
                else:
                    delTEB = data
                delTEB[0] = almxfl(delTEB[0], cli(self.beam_operator.transferfunction[0]), config.lm_max_sky[1], False)
                delTEB[1] = almxfl(delTEB[1], cli(self.beam_operator.transferfunction[1]), config.lm_max_sky[1], False)
                delTEB[2] = almxfl(delTEB[2], cli(self.beam_operator.transferfunction[2]), config.lm_max_sky[1], False)
                delTEB = self.sec_operator.act(delTEB, adjoint=True, backwards=True, out_sht_mode=self.shtmode, nomagn=True)
                almxfl(delTEB[0], _extend_cl(self.beam_operator.transferfunction[0], config.lm_max_pri[1]), config.lm_max_pri[1], True)
                almxfl(delTEB[1], _extend_cl(self.beam_operator.transferfunction[1], config.lm_max_pri[1]), config.lm_max_pri[1], True)
                almxfl(delTEB[2], _extend_cl(self.beam_operator.transferfunction[2], config.lm_max_pri[1]), config.lm_max_pri[1], True)
                field_operator = self.get_field_operator()
                config = get_config()
                zero_field = zeroed_copy(field_operator)
                self.update_operator(zero_field)
                teb_prep_alm = self.calc_prep(delTEB) # NOTE lm_sky -> lm_pri
                self.mchain.solve(cg_sol_curr, teb_prep_alm, self.fwd_op, maxiter=200)
                self.update_operator(field_operator)
            else:
                teb_prep_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
                self.mchain.solve(cg_sol_curr, teb_prep_alm, self.fwd_op, maxiter=200)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, data):
        # NOTE data can be alms or map
        """cg preoperation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}` (or the isotropic version of it)
        """
        assert data.shape[0] == 3, len(data)
        teblmc = self.inv_operator.act(data, adjoint=False)
        assert len(teblmc) == 3, teblmc.shape
        teblmc = self.beam_operator.act(teblmc, adjoint=False)
        assert len(teblmc) == 3, len(teblmc)
        teblm = self.sec_operator.act(teblmc, adjoint=True, backwards=True, nobire=self.nobire, out_sht_mode=self.shtmode,) # NOTE lm_sky -> lm_pri
        assert len(teblm) == 3, len(teblm)

        # teblm = almxfl_nd(teblm, self.cls_filt_bool, None, False)
        # teblm[1] = almxfl_nd(teblm[1], self.cls_filt_bool[1], None, False)
        # teblm[2] = almxfl_nd(teblm[2], self.cls_filt_bool[1], None, False)
        assert len(teblm) == 3, len(teblm)
        # if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
        #     teblm[2] = np.zeros_like(teblm[1], dtype=complex)
        # elif 'tt' in self.cls_filt:
        #     teblm[1] = np.zeros_like(teblm[0], dtype=complex)
        #     teblm[2] = np.zeros_like(teblm[0], dtype=complex)
        # elif 'ee' in self.cls_filt:
        #     pass
        #     teblm[0] = np.zeros_like(teblm[1], dtype=complex)
        #     teblm[2] = np.zeros_like(teblm[1], dtype=complex)
        return np.array(teblm)


    @log_on_start(logging.DEBUG, " ---- fwd_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op(self, tebwflm):
        def proj_E(teblm):
            #NOTE if we don't let B float freely, need to project out B part in fwd operation
            out = teblm.copy()
            out[0] *= 0
            out[2] *= 0
            return out
        """ 
        Pure alm space (full sky) for better readibility
        This is Equation (20) of the CMB-S4 paper
        acts on elm, which is a lm_max_pri map
        """
        # tebwflm = proj_E(tebwflm)

        assert tebwflm.shape[0] == 3, len(tebwflm)
        nlm = np.copy(tebwflm)
        teblm = self.sec_operator.act(nlm, adjoint=False, backwards=False, nobire=self.nobire, out_sht_mode=self.shtmode) # # NOTE lm_max_pri -> lm_max_sky
        assert len(teblm) == 3, len(teblm)
        teblm = self.beam_operator.act(teblm, adjoint=False)
        assert len(teblm) == 3, len(teblm)
        teblm = self.inv_operator.act(teblm, adjoint=False)
        teblm = self.beam_operator.act(teblm, adjoint=False)
        teblm = self.sec_operator.act(teblm, adjoint=True, backwards=True, nobire=self.nobire, out_sht_mode=self.shtmode) # lm_sky -> lm_pri
        nlm = teblm

        if 'ee' in self.cls_filt:
            iclsb = 1e-20*np.ones_like(self.icls[:, 0, 0])
            nlm[1] += almxfl(tebwflm[1], self.icls[:, 0, 0], len(self.cls_filt_bool[0])-1, False)
            nlm[2] += almxfl(tebwflm[2], iclsb, len(self.cls_filt_bool[0])-1, False)
            # almxfl(nlm[1], self.cls_filt['ee'] > 0, len(self.cls_filt_bool[0])-1, True)
        # return proj_E(nlm)
        return nlm


    @log_on_start(logging.DEBUG, " ---- fwd_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op_(self, tebwflm):
        """ This is Equation (20) of the CMB-S4 paper
        acts on elm, which is a lm_max_pri map
        """
        # NOTE if bb interesting, can be implemented here. Currently, bb is just zero, only shape is kept
        assert tebwflm.shape[0] == 3, len(tebwflm)
        nlm = np.copy(tebwflm)
        teblm = self.sec_operator.act(nlm, adjoint=False, backwards=False, nobire=self.nobire, out_sht_mode=self.shtmode) # NOTE lm_max_pri -> lm_max_sky
        assert len(teblm) == 3, len(teblm)
        teblm = self.beam_operator.act(teblm, adjoint=False)
        assert len(teblm) == 3, len(teblm)

        if self.sky_coverage == 'full':
            teblm = self.inv_operator.act(teblm, adjoint=False)
        else:
            lm_max = self.inv_operator.lm_max
            imap = self.inv_operator.geom_lib.synthesis(teblm[0], 0, *lm_max, self.sht_tr)
            qumap = self.inv_operator.geom_lib.synthesis(teblm[1:], 2, *lm_max, self.sht_tr)
            teblm = self.inv_operator.act(np.array([*imap, *qumap]))

        teblm = self.beam_operator.act(teblm, adjoint=False)
        teblm = self.sec_operator.act(teblm, adjoint=True, backwards=True, nobire=self.nobire, out_sht_mode=self.shtmode) # lm_sky -> lm_pri
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
            almxfl(nlm[1], self.cls_filt['ee'] > 0, len(self.cls_filt_bool[0])-1, True)
            nlm[0] = np.zeros_like(nlm[1],dtype=complex)
            nlm[2] = np.zeros_like(nlm[1],dtype=complex)
        return nlm


    @log_on_start(logging.DEBUG, " ---- preconditioner_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- preconditioner_op", logger=log)
    def preconditioner_op(self, teblm):
        self.solve_eb = True
        self.Cbb_reg = 1e-30
        """
        Diagonal (per-ell) preconditioner approximating (S^{-1} + B^T N^{-1} B)^{-1}
        for solves in T-only, E-only, or EB space.

        EB case:
        - E prior uses self.icls[:,0,0] (assumed = 1/C_ell^EE or block-inverse element)
        - B prior uses a small regularized C_ell^BB_reg (=> inverse prior = 1/C_ell^BB_reg)
        - noise term uses ninv_fel for both E and B (and ninv_ftl for T if applicable)
        """
        assert teblm.shape[0] == 3, teblm.shape

        lmax_pri_ = Alm.getlmax(teblm[1].size, None)

        # --- helper: extend transfer-dependent spectra safely ---
        def _extend_pos_spline(arr, lmax_target, name):
            if (arr is None) or (len(arr) == 0):
                return np.zeros(lmax_target + 1, dtype=float)
            if len(arr) - 1 >= lmax_target:
                return arr[:lmax_target + 1]
            # extrapolate in log-space over positive support
            nz = np.where(arr > 0)
            if nz[0].size < 5:
                # too few points -> pad with last positive or zeros
                out = np.zeros(lmax_target + 1, dtype=float)
                if nz[0].size > 0:
                    out[:len(arr)] = arr
                    out[len(arr):] = arr[nz[0][-1]]
                return out
            log.debug(f"PRE_OP_DIAG: extending {name} from lmax {len(arr)-1} to lmax {lmax_target}")
            spl_sq = spl(np.arange(len(arr), dtype=float)[nz], np.log(arr[nz]), k=2, ext='extrapolate')
            return np.exp(spl_sq(np.arange(lmax_target + 1, dtype=float)))

        # --- get effective noise spectra in harmonic space: B^2 * N^{-1} (or analogous) ---
        ninv_ftebl = self.inv_operator.get_ftebl(self.beam_operator.transferfunction)
        ninv_ftl = _extend_pos_spline(ninv_ftebl[0], lmax_pri_, "ninv_ftl")  # length lmax+1
        ninv_fel = _extend_pos_spline(ninv_ftebl[1], lmax_pri_, "ninv_fel")
        ninv_fbl = _extend_pos_spline(ninv_ftebl[2], lmax_pri_, "ninv_fbl")

        # Decide what space we're solving in
        has_T = ('tt' in self.cls_filt)
        has_E = ('ee' in self.cls_filt)

        # EB solve if you want to allow B and you have polarization
        solve_eb = bool(getattr(self, "solve_eb", False)) and has_E

        # --- Build S^{-1} + noise diagonal blocks ---
        if has_T and has_E:
            # NOTE TE mixing case (keep your old 2x2 T/E block; B handled separately if solve_eb)
            lmax_ = lmax_pri_
            Si_TE = np.zeros((lmax_ + 1, 2, 2), dtype=float)

            # self.icls assumed shape (ell,2,2) for T/E inverse prior block
            Si_TE[:lmax_+1, 0, 0] = self.icls[:lmax_+1, 0, 0]
            Si_TE[:lmax_+1, 1, 1] = self.icls[:lmax_+1, 1, 1]
            Si_TE[:lmax_+1, 0, 1] = self.icls[:lmax_+1, 0, 1]
            Si_TE[:lmax_+1, 1, 0] = self.icls[:lmax_+1, 1, 0]

            Si_TE[:, 0, 0] += ninv_ftl[:lmax_+1]
            Si_TE[:, 1, 1] += ninv_fel[:lmax_+1]

            flmat_TE = np.linalg.pinv(Si_TE)  # (ell,2,2)

            tebout = np.zeros((3, teblm[0].size), dtype=complex)
            tebout[0] = almxfl(teblm[0], flmat_TE[:, 0, 0], lmax_, False) + almxfl(teblm[1], flmat_TE[:, 0, 1], lmax_, False)
            tebout[1] = almxfl(teblm[0], flmat_TE[:, 1, 0], lmax_, False) + almxfl(teblm[1], flmat_TE[:, 1, 1], lmax_, False)

            if solve_eb:
                # B block: (S_B^{-1} + ninv_fbl)^{-1}
                # Choose regularized C_ell^BB (in "prior" units), convert to inverse.
                Cbb_reg = getattr(self, "Cbb_reg", 1e-30)  # you should set this sensibly
                if np.isscalar(Cbb_reg):
                    icls_bb = np.full(lmax_ + 1, 1.0 / float(Cbb_reg), dtype=float)
                else:
                    Cbb_reg = _extend_cl(Cbb_reg, lmax_)
                    icls_bb = np.where(Cbb_reg > 0, 1.0 / Cbb_reg, 0.0)
                Si_B = icls_bb + ninv_fbl[:lmax_+1]
                flmat_B = np.where(Si_B > 0, 1.0 / Si_B, 0.0)
                tebout[2] = almxfl(teblm[2], flmat_B, lmax_, False)

            return tebout

        elif has_T:
            lmax_ = Alm.getlmax(teblm[0].size, None)
            Si_T = self.icls[:lmax_+1, 0, 0] + ninv_ftl[:lmax_+1]
            flmat_T = np.where(Si_T > 0, 1.0 / Si_T, 0.0)
            tebout = np.zeros((3, teblm[0].size), dtype=complex)
            tebout[0] = almxfl(teblm[0], flmat_T, lmax_, False)
            return tebout

        elif has_E:
            # NOTE Polarization-only solve (E or EB)
            lmax_ = lmax_pri_

            # E block
            icls_ee = self.icls[:lmax_+1, 0, 0]  # assumed inverse EE prior
            Si_E = icls_ee + ninv_fel[:lmax_+1]
            flmat_E = np.where(Si_E > 0, 1.0 / Si_E, 0.0)

            tebout = np.zeros((3, teblm[1].size), dtype=complex)
            tebout[1] = almxfl(teblm[1], flmat_E, lmax_, False)

            if solve_eb:
                # B block with regularization
                Cbb_reg = getattr(self, "Cbb_reg", 1e-30)
                if np.isscalar(Cbb_reg):
                    icls_bb = np.full(lmax_ + 1, 1.0 / float(Cbb_reg), dtype=float)
                else:
                    Cbb_reg = _extend_cl(Cbb_reg, lmax_)
                    icls_bb = np.where(Cbb_reg > 0, 1.0 / Cbb_reg, 0.0)

                Si_B = icls_bb + ninv_fbl[:lmax_+1]
                flmat_B = np.where(Si_B > 0, 1.0 / Si_B, 0.0)
                tebout[2] = almxfl(teblm[2], flmat_B, lmax_, False)

            return tebout

        else:
            return np.zeros_like(teblm)

    @log_on_start(logging.DEBUG, " ---- preconditioner_op", logger=log)
    @log_on_end(logging.DEBUG, " done ---- preconditioner_op", logger=log)  
    def preconditioner_op_nulledB(self, teblm):
        """
        NOTE this is the old preconditioner operation that nulls the B channel. This is ok in principle.. above implementation let's B "float" freely which may help with robustness
        """
        lmax_pri_ = Alm.getlmax(teblm[1].size, None)

        ninv_ftebl = self.inv_operator.get_ftebl(self.beam_operator.transferfunction)
        if np.any(ninv_ftebl[0]) and len(ninv_ftebl[0]) - 1 < lmax_pri_:  # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_ftl = ninv_ftebl[0]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s" % (len(ninv_ftl)-1, lmax_pri_))
            nz = np.where(ninv_ftl > 0)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_pri_ + 1, dtype=float)))
        else:
            ninv_ftl = ninv_ftebl[0]
        if np.any(ninv_ftebl[1]) and len(ninv_ftebl[1]) - 1 < lmax_pri_: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            ninv_fel = ninv_ftebl[1]
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_pri_))
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_pri_+1, dtype=float)))
        else:
            ninv_fel = ninv_ftebl[1]

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            lmax_sky_ = self.cls_filt['tt'].size
            Si = np.zeros((lmax_ + 1,2,2), dtype=float)
            Si[:lmax_sky_+1,0,0] = self.icls[:lmax_sky_+1,0,0]
            Si[:lmax_sky_+1,1,1] = self.icls[:lmax_sky_+1,1,1]
            Si[:lmax_sky_+1,0,1] = self.icls[:lmax_sky_+1,0,1]
            Si[:lmax_sky_+1,1,0] = self.icls[:lmax_sky_+1,1,0]
            Si[:,0,0] += ninv_ftl[:lmax_+1]
            Si[:,1,1] += ninv_fel[:lmax_+1]
            tebout = np.zeros(shape=(3,teblm[0].size), dtype=complex)
        elif 'tt' in self.cls_filt:
            lmax_sky_ = self.cls_filt['tt'].size
            Si = np.zeros((lmax_ + 1,1,1), dtype=float)
            Si[:lmax_sky_+1,0,0] = self.icls[:lmax_sky_+1,0,0]
            Si[:lmax_sky_+1,0,0] += ninv_ftl[:lmax_+1]
            tebout = np.zeros(shape=(3,teblm[0].size), dtype=complex)
        elif 'ee' in self.cls_filt:
            Si = np.zeros((lmax_pri_ + 1,1,1), dtype=float)
            Si[:lmax_pri_+1,0,0] = self.icls[:lmax_pri_+1,0,0]
            Si[:lmax_pri_+1,0,0] += ninv_fel[:lmax_pri_+1]

            tebout = np.zeros(shape=(3,teblm[1].size), dtype=complex)
        flmat = np.linalg.pinv(Si) # TODO lmin_teb fix

        if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], lmax_, False) + almxfl(teblm[1], flmat[:, 0, 1], lmax_, False)
            tebout[1] = almxfl(teblm[0], flmat[:, 1, 0], lmax_, False) + almxfl(teblm[1], flmat[:, 1, 1], lmax_, False)
        elif 'tt' in self.cls_filt:
            tebout[0] = almxfl(teblm[0], flmat[:, 0, 0], lmax_, False)
        elif 'ee' in self.cls_filt:
            tebout[1] = almxfl(teblm[1], flmat[:, 0, 0], lmax_pri_, False)
        return tebout
    

    @log_on_start(logging.DEBUG, " ---- get_ivfreslm: {it}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- get_ivfreslm", logger=log)
    def get_ivfreslm(self, it, data=None, elm_wf=None):
        # assert elm_wf.shape[0] == 3, elm_wf.shape
        # NOTE this is eq. 21 of the paper
        if not self.ivfres_field.is_cached(it=it):
            assert elm_wf is not None and data is not None
            ivfreslm = self.sec_operator.act(elm_wf, nobire=False, out_sht_mode=self.shtmode)
            assert ivfreslm.shape[0] == 3, ivfreslm.shape
            ivfreslm = 1*self.beam_operator.act(ivfreslm)
            
            if data[0].dtype in [np.complex64, np.complex128]:
                ivfreslm = data - ivfreslm
                ivfreslm = self.inv_operator.act(ivfreslm, adjoint=False)
            else:
                ivfresmap = [
                    self.inv_operator.geom_lib.synthesis(ivfreslm[0], 0, *self.inv_operator.lm_max, self.sht_tr)[0],
                    *self.inv_operator.geom_lib.synthesis(ivfreslm[1:], 2, *self.inv_operator.lm_max, self.sht_tr)
                ]
                ivfresmap = [d-ivf for ivf,d in zip(ivfresmap,data)]
                ivfreslm = self.inv_operator.act(np.array(ivfresmap))

            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False, factor_p=.5)
            # TODO need to check why I have this if-tree here, seems fishy
            if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                pass
                # ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'tt' in self.cls_filt:
                ivfreslm[1] = np.zeros_like(ivfreslm[0],dtype=complex)
                ivfreslm[2] = np.zeros_like(ivfreslm[0],dtype=complex)
            elif 'ee' in self.cls_filt:
                ivfreslm[0] = np.zeros_like(ivfreslm[1],dtype=complex)
                # ivfreslm[2] = np.zeros_like(ivfreslm[1],dtype=complex)
            self.ivfres_field.cache(ivfreslm, it=it)
        return self.ivfres_field.get_field(it=it)# or ivfreslm
    

    def invert_cls_filt(self, cls_filt):
        if 'tt' in cls_filt and 'ee' in cls_filt:
            Si = np.zeros((cls_filt['tt'].size, 2, 2), dtype=float)
            Si[:, 0, 0] = cls_filt['tt']
            Si[:, 0, 1] = cls_filt['te']
            Si[:, 1, 0] = cls_filt['te']
            Si[:, 1, 1] = cls_filt['ee']
        elif 'tt' in cls_filt:
            Si = np.zeros((cls_filt['tt'].size, 1, 1), dtype=float)
            Si[:, 0, 0] = cls_filt['tt']
        elif 'ee' in cls_filt:
            Si = np.zeros((cls_filt['ee'].size, 1, 1), dtype=float)
            Si[:, 0, 0] = cls_filt['ee']
        return np.linalg.pinv(Si)


    def update_operator(self, field):
        self.sec_operator.set_field(field)

    def get_field_operator(self):
        return self.sec_operator.get_field()

    # TODO this should not sit in filter, rather in ..TBD
    def get_template(self, it, QE_perturbative=True, secondary=None, component=None, order='reversed'):
        config = get_config()
        estCMB = np.zeros(shape=(3,Alm.getsize(*config.lm_max_pri)),dtype=complex)
        if it == 0:
            if 'tt' in self.cls_filt and 'ee' in self.cls_filt:
                estCMB[0:2] = self.wf_field.get_field(it=it)
            elif 'tt' in self.cls_filt:
                estCMB[0] = self.wf_field.get_field(it=it)
            elif 'ee' in self.cls_filt:
                estCMB[1] = self.wf_field.get_field(it=it)
            
            estCMB = alm_copy_nd(estCMB, config.lm_max_pri[1], config.lm_max_sky)
        else:
            estCMB = self.wf_field.get_field(it=it)

        for operator in self.sec_operator.operators:
            # if operator.ID == 'lensing':
            if QE_perturbative:
                operator.perturbative = (it == 0)
            else:
                operator.perturbative = False
        return self.sec_operator.act(estCMB, secondary=secondary, order=order)
    
    def get_mchain(self):
        return self.mchain