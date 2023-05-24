"""Lenspyx-geometry based inverse-variance filters, inclusive of CMB lensing remapping

    This module deals with joint T-P (GMV-like) reconstruction working on idealized skies with homogeneous or colored noise spectra


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from lenspyx.utils_hp import almxfl,   Alm, synalm, alm2cl
from lenspyx.utils import timer, cli
from lenspyx.remapping.utils_geom import pbdGeometry
from lenspyx import remapping
from scipy.interpolate import UnivariateSpline as spl
from delensalot.core.opfilt import opfilt_base
from lenspyx.remapping.deflection_028 import rtype, ctype
pre_op_dense = None # not implemented

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass
def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class alm_filter_nlev_wl(opfilt_base.alm_filter_wl):
    def __init__(self, nlev_t:float or np.ndarray, nlev_p:float or np.ndarray, ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple,
                 transf_e:None, transf_b:None or np.ndarray=None, nlev_b:None or float or np.ndarray=None, verbose=False):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_t: CMB-T filtering noise level in uK-amin
                            (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    nlev_p: CMB-E filtering noise level in uK-amin
                            (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    ffi: lenscarf deflection instance
                    transf: CMB T-mode transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)
                    transf_e(optional): CMB B-mode transfer function (if different from T)
                    transf_b(optional): CMB B-mode transfer function (if different from T)
                    nlev_b(optional): CMB-B filtering noise level in uK-amin
                             (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)

                Note:
                    All operations are in harmonic space.
                    Mode exclusions can be implemented setting the transfer fct to zero
                    (but the instance still expects the Elm and Blm arrays to have the same formal lmax)


        """
        lmax_sol, mmax_sol = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = max(len(transf), len(transf if transf_b is None else transf_b)) - 1
        nlev_e = nlev_p
        nlev_b = nlev_p if nlev_b is None else nlev_b

        super().__init__(lmax_sol, mmax_sol, ffi)
        self.lmax_len = min(lmax_len, lmax_transf)
        self.mmax_len = min(mmax_len, self.lmax_len)

        transf_tlm = transf
        transf_elm = transf_e if transf_e is not None else transf
        transf_blm = transf_b if transf_b is not None else transf

        nlev_tlm = _extend_cl(nlev_t, lmax_len)
        nlev_elm = _extend_cl(nlev_e, lmax_len)
        nlev_blm = _extend_cl(nlev_b, lmax_len)

        self.inoise_2_tlm  = _extend_cl(transf_tlm ** 2, lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_tlm  = _extend_cl(transf_tlm ** 1 ,lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2

        self.inoise_2_elm  = _extend_cl(transf_elm ** 2, lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_elm  = _extend_cl(transf_elm ** 1 ,lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2

        self.inoise_2_blm = _extend_cl(transf_blm ** 2, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_blm = _extend_cl(transf_blm ** 1, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2

        self.transf_tlm  = _extend_cl(transf_tlm, lmax_len)
        self.transf_elm  = _extend_cl(transf_elm, lmax_len)
        self.transf_blm  = _extend_cl(transf_blm, lmax_len)

        self.nlev_tlm = nlev_tlm
        self.nlev_elm = nlev_elm
        self.nlev_blm = nlev_blm

        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')

    def get_ftebl(self):
        return np.copy(self.inoise_2_tlm), np.copy(self.inoise_2_elm), np.copy(self.inoise_2_blm)

    def set_ffi(self, ffi:remapping.deflection):
        self.ffi = ffi

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def apply_alm(self, telm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        assert telm.ndim == 2 and len(telm) == 2
        self.tim.reset()
        lmax_unl = Alm.getlmax(telm[0].size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        # View to the same array for GRAD_ONLY mode:
        eblm = self.ffi.lensgclm(telm[1:],  self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        tlm  = self.ffi.lensgclm(telm[0:1], self.mmax_sol, 0, self.lmax_len, self.mmax_len)
        tlm.reshape((1, tlm.size))
        self.tim.add('lensgclm fwd')
        almxfl( tlm[0], self.inoise_2_tlm, self.mmax_len, inplace=True)
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)
        self.tim.add('transf')

        # NB: inplace is fine but only if precision of elm array matches that of the interpolator
        self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
                                 backwards=True, gclm_out=telm[1:], out_sht_mode='GRAD_ONLY')
        self.ffi.lensgclm(tlm,  self.mmax_len, 0, self.lmax_sol, self.mmax_sol,
                                 backwards=True, gclm_out=telm[0:1])
        self.tim.add('lensgclm bwd')
        if self.verbose:
            print(self.tim)

    def apply_map(self, teblm:np.ndarray):
        """Applies noise operator in place"""
        assert teblm.ndim == 2 and len(teblm) == 3
        almxfl(teblm[0], self.inoise_1_tlm * cli(self.transf_tlm), self.mmax_len, True)
        almxfl(teblm[1], self.inoise_1_elm * cli(self.transf_elm), self.mmax_len, True)
        almxfl(teblm[2], self.inoise_1_blm * cli(self.transf_blm), self.mmax_len, True)

    def synalm(self, unlcmb_cls:dict, cmb_phas=None, get_unlelm=True):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        assert 0, 'fixthis'
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        eblm = self.ffi.lensgclm(np.atleast_2d(elm), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, True)
        eblm[0] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_elm / 180 / 60 * np.pi) ** 2) * (self.transf_elm > 0), self.lmax_len, self.mmax_len)
        eblm[1] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_blm / 180 / 60 * np.pi) ** 2) * (self.transf_blm > 0), self.lmax_len, self.mmax_len)
        return elm, eblm if get_unlelm else eblm


    def get_qlms(self, teblm_dat: np.ndarray or list, telm_wf: np.ndarray, q_pbgeom: pbdGeometry, alm_wf_leg2:None or np.ndarray =None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert alm_wf_leg2 is None
        assert Alm.getlmax(teblm_dat[0].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(teblm_dat[0].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(teblm_dat[1].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(teblm_dat[1].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(teblm_dat[2].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(teblm_dat[2].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(telm_wf[0].size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(telm_wf[0].size, self.mmax_sol), self.lmax_sol)
        assert Alm.getlmax(telm_wf[0].size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(telm_wf[1].size, self.mmax_sol), self.lmax_sol)

        tlm_wf, elm_wf = telm_wf
        tlm_dat = teblm_dat[0]
        eblm_dat = teblm_dat[1:3]

        # Spin-2 part
        resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
        resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        self._get_irespmap(eblm_dat, elm_wf, q_pbgeom, map_out=resmap_r) # inplace onto resmap_c and resmap_r

        gcs_r = self._get_gpmap(elm_wf, 3, q_pbgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = self._get_gpmap(elm_wf, 1, q_pbgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r

        # Spin-0 part
        gc_c += self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(tlm_wf, q_pbgeom)

        # Projection onto gradient and curl
        lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        gc = q_pbgeom.geom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
        del gc_r, gc_c
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(gc[0], fl, mmax_qlm, True)
        almxfl(gc[1], fl, mmax_qlm, True)
        return gc

    def get_qlms_mf(self, mfkey, q_pbgeom:pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        assert 0, 'fix this'
        if mfkey in [1]: # This should be B^t x, D dC D^t B^t Covi x, x random phases in alm space
            if phas is None:
                phas = np.array([synalm(np.ones(self.lmax_len + 1, dtype=float), self.lmax_len, self.mmax_len),
                                 synalm(np.ones(self.lmax_len + 1, dtype=float), self.lmax_len, self.mmax_len)])
            assert Alm.getlmax(phas[0].size, self.mmax_len) == self.lmax_len
            assert Alm.getlmax(phas[1].size, self.mmax_len) == self.lmax_len

            soltn = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(soltn, phas, dot_op=self.dot_op())

            almxfl(phas[0], 0.5 * self.transf_elm, self.mmax_len, True)
            almxfl(phas[1], 0.5 * self.transf_blm, self.mmax_len, True)
            repmap, impmap = q_pbgeom.geom.synthesis(phas, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr)

            Gs, Cs = self._get_gpmap(soltn, 3, q_pbgeom)  # 2 pos.space maps
            GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
            Gs, Cs = self._get_gpmap(soltn, 1, q_pbgeom)
            GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
            del repmap, impmap, Gs, Cs
        elif mfkey in [0]: # standard gQE, quite inefficient but simple
            assert phas is None, 'discarding this phase anyways'
            elm_pha, eblm_dat = self.synalm(cls_filt)
            eblm_dat = np.array(eblm_dat)
            elm_wf = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(elm_wf, eblm_dat, dot_op=self.dot_op())
            return self.get_qlms(eblm_dat, elm_wf, q_pbgeom)

        else:
            assert 0, mfkey + ' not implemented'
        lmax_qlm = self.ffi.lmax_dlm
        mmax_qlm = self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.adjoint_synthesis(np.array([GC.real, GC.imag]), 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray, q_pbgeom:pbdGeometry, map_out=None):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert len(eblm_dat) == 2
        ebwf = self.ffi.lensgclm(np.atleast_2d(eblm_wf), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
        ebwf += eblm_dat
        almxfl(ebwf[0], self.inoise_1_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.inoise_1_blm * 0.5, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)

    def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert elm_wf.ndim == 1
        assert Alm.getlmax(elm_wf.size, self.mmax_sol) == self.lmax_sol
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = np.atleast_2d(almxfl(elm_wf, fl, self.mmax_sol, False))
        ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)

    def _get_irestmap(self, tlm_dat:np.ndarray, tlm_wf:np.ndarray, q_pbgeom: pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        twf = tlm_dat - almxfl(self.ffi.lensgclm(tlm_wf, self.mmax_sol, 0, self.lmax_len, self.mmax_len).squeeze(), self.transf_tlm, self.mmax_len, False)
        almxfl(twf, self.inoise_1_tlm, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(twf, 0, self.lmax_len, self.mmax_len, self.ffi.sht_tr).squeeze()

    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom: pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert Alm.getlmax(tlm_wf.size, self.mmax_sol) == self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        alm = almxfl(tlm_wf, fl, self.mmax_sol, False)
        return ffi.gclm2lenmap(alm, self.mmax_sol, 1, False)



class pre_op_diag:
    """Cg-inversion diagonal preconditioner


    """
    def __init__(self, s_cls: dict, ninv_filt: alm_filter_nlev_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        assert len(s_cls['tt']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['tt']))
        assert len(s_cls['te']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['te']))

        lmax_sol = ninv_filt.lmax_sol
        ninv_ftl, ninv_fel, ninv_fbl = ninv_filt.get_ftebl()
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        if len(ninv_ftl) - 1 < lmax_sol:  # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s" % (len(ninv_fel) - 1, lmax_sol))
            assert np.all(ninv_ftl >= 0)
            nz = np.where(ninv_ftl > 0)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))

        Si = np.empty((lmax_sol + 1,2,2), dtype=float)
        Si[:, 1, 1] = s_cls['ee'][:lmax_sol + 1]
        Si[:, 0, 1] = s_cls['te'][:lmax_sol + 1]
        Si[:, 1, 0] = s_cls['te'][:lmax_sol + 1]
        Si[:, 0, 0] = s_cls['tt'][:lmax_sol + 1]
        Si = np.linalg.pinv(Si)
        Si[:, 0, 0] += ninv_ftl[:lmax_sol + 1]
        Si[:, 1, 1] += ninv_fel[:lmax_sol + 1]
        flmat = np.linalg.pinv(Si)
        self.flmat = flmat
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, telm):
        return self.calc(telm)

    def calc(self, telm):
        assert telm.ndim == 2 and len(telm) == 2, (telm.shape)
        tlm, elm = telm
        assert Alm.getsize(self.lmax, self.mmax) == elm.size, (self.lmax, self.mmax, Alm.getlmax(elm.size, self.mmax))
        assert Alm.getsize(self.lmax, self.mmax) == tlm.size, (self.lmax, self.mmax, Alm.getlmax(tlm.size, self.mmax))
        teout = np.empty_like(telm)
        teout[0] = almxfl(tlm, self.flmat[:, 0, 0], self.mmax, False) + almxfl(elm, self.flmat[:, 0, 1], self.mmax, False)
        teout[1] = almxfl(tlm, self.flmat[:, 1, 0], self.mmax, False) + almxfl(elm, self.flmat[:, 1, 1], self.mmax, False)
        return teout


def calc_prep(teblm:np.ndarray, s_cls: dict, ninv_filt: alm_filter_nlev_wl):
    r"""cg-inversion pre-operation

        This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`

        Args:
            teblm: input data polarisation elm and blm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(teblm, np.ndarray) and teblm.ndim == 2 and len(teblm) == 3, teblm.shape
    assert np.iscomplexobj(teblm)
    assert Alm.getlmax(teblm[0].size, ninv_filt.mmax_len) == ninv_filt.lmax_len
    assert Alm.getlmax(teblm[1].size, ninv_filt.mmax_len) == ninv_filt.lmax_len
    assert Alm.getlmax(teblm[2].size, ninv_filt.mmax_len) == ninv_filt.lmax_len
    eblmc = np.empty_like(teblm)
    eblmc[0] = almxfl(teblm[0], ninv_filt.inoise_1_tlm, ninv_filt.mmax_len, False)
    eblmc[1] = almxfl(teblm[1], ninv_filt.inoise_1_elm, ninv_filt.mmax_len, False)
    eblmc[2] = almxfl(teblm[2], ninv_filt.inoise_1_blm, ninv_filt.mmax_len, False)

    telm = np.empty((2, Alm.getsize(ninv_filt.lmax_sol, ninv_filt.mmax_sol)), dtype=teblm.dtype)
    ninv_filt.ffi.lensgclm(eblmc[1:], ninv_filt.mmax_len, 2, ninv_filt.lmax_sol,ninv_filt.mmax_sol,
                                      backwards=True, out_sht_mode='GRAD_ONLY', gclm_out=telm[1:]).squeeze()
    ninv_filt.ffi.lensgclm(eblmc[0], ninv_filt.mmax_len, 0, ninv_filt.lmax_sol, ninv_filt.mmax_sol,
                                      backwards=True, out_sht_mode='STANDARD', gclm_out=telm[0:1]).squeeze()
    almxfl(telm[0], s_cls['tt'] > 0., ninv_filt.mmax_sol, True)
    almxfl(telm[1], s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
    return telm


class fwd_op:
    """Forward operation for temperature-only


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
        Si = np.empty((ninv_filt.lmax_sol + 1, 2, 2), dtype=float)
        Si[:, 1, 1] = s_cls['ee'][:ninv_filt.lmax_sol + 1]
        Si[:, 0, 1] = s_cls['te'][:ninv_filt.lmax_sol + 1]
        Si[:, 1, 0] = s_cls['te'][:ninv_filt.lmax_sol + 1]
        Si[:, 0, 0] = s_cls['tt'][:ninv_filt.lmax_sol + 1]
        self.cls = s_cls
        self.icls = np.linalg.pinv(Si)
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def __call__(self, telm):
        return self.calc(telm)

    def calc(self, telm):
        assert len(telm) == 2
        nlm = np.copy(telm)
        self.ninv_filt.apply_alm(nlm)
        nlm[0] += almxfl(telm[0], self.icls[:, 0, 0], self.mmax_sol, False)
        nlm[0] += almxfl(telm[1], self.icls[:, 0, 1], self.mmax_sol, False)
        nlm[1] += almxfl(telm[1], self.icls[:, 1, 1], self.mmax_sol, False)
        nlm[1] += almxfl(telm[0], self.icls[:, 1, 0], self.mmax_sol, False)
        almxfl(nlm[0], self.cls['tt'] > 0, self.mmax_sol, True)
        almxfl(nlm[1], self.cls['ee'] > 0, self.mmax_sol, True)
        return nlm

class dot_op:
    def __init__(self, lmax: int, mmax: int or None, lmin=0):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


        """
        if mmax is None or mmax < 0: mmax = lmax
        self.lmax = lmax
        self.mmax = min(mmax, lmax)
        self.lmin = int(lmin)

    def __call__(self, telm1, telm2):
        assert len(telm1) == 2 and len(telm2) == 2
        tlm1, elm1 = telm1
        tlm2, elm2 = telm2

        ret =  np.sum(alm2cl(elm1, elm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))
        ret += np.sum(alm2cl(tlm1, tlm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))
        return ret