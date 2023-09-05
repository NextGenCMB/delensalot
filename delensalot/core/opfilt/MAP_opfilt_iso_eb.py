"""Inverse-variance CMB filter, inclusive of CMB lensing remapping

    This module distinguishes between deflection fields mapping E to E and E to B



"""
import logging
log = logging.getLogger(__name__)
import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from lenspyx import remapping
from lenspyx.utils_hp import almxfl,   Alm, synalm
from lenspyx.utils import timer, cli
from lenspyx.remapping.utils_geom import pbdGeometry

from delensalot.core.opfilt import opfilt_base, MAP_opfilt_aniso_p

pre_op_dense = None # not implemented
dot_op = MAP_opfilt_aniso_p.dot_op
fwd_op = MAP_opfilt_aniso_p.fwd_op
apply_fini = MAP_opfilt_aniso_p.apply_fini

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class alm_filter_nlev_wl(opfilt_base.alm_filter_wl):
    def __init__(self, nlev_p:float or np.ndarray, ffi_ee:remapping.deflection, ffi_eb:remapping.deflection, transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple,
                 transf_b:None or np.ndarray=None, nlev_b:None or float or np.ndarray=None, wee=True, verbose=False):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_p: CMB-E filtering noise level in uK-amin
                            (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    ffi: lenscarf deflection instance
                    transf: CMB E-mode transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)
                    transf_b(optional): CMB B-mode transfer function (if different from E)
                    nlev_b(optional): CMB-B filtering noise level in uK-amin
                             (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    wee: includes EE-like term in generalized QE if set

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

        super().__init__(lmax_sol, mmax_sol, ffi_ee)
        self.lmax_len = min(lmax_len, lmax_transf)
        self.mmax_len = min(mmax_len, self.lmax_len)

        transf_elm = transf
        transf_blm = transf_b if transf_b is not None else transf

        nlev_elm = _extend_cl(nlev_e, lmax_len)
        nlev_blm = _extend_cl(nlev_b, lmax_len)

        self.inoise_2_elm  = _extend_cl(transf_elm ** 2, lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_elm  = _extend_cl(transf_elm ** 1 ,lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2

        self.inoise_2_blm = _extend_cl(transf_blm ** 2, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_blm = _extend_cl(transf_blm ** 1, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2

        self.transf_elm  = _extend_cl(transf_elm, lmax_len)
        self.transf_blm  = _extend_cl(transf_blm, lmax_len)

        self.nlev_elm = nlev_elm
        self.nlev_blm = nlev_blm

        self.ffi_eb = ffi_eb
        self.ffi_ee = ffi_ee

        del self.ffi

        self.verbose = verbose
        self.wee = wee
        self.tim = timer(True, prefix='opfilt')

    def lensforward(self, elm): # this can only take a e and no b
        assert elm.size == Alm.getsize(self.lmax_sol, self.mmax_sol)
        eblm = self.ffi_ee.lensgclm(elm, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        if self.ffi_eb is not self.ffi_ee: # filling B-mode with second deflection
            eblm_2 = self.ffi_eb.lensgclm(elm, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
            eblm[1][:] = eblm_2[1]
        return eblm

    def lensbackward(self, eblm):
        eblm_ee = self.ffi_ee.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        if self.ffi_eb is not self.ffi_ee: # filling B-mode with second deflection
            eblm_eb = self.ffi_eb.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        else:
            eblm_eb = eblm_ee
        return 0.5 * (eblm_ee[0] + eblm_eb[0])

    def _test_adjoint(self, cl):
        from lenspyx.utils_hp import alm2cl
        elm = synalm(cl, self.lmax_sol, self.mmax_sol)
        elm_len = synalm(cl, self.lmax_len, self.mmax_len)
        blm_len = synalm(cl, self.lmax_len, self.mmax_len)
        De = self.lensforward(elm)
        ret1  = np.sum(alm2cl(De[0], elm_len, self.lmax_len, self.mmax_len, None) * (2 * np.arange(self.lmax_len + 1) + 1))
        ret1 += np.sum(alm2cl(De[1], blm_len, self.lmax_len, self.mmax_len, None) * (2 * np.arange(self.lmax_len + 1) + 1))
        del De
        Dt = self.lensbackward(np.array([elm_len, blm_len]))
        ret2 =  np.sum(alm2cl(elm, Dt, self.lmax_sol, self.mmax_sol, None) * (2 * np.arange(self.lmax_sol + 1) + 1))
        print(ret1, ret2-ret1, ret2)

    def get_febl(self):
        return np.copy(self.inoise_2_elm), np.copy(self.inoise_2_blm)

    def set_ffi(self, ffi:list):
        """Update of lensing deflection instance"""
        assert len(ffi) == 2
        self.ffi_ee = ffi[0]
        self.ffi_eb = ffi[1]

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        self.tim.reset()
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        eblm = self.lensforward(elm)
        self.tim.add('lensgclm fwd')
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)
        self.tim.add('transf')

        # NB: inplace is fine but only if precision of elm array matches that of the interpolator
        elm[:] = self.lensbackward(eblm)
        self.tim.add('lensgclm bwd')
        if self.verbose:
            print(self.tim)

    def apply_map(self, eblm:np.ndarray):
        """Applies noise operator in place"""
        almxfl(eblm[0], self.inoise_1_elm * cli(self.transf_elm), self.mmax_len, True)
        almxfl(eblm[1], self.inoise_1_blm * cli(self.transf_elm), self.mmax_len, True)

    def synalm(self, unlcmb_cls:dict, cmb_phas=None, get_unlelm=True):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        eblm = self.lensbackward(elm)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, True)
        eblm[0] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_elm / 180 / 60 * np.pi) ** 2) * (self.transf_elm > 0), self.lmax_len, self.mmax_len)
        eblm[1] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_blm / 180 / 60 * np.pi) ** 2) * (self.transf_blm > 0), self.lmax_len, self.mmax_len)
        return elm, eblm if get_unlelm else eblm


    def get_qlms(self, qudat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, alm_wf_leg2 :None or np.ndarray=None):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: gradient leg Winer filtered CMB, if different from ivf leg
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            Note: all implementation signs are super-weird but end result correct...
        """
        assert len(qudat) == 2
        (repmap_e, impmap_e), (repmap_b, impmap_b) = self._get_irespmap(qudat, elm_wf, q_pbgeom)
        if alm_wf_leg2 is not None:
            elm_wf[:] = alm_wf_leg2
        Gs, Cs = self._get_gpmap(elm_wf, 3, q_pbgeom, self.ffi_ee)  # 2 pos.space maps
        GC_ee = (repmap_e - 1j * impmap_e) * (Gs + 1j * Cs)  # (-2 , +3)
        if self.ffi_eb is not self.ffi_ee:
            Gs, Cs = self._get_gpmap(elm_wf, 3, q_pbgeom, self.ffi_eb)  # 2 pos.space maps
        GC_eb = (repmap_b - 1j * impmap_b) * (Gs + 1j * Cs)  # (-2 , +3)

        Gs, Cs = self._get_gpmap(elm_wf, 1,  q_pbgeom, self.ffi_ee)
        GC_ee -= (repmap_e + 1j * impmap_e) * (Gs - 1j * Cs)  # (+2 , -1) # this comes with minus sign
        if self.ffi_eb is not self.ffi_ee:
            Gs, Cs = self._get_gpmap(elm_wf, 1, q_pbgeom, self.ffi_eb)
        GC_eb -= (repmap_b + 1j * impmap_b) * (Gs - 1j * Cs)  # (+2 , -1)

        del repmap_e, impmap_e, repmap_b, impmap_b, Gs, Cs
        lmax_qlm = self.ffi_ee.lmax_dlm
        mmax_qlm = self.ffi_ee.mmax_dlm
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        Gee, Cee = q_pbgeom.geom.adjoint_synthesis([GC_ee.real, GC_ee.imag], 1, lmax_qlm, mmax_qlm, self.ffi_ee.sht_tr)
        del GC_ee
        for G in [Gee, Cee]:
            almxfl(G, fl, mmax_qlm, True)
        lmax_qlm = self.ffi_eb.lmax_dlm
        mmax_qlm = self.ffi_eb.mmax_dlm
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        Geb, Ceb = q_pbgeom.geom.adjoint_synthesis([GC_eb.real, GC_eb.imag], 1, lmax_qlm, mmax_qlm, self.ffi_eb.sht_tr)
        del GC_eb
        for G in [Geb, Ceb]:
            almxfl(G, fl, mmax_qlm, True)
        return (Gee, Geb), (Cee, Ceb)

    def get_qlms_mf(self, mfkey, q_pbgeom:pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        assert 0, 'implement this'

    def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray, q_pbgeom:pbdGeometry, map_out=None):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert len(eblm_dat) == 2
        ebwf = self.lensforward(eblm_wf)
        almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
        ebwf += eblm_dat
        almxfl(ebwf[0], self.inoise_1_elm * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.inoise_1_blm * 0.5,            self.mmax_len, True)
        res_e = q_pbgeom.geom.synthesis(ebwf[0], 2, self.lmax_len, self.mmax_len, self.ffi_ee.sht_tr, mode='GRAD_ONLY')
        res_b = q_pbgeom.geom.synthesis([np.zeros_like(ebwf[1]), ebwf[1]], 2, self.lmax_len, self.mmax_len, self.ffi_eb.sht_tr)
        return res_e, res_b

        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi_ee.sht_tr, map=map_out)

    def _get_gpmap(self, elm_wf:np.ndarray or list, spin:int, q_pbgeom:pbdGeometry, ffi:remapping.deflection):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert  Alm.getlmax(elm_wf.size, self.mmax_sol) == self.lmax_sol, ( Alm.getlmax(elm_wf.size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = almxfl(elm_wf, fl, self.mmax_sol, False)
        ffi = ffi.change_geom(q_pbgeom.geom) if q_pbgeom.geom is not ffi.pbgeom.geom else ffi
        return ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)

class pre_op_diag:
    """Cg-inversion diagonal preconditioner


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel, ninv_fbl = ninv_filt.get_febl()
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        flmat = cli(s_cls['ee'][:lmax_sol + 1]) + ninv_fel[:lmax_sol + 1]
        self.flmat = cli(flmat) * (s_cls['ee'][:lmax_sol + 1] > 0.)
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, eblm):
        return self.calc(eblm)

    def calc(self, elm):
        assert Alm.getsize(self.lmax, self.mmax) == elm.size, (self.lmax, self.mmax, Alm.getlmax(elm.size, self.mmax))
        return almxfl(elm, self.flmat, self.mmax, False)


def calc_prep(eblm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
    """cg-inversion pre-operation

        This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`

        Args:
            eblm: input data polarisation elm and blm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(eblm, np.ndarray) and eblm.ndim == 2
    assert Alm.getlmax(eblm[0].size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(eblm[0].size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    eblmc = np.empty_like(eblm)
    eblmc[0] = almxfl(eblm[0], ninv_filt.inoise_1_elm, ninv_filt.mmax_len, False)
    eblmc[1] = almxfl(eblm[1], ninv_filt.inoise_1_blm, ninv_filt.mmax_len, False)
    elm = ninv_filt.lensbackward(eblmc)
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
    return elm