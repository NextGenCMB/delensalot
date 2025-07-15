"""Inverse-variance filters, inclusive of CMB lensing remapping

    This module is suitable for analysis of E-only CMB polarization data



"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
from scipy.interpolate import UnivariateSpline as spl
import numpy as np

from lenspyx import remapping
from lenspyx.utils import timer, cli
from lenspyx.utils_hp import almxfl, Alm, synalm
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
    def __init__(self, nlev_p:float or np.ndarray, ffi:remapping.deflection, transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple,
                        verbose=False):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_p: CMB-E filtering noise level in uK-amin
                            (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    ffi: lenscarf deflection instance
                    transf: CMB E-mode transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)

                Note:
                    All operations are in harmonic space.
                    Mode exclusions can be implemented setting the transfer fct to zero


        """
        lmax_sol, mmax_sol = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = len(transf) - 1
        nlev_e = nlev_p

        super().__init__(lmax_sol, mmax_sol, ffi)
        self.lmax_len = min(lmax_len, lmax_transf)
        self.mmax_len = min(mmax_len, self.lmax_len)

        transf_elm = transf

        nlev_elm = _extend_cl(nlev_e, lmax_len)

        self.inoise_2_elm  = _extend_cl(transf_elm ** 2, lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_elm  = _extend_cl(transf_elm ** 1 ,lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2

        self.transf_elm  = _extend_cl(transf_elm, lmax_len)

        self.nlev_elm = nlev_elm

        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')

    def get_fel(self):
        return np.copy(self.inoise_2_elm)

    def set_ffi(self, ffi:remapping.deflection):
        self.ffi = ffi

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        self.tim.reset()
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = elm.reshape((1, elm.size))
        elm_t = self.ffi.lensgclm(elm_2d, self.mmax_sol, 2, self.lmax_len, self.mmax_len, out_sht_mode='GRAD_ONLY')
        self.tim.add('lensgclm fwd')
        almxfl(elm_t.squeeze(), self.inoise_2_elm, self.mmax_len, inplace=True)
        self.tim.add('transf')

        self.ffi.lensgclm(np.atleast_2d(elm_t), self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
                                 backwards=True, out_sht_mode='GRAD_ONLY', gclm_out=elm_2d)
        self.tim.add('lensgclm bwd')
        if self.verbose:
            print(self.tim)

    def apply_map(self, elm:np.ndarray):
        """Applies noise operator in place"""
        almxfl(elm.squeeze(), self.inoise_1_elm * cli(self.transf_elm), self.mmax_len, True)
    
    def degrade(self, nside, lmax, mmax, set_deflection_to_zero=True):
        """Degradation of the filter to lower resolution
        """
        print('Not degrading filter {} {}'.format(len(self.nlev_elm), lmax))
        return self
    
    def synalm(self, unlcmb_cls:dict, cmb_phas=None, get_unlelm=True):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        elm = self.ffi.lensgclm(np.atleast_2d(elm), self.mmax_sol, 2, self.lmax_len, self.mmax_len, False,
                                out_sht_mode='GRAD_ONLY').squeeze()
        almxfl(elm, self.transf_elm, self.mmax_len, True)
        elm += synalm((np.ones(self.lmax_len + 1) * (self.nlev_elm / 180 / 60 * np.pi) ** 2) * (self.transf_elm > 0), self.lmax_len, self.mmax_len)
        return elm

    def get_qlms(self, elm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, alm_wf_leg2:None or np.ndarray =None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        elm_dat_2d = np.atleast_2d(elm_dat)
        assert Alm.getlmax(elm_dat_2d[0].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(elm_dat_2d[0].size, self.mmax_len), self.lmax_len)
        ebwf = np.atleast_2d(elm_wf)
        repmap, impmap = self._get_irespmap(elm_dat_2d, ebwf, q_pbgeom)
        if alm_wf_leg2 is not None:
            assert Alm.getlmax(alm_wf_leg2.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(alm_wf_leg2.size, self.mmax_sol), self.lmax_sol)
            ebwf[0, :] = alm_wf_leg2
        Gs, Cs = self._get_gpmap(ebwf, 3, q_pbgeom)  # 2 pos.space maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(ebwf, 1, q_pbgeom)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.adjoint_synthesis(np.array([GC.real, GC.imag]), 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C


    def _get_irespmap(self, elm_dat:np.ndarray, elm_wf:np.ndarray, q_pbgeom:pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        ewf = self.ffi.lensgclm(elm_wf, self.mmax_sol, 2, self.lmax_len, self.mmax_len, out_sht_mode='GRAD_ONLY').squeeze()
        almxfl(ewf, (-1) * self.transf_elm, self.mmax_len, True)
        ewf += elm_dat.squeeze()
        almxfl(ewf, self.inoise_1_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        return q_pbgeom.geom.synthesis(np.atleast_2d(ewf), 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, mode='GRAD_ONLY')

    def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        elm_wf_2d = np.atleast_2d(elm_wf)
        assert  Alm.getlmax(elm_wf_2d[0].size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(elm_wf_2d[0].size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf_2d[0].size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm_wf_s = np.atleast_2d(almxfl(elm_wf_2d[0], fl, self.mmax_sol, False))
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap(elm_wf_s, self.mmax_sol, spin, False)

class pre_op_diag:
    """Cg-inversion diagonal preconditioner


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel = ninv_filt.get_fel()
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


def calc_prep(elm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
    """cg-inversion pre-operation

        This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`

        Args:
            eblm: input data polarisation elm and blm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(elm, np.ndarray) and elm.ndim == 1
    assert Alm.getlmax(elm.size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(elm.size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    elmc = np.atleast_2d(almxfl(elm, ninv_filt.inoise_1_elm, ninv_filt.mmax_len, False))
    elm = ninv_filt.ffi.lensgclm(elmc, ninv_filt.mmax_len, 2, ninv_filt.lmax_sol,ninv_filt.mmax_sol,
                                      backwards=True, out_sht_mode='GRAD_ONLY').squeeze()
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
    return elm