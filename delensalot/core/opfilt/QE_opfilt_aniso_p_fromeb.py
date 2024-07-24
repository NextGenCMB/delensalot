"""Lenspyx-geometry based inverse-variance filter, without any lensing remapping

    Same as QE_opfilt_aniso_p but inputs are E and B real-space maps instead of QU

"""
import logging
log = logging.getLogger(__name__)

import numpy as np

from lenspyx.remapping import utils_geom

from delensalot.utils import timer
from delensalot.utility.utils_hp import almxfl, Alm
from delensalot.core.opfilt import bmodes_ninv as bni
from delensalot.core.opfilt import QE_opfilt_aniso_p


apply_fini = QE_opfilt_aniso_p.apply_fini
pre_op_dense = None
pre_op_diag = QE_opfilt_aniso_p.pre_op_diag
dot_op = QE_opfilt_aniso_p.dot_op
fwd_op = QE_opfilt_aniso_p.fwd_op


class alm_filter_ninv(QE_opfilt_aniso_p.alm_filter_ninv):
    def __init__(self, ninv_geom:utils_geom.Geom, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,
                 transf_b:np.ndarray or None=None,
                 tpl:bni.template_dense or None=None, verbose=False):
        r"""CMB inverse-variance and Wiener filtering instance, to use for cg-inversion

            Args:
                ninv_geom: lenspyx geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (itself can be (a list of) string, or array, or ...)
                        (noise maps are now E and B noise maps)
                transf: CMB transfer function (assumed to be the same in E and B)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for lenspyx SHTs
                verbose: some printout if set, defaults to False


        """
        transf_elm = transf
        transf_blm = transf_b if transf_b is not None else transf
        assert transf_blm.size == transf_elm.size, 'check if not same size OK'

        self.n_inv = ninv
        self.transf_elm = transf_elm
        self.transf_blm = transf_blm
        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = max(len(transf), len(transf_blm)) - 1
        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)
        self.lmax_sol = lmax_unl
        self.mmax_sol = min(lmax_unl, mmax_unl)

        self.sht_threads = sht_threads
        self.ninv_geom = ninv_geom

        self.verbose=verbose

        self._nlevp = None
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed



    def apply_alm(self, eblm:np.ndarray):
        """Applies operator B^T N^{-1} B

        """
        assert self.lmax_sol == self.lmax_len, (self.lmax_sol, self.lmax_len) # not implemented wo lensing
        assert self.mmax_sol == self.mmax_len, (self.mmax_sol, self.mmax_len)
        assert len(self.n_inv) == 1, 'assuming E and B noise the same here'
        tim = timer(True, prefix='opfilt_pp')
        lmax_unl = Alm.getlmax(eblm[0].size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')
        ma = np.empty((1, self.ninv_geom.npix()), dtype=float)
        for sli in [slice(0, 1), slice(1, 2)]:
            self.ninv_geom.synthesis(eblm[sli], 0, self.lmax_len, self.mmax_len, self.sht_threads, map=ma)
            tim.add('alm2map spin 0 lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.ninv_geom.theta.size))
            self.ninv_geom.adjoint_synthesis(ma * self.n_inv[0], 0, self.lmax_sol, self.mmax_sol, self.sht_threads,
                                                   apply_weights=False, alm=eblm[sli])
            tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.ninv_geom.theta.size))
        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')
        if self.verbose:
            print(tim)

    def apply_map(self, qumap):
        """Applies pixel inverse-noise variance maps


        """
        assert 0, 'not used for this instance'

    def get_qlms(self, qudat: np.ndarray or list, eblm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, lmax_qlm, mmax_qlm):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                eblm_wf: Wiener-filtered CMB maps (alm arrays)
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs
                lmax_qlm: maximum multipole of output
                mmax_qlm: maximum m of lm output

        """
        assert 0, 'fix this if you want this (QU -> EB)'
        assert len(qudat) == 2 and len(eblm_wf)
        assert (qudat[0].size == self.geom_.npix()) and (qudat[0].size == qudat[1].size)

        repmap, impmap = self._get_irespmap(qudat, eblm_wf, q_pbgeom)
        Gs, Cs = self._get_gpmap(eblm_wf, 3, q_pbgeom)  # 2 pos.space maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(eblm_wf, 1, q_pbgeom)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.sht_threads, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_gpmap(self, eblm_wf:np.ndarray or list, spin:int, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert 0, 'fix this if you want this (QU -> EB)'
        assert len(eblm_wf) == 2
        assert  Alm.getlmax(eblm_wf[0].size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(eblm_wf[0].size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(eblm_wf[0].size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        eblm = [almxfl(eblm_wf[0], fl, self.mmax_sol, False), almxfl(eblm_wf[1], fl, self.mmax_sol, False)]
        return q_pbgeom.geom.synthesis(eblm, spin, lmax, self.mmax_sol, self.sht_threads)

    def _get_irespmap(self, qu_dat:np.ndarray, eblm_wf:np.ndarray or list, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE

                :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert 0, 'fix this if you want this (QU -> EB)'
        assert len(qu_dat) == 2 and len(eblm_wf) == 2, (len(eblm_wf), len(qu_dat))
        ebwf = np.copy(eblm_wf)
        almxfl(ebwf[0], self.transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], self.transf_blm, self.mmax_len, True)
        qu = qu_dat - self.ninv_geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads)
        self.apply_map(qu)
        self.ninv_geom.adjoint_synthesis(qu, 2, self.lmax_sol, self.mmax_sol, self.sht_threads,
                                                   apply_weights=False, alm=ebwf)
        almxfl(ebwf[0], self.transf_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.transf_blm * 0.5, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads)


def calc_prep(ebmaps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv, sht_threads:int=0):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            ebmaps: input polarisation E and B maps
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance
            sht_threads: number of threads for lenspyx/ducc SHTs


    """
    assert ninv_filt.lmax_sol == ninv_filt.lmax_len, (ninv_filt.lmax_sol, ninv_filt.lmax_len)  # not implemented wo lensing
    assert ninv_filt.mmax_sol == ninv_filt.mmax_len, (ninv_filt.mmax_sol, ninv_filt.mmax_len)
    assert len(ninv_filt.n_inv) == 1, 'Assuming E and B noise the same here'
    eblm = np.empty((2, ninv_filt.ninv_geom.npix()), dtype=complex)
    for sli in [slice(0, 1), slice(1, 2)]:
        ninv_filt.ninv_geom.adjoint_synthesis(ebmaps[sli] * ninv_filt.n_inv[0], 0,
                        ninv_filt.lmax_len,  ninv_filt.mmax_len, sht_threads, apply_weights=False, alm=eblm[sli])
    lmax_tr = ninv_filt.lmax_len
    almxfl(eblm[0], ninv_filt.transf_elm * (s_cls['ee'][:lmax_tr+1] > 0.), ninv_filt.mmax_len, inplace=True)
    almxfl(eblm[1], ninv_filt.transf_blm * (s_cls['bb'][:lmax_tr+1] > 0.), ninv_filt.mmax_len, inplace=True)
    return eblm

