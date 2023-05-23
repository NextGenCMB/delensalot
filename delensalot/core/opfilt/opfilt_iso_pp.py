"""Lenspyx-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import os
import numpy as np
from delensalot.utility.utils_hp import almxfl, Alm
from delensalot.utils import timer, cli
from lenspyx.remapping import utils_geom
from scipy.interpolate import UnivariateSpline as spl
from delensalot.core.opfilt import opfilt_pp

pre_op_dense = None # not implemented
dot_op = opfilt_pp.dot_op
fwd_op = opfilt_pp.fwd_op
apply_fini = opfilt_pp.apply_fini

class alm_filter_nlev:
    def __init__(self, nlev_p:float, transf:np.ndarray, alm_info:tuple, verbose=False, wee=True):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_p: filtering noise level in uK-amin
                    transf: transfer function (beam, pixel window, mutlipole cuts, ...)
                    alm_info: lmax and mmax of unlensed CMB
                    wee: includes EE-like term in QE

                Note:
                    All operations are in harmonic space.
                    Mode exclusions can be implemented setting the transfer fct to zero



        """
        lmax_sol, mmax_sol = alm_info
        lmax_transf = len(transf) - 1


        self.lmax_sol = lmax_sol
        self.mmax_sol = mmax_sol
        self.lmax_len = min(lmax_sol, lmax_transf)
        self.mmax_len = min(mmax_sol, self.lmax_len)

        self.inoise_2  = transf[:self.lmax_len + 1] ** 2 / (nlev_p / 180 / 60 * np.pi) ** 2
        self.inoise_1  = transf[:self.lmax_len + 1] ** 1 / (nlev_p / 180 / 60 * np.pi) ** 2
        self.transf    = transf[:self.lmax_len + 1]

        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')
        self.wee = wee

        self._nthreads = int(os.environ.get('OMP_NUM_THREADS', 1))

    def get_febl(self):
        return np.copy(self.inoise_2), np.copy(self.inoise_2)

    def apply_alm(self, eblm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        assert len(eblm) == 2
        lmax_unl = Alm.getlmax(eblm[0].size, self.mmax_sol)
        assert Alm.getlmax(eblm[0].size, self.mmax_sol) == self.lmax_sol, (lmax_unl, self.lmax_sol)
        assert Alm.getlmax(eblm[1].size, self.mmax_sol) == self.lmax_sol, (lmax_unl, self.lmax_sol)
        almxfl(eblm[0], self.inoise_2, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2, self.mmax_len, inplace=True)

    def get_qlms(self, eblm_dat: np.ndarray or list, eblm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, lmax_qlm, mmax_qlm):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                eblm_wf: Wiener-filtered CMB maps (alm arrays)
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs
                lmax_qlm: maximum l of l,m output
                mmax_qlm: maximum m of l,m output

            All implementation signs are super-weird but end result should be correct...

        """
        assert Alm.getlmax(eblm_dat[0].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(eblm_dat[0].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(eblm_dat[1].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(eblm_dat[1].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(eblm_wf[0].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(eblm_wf[0].size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(eblm_wf[1].size, self.mmax_len) == self.lmax_len, (Alm.getlmax(eblm_wf[1].size, self.mmax_len), self.lmax_len)

        repmap, impmap = self._get_irespmap(eblm_dat, eblm_wf, q_pbgeom)
        Gs, Cs = self._get_gpmap(eblm_wf, 3, q_pbgeom)  # 2 pos.space maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(eblm_wf, 1, q_pbgeom)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self._nthreads, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray or list, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert len(eblm_dat) == 2 and len(eblm_wf) == 2
        ebwf = np.copy(eblm_wf)
        almxfl(ebwf[0], self.transf, self.mmax_len, True)
        almxfl(ebwf[1], self.transf, self.mmax_len, True)
        ebwf[:] = eblm_dat - ebwf
        almxfl(ebwf[0], self.inoise_1 * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.inoise_1 * 0.5,            self.mmax_len, True)
        return q_pbgeom.geom.alm2map_spin(ebwf, 2, self.lmax_len, self.mmax_len, self._nthreads, (-1., 1.))

    def _get_gpmap(self, eblm_wf:np.ndarray or list, spin:int, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert len(eblm_wf) == 2
        assert  Alm.getlmax(eblm_wf[0].size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(eblm_wf[0].size, self.mmax_sol), self.lmax_sol)
        assert  Alm.getlmax(eblm_wf[1].size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(eblm_wf[1].size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(eblm_wf[0].size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        eblm = [almxfl(eblm_wf[0], fl, self.mmax_sol, False), almxfl(eblm_wf[1], fl, self.mmax_sol, False)]
        return q_pbgeom.geom.alm2map_spin(eblm, spin, self.lmax_len, self.mmax_len, self._nthreads, (-1., 1.))

class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        assert len(s_cls['bb']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['bb']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel, ninv_fbl = ninv_filt.get_febl()
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending E transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float), np.log(ninv_fel), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        if len(ninv_fbl) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending N transfer fct from lmax %s to lmax %s"%(len(ninv_fbl)-1, lmax_sol))
            assert np.all(ninv_fbl > 0)
            spl_sq = spl(np.arange(len(ninv_fbl), dtype=float), np.log(ninv_fbl), k=2, ext='extrapolate')
            ninv_fbl = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))

        flmat_ee = cli(s_cls['ee'][:lmax_sol + 1]) + ninv_fel[:lmax_sol + 1]
        flmat_bb = cli(s_cls['bb'][:lmax_sol + 1]) + ninv_fbl[:lmax_sol + 1]

        self.flmat_ee = cli(flmat_ee) * (s_cls['ee'][:lmax_sol +1] > 0.)
        self.flmat_bb = cli(flmat_bb) * (s_cls['bb'][:lmax_sol +1] > 0.)

        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, eblm):
        return self.calc(eblm)

    def calc(self, eblm):
        assert Alm.getsize(self.lmax, self.mmax) == eblm[0].size, (self.lmax, self.mmax, Alm.getlmax(eblm[0].size, self.mmax))
        assert Alm.getsize(self.lmax, self.mmax) == eblm[1].size, (self.lmax, self.mmax, Alm.getlmax(eblm[1].size, self.mmax))
        ret = np.copy(eblm)
        almxfl(ret[0], self.flmat_ee, self.mmax, True)
        almxfl(ret[1], self.flmat_bb, self.mmax, True)
        return ret

def calc_prep(eblm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev, sht_threads:int=4):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            eblm: input data polarisation elm and blm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(eblm, np.ndarray)
    assert Alm.getsize(eblm[0].size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getsize(eblm[0].size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    assert ninv_filt.lmax_len == ninv_filt.lmax_sol
    assert ninv_filt.mmax_len == ninv_filt.mmax_sol
    eblmc = np.copy(eblm)
    almxfl(eblmc[0], ninv_filt.inoise_1 * (s_cls['ee'][:ninv_filt.lmax_len + 1] > 0.), ninv_filt.mmax_len, True)
    almxfl(eblmc[1], ninv_filt.inoise_1 * (s_cls['bb'][:ninv_filt.lmax_len + 1] > 0.), ninv_filt.mmax_len, True)
    return eblmc
