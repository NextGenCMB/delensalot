"""Lenspyx-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import time
import numpy as np

from plancklens.utils import cli

from lenspyx import remapping
from lenspyx.remapping import utils_geom

from delensalot.utility.utils_hp import almxfl, Alm, synalm
from delensalot.utils import timer, clhash
from delensalot.core.opfilt.QE import opfilt_iso_t
from delensalot.core.opfilt import opfilt_base


fwd_op = opfilt_iso_t.fwd_op
dot_op = opfilt_iso_t.dot_op
pre_op_diag = opfilt_iso_t.pre_op_diag
pre_op_dense = None # not implemented
apply_fini = opfilt_iso_t.apply_fini

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class alm_filter_nlev_wl(opfilt_base.alm_filter_wl):
    def __init__(self, nlev_t:float or np.ndarray, ffi:remapping.deflection, transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple, verbose=False, rescal=None):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_t: filtering noise level in uK-amin (can be a function of multipole)
                    ffi: delensalot deflection instance
                    transf: transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)
                    rescal: the WF will search for the sol of Tlm * rescal(l) if set (should not change anything)

                Note:
                    All operations are in harmonic space.
                    Mode exclusions can be implemented setting the transfer fct to zero


        """
        lmax_sol, mmax_sol = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = len(transf) - 1

        super(alm_filter_nlev_wl, self).__init__(lmax_sol, mmax_sol, ffi)

        self.lmax_len = min(lmax_len, lmax_transf)
        self.mmax_len = min(mmax_len, self.lmax_len)

        nlev_tlm = _extend_cl(nlev_t, lmax_len)

        self.inoise_2 = _extend_cl(transf ** 2, lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1 = _extend_cl(transf ** 1, lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2
        self.transf   = _extend_cl(transf, lmax_len)

        if rescal is None:
            rescal = np.ones(lmax_sol + 1, dtype=float)
        assert rescal.size > lmax_sol and np.all(rescal >= 0.)
        self.dorescal = np.any(rescal != 1.)
        self.rescali = cli(rescal)

        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')


    def hashdict(self):
        return {'transf': clhash(self.transf), 'inoise2':clhash(self.inoise_2),
                'lmax_sol':self.lmax_sol, 'lmax_len':self.lmax_len, 'ffi':self.ffi.hashdict()}

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def get_ftl(self, len=False):
        if len:#FIXME
            print("Building fancy inverse noise cls!")
            from plancklens import lensedcls
            from lenspyx.utils_hp import alm2cl
            from scipy.interpolate import UnivariateSpline as spl
            npts = (self.lmax_sol + self.lmax_len) // 2 + 1 + 2000
            lib = lensedcls.lensed2pcf(npts, alm2cl(self.ffi.dlm, self.ffi.dlm, self.ffi.lmax_dlm, self.ffi.mmax_dlm, self.ffi.lmax_dlm))
            ls = np.unique(np.int_(np.linspace(1, self.lmax_sol, 300)))
            ls = np.insert(ls, 0, 0)
            if self.lmax_sol not in ls:
                ls = np.insert(ls, ls.size, self.lmax_sol)
            ret = spl(ls, lib.get_leninoisecls(self.get_ftl(len=False), ls), k=1, s=0, ext='zeros')(np.arange(self.lmax_sol+1))
            print(ret[1:10] / self.get_ftl(len=False)[1:10])
            print(ret[self.lmax_len-10:self.lmax_len+1] / self.get_ftl(len=False)[self.lmax_len-10:self.lmax_len + 1])

            return ret
        return np.copy(self.inoise_2)

    def apply_alm(self, tlm:np.ndarray):
        """Applies operator Y^T N^{-1} Y

        """
        lmax_unl = Alm.getlmax(tlm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        if self.dorescal:
            almxfl(tlm, self.rescali, self.mmax_sol, True)
        tlmc = self.ffi.lensgclm(tlm, self.mmax_sol, 0, self.lmax_len, self.mmax_len)
        almxfl(tlmc, self.inoise_2, self.mmax_len, True)
        tlm[:] = self.ffi.lensgclm(tlmc, self.mmax_len, 0, self.lmax_sol, self.mmax_sol, backwards=True)
        if self.dorescal:
            almxfl(tlm, self.rescali, self.mmax_sol, True)
        # TODO: should add here the projection into cls > 0

    def get_qlms(self, tlm_dat: np.ndarray, tlm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, alm_wf_leg2=None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                tlm_dat: input temperature data maps (geom must match that of the filter)
                tlm_wf: Wiener-filtered T CMB map (alm arrays)
                alm_wf_leg2: Gradient leg Wiener-filtered T CMB map (alm arrays), if different from ivf leg
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert Alm.getlmax(tlm_wf.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        if alm_wf_leg2 is None:
            d1 = self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(tlm_wf, q_pbgeom)
        else:
            assert Alm.getlmax(alm_wf_leg2.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(alm_wf_leg2.size, self.mmax_sol), self.lmax_sol)
            d1 = self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(alm_wf_leg2, q_pbgeom)
        G, C = q_pbgeom.geom.map2alm_spin(d1, 1, self.ffi.lmax_dlm, self.ffi.mmax_dlm, self.ffi.sht_tr, (-1., 1.))
        del d1
        fl = - np.sqrt(np.arange(self.ffi.lmax_dlm + 1, dtype=float) * np.arange(1, self.ffi.lmax_dlm + 2))
        almxfl(G, fl, self.ffi.mmax_dlm, True)
        almxfl(C, fl, self.ffi.mmax_dlm, True)
        return G, C

    def get_qlms_mf(self, mfkey, q_pbgeom:utils_geom.pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        if mfkey in [1]: # This should be B^t x, D dC D^t B^t Covi x, x random phases in alm space
            if phas is None:
                phas = synalm(np.ones(self.lmax_len + 1, dtype=float), self.lmax_len, self.mmax_len)
            assert Alm.getlmax(phas.size, self.mmax_len) == self.lmax_len

            soltn = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(soltn, phas, dot_op=self.dot_op())

            almxfl(phas,  self.transf, self.mmax_len, True)
            #
            assert 0, ' finish this'
            repmap, impmap = q_pbgeom.geom.alm2map_spin(phas, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))

            Gs, Cs = self._get_gpmap([soltn, np.zeros_like(soltn)], 3, q_pbgeom)  # 2 pos.space maps
            GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
            Gs, Cs = self._get_gpmap([soltn, np.zeros_like(soltn)], 1, q_pbgeom)
            GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
            del repmap, impmap, Gs, Cs
        elif mfkey in [0]: # standard gQE, quite inefficient but simple
            assert 0, 'not implemented'

        else:
            assert 0, mfkey + ' not implemented'
        lmax_qlm = self.ffi.lmax_dlm
        mmax_qlm = self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_irestmap(self, tlm_dat:np.ndarray, tlm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        twf = tlm_dat - almxfl(self.ffi.lensgclm(tlm_wf, self.mmax_sol, 0, self.lmax_len, self.mmax_len), self.transf, self.mmax_len, False)
        almxfl(twf, self.inoise_1, self.mmax_len, True)
        return q_pbgeom.geom.alm2map(twf, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))

    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert Alm.getlmax(tlm_wf.size, self.mmax_sol) == self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap([almxfl(tlm_wf, fl, self.mmax_sol, False), np.zeros_like(tlm_wf)], self.mmax_sol, 1, False)


def calc_prep(tlm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev_wl, sht_threads:int=4):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            tlm: input data temperature tlm
            s_cls: CMB spectra dictionary (here only 'tt' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(tlm, np.ndarray)
    assert Alm.getlmax(tlm.size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(tlm.size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    tlmc = almxfl(tlm, ninv_filt.inoise_1, ninv_filt.mmax_len, False)
    tlmc = ninv_filt.ffi.lensgclm(tlmc, ninv_filt.mmax_len, 0, ninv_filt.lmax_sol, ninv_filt.mmax_sol, backwards=True)
    almxfl(tlmc, ninv_filt.rescali * (s_cls['tt'][:ninv_filt.lmax_sol + 1] > 0.), ninv_filt.mmax_sol, True)
    return tlmc
