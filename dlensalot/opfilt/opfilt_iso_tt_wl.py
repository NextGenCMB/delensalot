"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import numpy as np
from dlensalot.utils_hp import almxfl, Alm
from dlensalot.utils import timer, clhash
from dlensalot import utils_scarf, remapping
from dlensalot.opfilt import opfilt_iso_tt, opfilt_base
from plancklens.utils import cli

fwd_op = opfilt_iso_tt.fwd_op
dot_op = opfilt_iso_tt.dot_op
pre_op_diag = opfilt_iso_tt.pre_op_diag
pre_op_dense = None # not implemented

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass

class alm_filter_nlev_wl(opfilt_base.scarf_alm_filter_wl):
    def __init__(self, nlev_t:float or np.ndarray, ffi:remapping.deflection, transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple, verbose=False):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_t: filtering noise level in uK-amin (can be a function of multipole)
                    ffi: lenscarf deflection instance
                    transf: transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)


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

        self.inoise_2  = _extend_cl(transf ** 2, lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1  = _extend_cl(transf ** 1 ,lmax_len) * cli(nlev_tlm ** 2) * (180 * 60 / np.pi) ** 2
        self.transf    = _extend_cl(transf, lmax_len)


        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')


    def hashdict(self):
        return {'transf': clhash(self.transf), 'inoise2':clhash(self.inoise_2),
                'lmax_sol':self.lmax_sol, 'lmax_len':self.lmax_len, 'ffi':self.ffi.hashdict()}

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def get_ftl(self):
        return np.copy(self.inoise_2)

    def apply_alm(self, tlm:np.ndarray):
        """Applies operator Y^T N^{-1} Y

        """
        lmax_unl = Alm.getlmax(tlm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        tlmc = self.ffi.lensgclm(tlm, self.mmax_sol, 0, self.lmax_len, self.mmax_len)
        almxfl(tlmc, self.inoise_2, self.mmax_len, inplace=True)
        tlm[:] = self.ffi.lensgclm(tlmc, self.mmax_len, 0, self.lmax_sol, self.mmax_sol, backwards=True)

    def get_qlms(self, tlm_dat: np.ndarray, tlm_wf: np.ndarray, q_pbgeom: utils_scarf.pbdGeometry, alm_wf_leg2=None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                tlm_dat: input temperature data maps (geom must match that of the filter)
                tlm_wf: Wiener-filtered T CMB map (alm arrays)
                alm_wf_leg2: Gradient leg Wiener-filtered T CMB map (alm arrays), if different from ivf leg
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert Alm.getlmax(tlm_dat.size, self.mmax_len) == self.lmax_len, (Alm.getlmax(tlm_dat.size, self.mmax_len), self.lmax_len)
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

    def _get_irestmap(self, tlm_dat:np.ndarray, tlm_wf:np.ndarray, q_pbgeom:utils_scarf.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        twf = tlm_dat - almxfl(self.ffi.lensgclm(tlm_wf, self.mmax_sol, 0, self.lmax_len, self.mmax_len), self.transf, self.mmax_len, False)
        almxfl(twf, self.inoise_1, self.mmax_len, True)
        return q_pbgeom.geom.alm2map(twf, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))

    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom:utils_scarf.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert  Alm.getlmax(tlm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap([almxfl(tlm_wf, fl, self.mmax_sol, False), np.zeros_like(tlm_wf)], self.mmax_sol, 1, False)


def calc_prep(tlm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            tlm: input data temperature tlm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(tlm, np.ndarray)
    assert Alm.getlmax(tlm.size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(tlm.size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    tlmc = np.copy(tlm)
    almxfl(tlmc, ninv_filt.inoise_1, ninv_filt.mmax_len, True)
    tlmc = ninv_filt.ffi.lensgclm(tlmc, ninv_filt.mmax_len, 0, ninv_filt.lmax_sol,ninv_filt.mmax_sol, backwards=True)
    almxfl(tlmc, s_cls['tt'] > 0., ninv_filt.mmax_sol, True)
    return tlmc