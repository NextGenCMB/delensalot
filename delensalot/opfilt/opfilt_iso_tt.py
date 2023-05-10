"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import os
import numpy as np
from delensalot.utils_hp import almxfl, Alm, alm2cl
from delensalot.utils import timer, cli, clhash
from lenspyx.remapping import utils_geom
from scipy.interpolate import UnivariateSpline as spl

pre_op_dense = None # not implemented


class alm_filter_nlev:
    def __init__(self, nlev_t:float, transf:np.ndarray, alm_info:tuple, verbose=False, rescal=None):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_t: filtering noise level in uK-amin
                    transf: transfer function (beam, pixel window, mutlipole cuts, ...)
                    alm_info: lmax and mmax of unlensed CMB
                    rescal: the WF will search for the sol of Tlm * rescal(l) if set (should not change anything)

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

        self.inoise_2  = transf[:self.lmax_len + 1] ** 2 / (nlev_t / 180 / 60 * np.pi) ** 2
        self.inoise_1  = transf[:self.lmax_len + 1] ** 1 / (nlev_t / 180 / 60 * np.pi) ** 2
        self.transf    = transf[:self.lmax_len + 1]

        self.verbose = verbose
        self.tim = timer(True, prefix='opfilt')

        self._nthreads = int(os.environ.get('OMP_NUM_THREADS', 1))

        if rescal is None:
            rescal = np.ones(lmax_sol + 1, dtype=float)
        assert rescal.size > lmax_sol and np.all(rescal >= 0.)
        self.dorescal = np.any(rescal != 1.)
        self.rescali = cli(rescal)

    def hashdict(self):
        return {'transf': clhash(self.transf), 'inoise2':clhash(self.inoise_2),
                'lmax_sol':self.lmax_sol, 'lmax_len':self.lmax_len}

    def get_ftl(self):
        return np.copy(self.inoise_2)

    def apply_alm(self, tlm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        lmax_unl = Alm.getlmax(tlm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        almxfl(tlm, self.inoise_2 * self.rescali ** 2, self.mmax_len, inplace=True)

    def get_qlms(self, tlm_dat: np.ndarray, tlm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, lmax_qlm, mmax_qlm):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                tlm_dat: input temperature data maps (geom must match that of the filter)
                tlm_wf: Wiener-filtered T CMB map (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs
                lmax_qlm: maximum l of l,m output
                mmax_qlm: maximum m of l,m output

            All implementation signs are super-weird but end result should be correct...

        """
        assert Alm.getlmax(tlm_dat.size, self.mmax_len) == self.lmax_len, (Alm.getlmax(tlm_dat.size, self.mmax_len), self.lmax_len)
        assert Alm.getlmax(tlm_wf.size, self.mmax_len) == self.lmax_len, (Alm.getlmax(tlm_wf.size, self.mmax_len), self.lmax_len)

        d1 = self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(tlm_wf, q_pbgeom)
        G, C = q_pbgeom.geom.map2alm_spin(d1, 1, lmax_qlm, mmax_qlm, self._nthreads, (-1., 1.))
        del d1
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_irestmap(self, tlm_dat:np.ndarray, tlm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert self.lmax_sol == self.lmax_len  and self.mmax_sol == self.mmax_len
        twf = tlm_dat - almxfl(tlm_wf, self.transf, self.mmax_sol, False)
        almxfl(twf, self.inoise_1, self.mmax_len, True)
        return q_pbgeom.geom.alm2map(twf, self.lmax_len, self.mmax_len, self._nthreads, (-1., 1.))

    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert  Alm.getlmax(tlm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        return q_pbgeom.geom.alm2map_spin([almxfl(tlm_wf, fl, self.mmax_sol, False), np.zeros_like(tlm_wf)], 1, self.lmax_len, self.mmax_len, self._nthreads, (-1., 1.))

class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev):
        assert len(s_cls['tt']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['tt']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_ftl = ninv_filt.get_ftl()
        if len(ninv_ftl) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            print("PRE_OP_DIAG: extending T transfer fct from lmax %s to lmax %s"%(len(ninv_ftl)-1, lmax_sol))
            assert np.all(ninv_ftl >= 0.)
            nz = np.where(ninv_ftl > 0.)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))

        flmat_tt = cli(s_cls['tt'][:lmax_sol + 1]) + ninv_ftl[:lmax_sol + 1]
        self.flmat_tt = cli(flmat_tt * ninv_filt.rescali ** 2) * (s_cls['tt'][:lmax_sol + 1] > 0.)

        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, tlm):
        return self.calc(tlm)

    def calc(self, tlm):
        assert Alm.getsize(self.lmax, self.mmax) == tlm.size, (self.lmax, self.mmax, Alm.getlmax(tlm.size, self.mmax))
        return almxfl(tlm, self.flmat_tt, self.mmax, False)

def calc_prep(tlm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            tlm: input data temperature tlm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(tlm, np.ndarray)
    assert Alm.getsize(tlm.size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getsize(tlm.size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    assert ninv_filt.lmax_len == ninv_filt.lmax_sol
    assert ninv_filt.mmax_len == ninv_filt.mmax_sol
    return almxfl(tlm, ninv_filt.rescali * ninv_filt.inoise_1 * (s_cls['tt'][:ninv_filt.lmax_len + 1] > 0.), ninv_filt.mmax_len, False)


class dot_op:
    def __init__(self, lmax:int, mmax:int or None):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


            Note: here by defaults the scalar product is sum(Dl) instead of sum(Cl) !

        """
        if mmax is None or mmax < 0:
            mmax = lmax
        self.lmax = lmax
        self.mmax = min(mmax, lmax)
        self.scal = np.arange(lmax + 1) * np.arange(1, lmax + 2) * (2 * np.arange(self.lmax + 1) + 1) / (2. * np.pi)
        #FIXME
        self.scal = (2 * np.arange(self.lmax + 1) + 1)
        print("Cl norm!")

    def __call__(self, tlm1, tlm2):
        assert tlm1.size == Alm.getsize(self.lmax, self.mmax), (tlm1.size, Alm.getsize(self.lmax, self.mmax))
        assert tlm2.size == Alm.getsize(self.lmax, self.mmax), (tlm2.size, Alm.getsize(self.lmax, self.mmax))
        return np.sum(alm2cl(tlm1, tlm2, self.lmax, self.mmax, None) * self.scal)


class fwd_op:
    """Forward operation for temperature-only


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev):
        self.icls = {'tt': cli(s_cls['tt'][:ninv_filt.lmax_sol + 1]) * ninv_filt.rescali ** 2}
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'icltt': clhash(self.icls['tt']), 'ninv_filt': self.ninv_filt.hashdict()}

    def __call__(self, tlm):
        return self.calc(tlm)

    def calc(self, tlm):
        nlm = np.copy(tlm)
        self.ninv_filt.apply_alm(nlm)
        #TODO: in principle the icls > 0 should already be ok?
        nlm = almxfl(nlm + almxfl(tlm, self.icls['tt'], self.mmax_sol, False), self.icls['tt'] > 0., self.mmax_sol, False)
        return nlm

def apply_fini(alm, s_cls, ninv_filt:alm_filter_nlev):
    """ This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB.

    """
    if ninv_filt.dorescal:
        almxfl(alm, ninv_filt.rescali, ninv_filt.mmax_sol, inplace=True)
