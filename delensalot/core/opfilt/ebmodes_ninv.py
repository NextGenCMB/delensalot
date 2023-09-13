"""Module for harmonic curl modes template deprojection of a spin-2 field (e.g. B-modes)


"""
import os, sys
import numpy as np
import healpy as hp

import logging
log = logging.getLogger(__name__)


from lenspyx.remapping import utils_geom
from delensalot.core import mpi
from delensalot.utils import enumerate_progress, read_map
from delensalot.utility.utils_hp import Alm

def lmax2nlm(lmax):
    """ returns the length of the complex alm array required for maximum multipole lmax. """

    return (lmax + 1) * (lmax + 2) // 2


def nlm2lmax(nlm):

    """ returns the lmax for an array of alm with length nlm. """
    lmax = int(np.floor(np.sqrt(2 * nlm) - 1))
    assert ((lmax + 2) * (lmax + 1) // 2 == nlm)

    return lmax


def rlm2alm(rlm):
    """ converts 'real harmonic' coefficients rlm to complex alm. """

    lmax = int(np.sqrt(len(rlm)) - 1)
    assert ((lmax + 1) ** 2 == len(rlm))

    alm = np.zeros(lmax2nlm(lmax), dtype=complex)
    ls = np.arange(0, lmax + 1, dtype=int)
    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in range(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1.j * rlm[l2s[m:] + 2 * m + 0]) * ir2

    return alm


def alm2rlm(alm):
    """ converts a complex alm to 'real harmonic' coefficients rlm. """

    lmax = nlm2lmax(len(alm))
    rlm = np.zeros((lmax + 1) ** 2)

    ls = np.arange(0, lmax + 1)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in range(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].imag * rt2

    return rlm


class template_ebfilt(object):
    def __init__(self, lmax_marg:int, geom:utils_geom.Geom, sht_threads:int, _lib_dir=None):
        """
        Class for building tniti matrix.
        Here all E and B-modes up to lmax are set to infinite noise

            Args:
                lmax_marg: all B-mulitpoles up to and inclusive of lmax are marginalized
                geom: lenspyx geometry of SHTs
                sht_threads: number of OMP threads for SHTs
                _lib_dir: some stuff might be cached there in some cases


        """
        assert lmax_marg >= 2, lmax_marg
        self.lmax = lmax_marg
        self.nmodes = 2  * ( int((lmax_marg + 1) * lmax_marg + lmax_marg + 1 - 4) )

        self.geom = geom
        self.npix = geom.npix()

        self.lib_dir = None
        if _lib_dir is not None and lmax_marg > 10: #just to avoid problems if user does not understand what is doing...
            if not os.path.exists(_lib_dir):
                os.makedirs(_lib_dir)
            self.lib_dir = _lib_dir

        self.sht_threads = sht_threads


    def hashdict(self):

        return {'lmax':self.lmax,}


    @staticmethod
    def get_nmodes(lmax):

        assert lmax >= 2, lmax
        return 2 * ((lmax + 1) * lmax + lmax + 1 - 4)


    def get_modelmax(self, mode):

        assert mode >= 0 and mode < self.nmodes, mode
        nmodes = 0
        l = -1
        while nmodes - 1 < (mode % (self.nmodes // 2) ) + 4:
            l += 1
            nmodes += 2 * l + 1
        return l


    @staticmethod
    def _rlm2eblm(rlm):
        assert rlm.size % 2 == 0
        rlm_s = rlm.size // 2
        elm = rlm2alm(np.concatenate([np.zeros(4), rlm[rlm_s * 0: rlm_s * 1]]))
        blm = rlm2alm(np.concatenate([np.zeros(4), rlm[rlm_s * 1: rlm_s * 2]]))
        return np.stack([elm, blm])


    @staticmethod
    def _eblm2rlm(eblm):
        assert len(eblm) == 2
        return np.concatenate([alm2rlm(eblm[0])[4:], alm2rlm(eblm[1])[4:]])


    def apply_qumode(self, qumap, mode):
        assert mode < self.nmodes, (mode, self.nmodes)
        assert len(qumap) == 2
        isE = mode < (self.nmodes // 2) # Need to adapt index if B-mode
        tcoeffs = np.zeros(self.get_nmodes(self.get_modelmax(mode)), dtype=float)
        tcoeffs[mode if isE else tcoeffs.size // 2 + (mode % (self.nmodes//2))] = 1.0
        self.apply_qu(qumap, tcoeffs)


    def apply_qu(self, qumap, coeffs):  # RbQ  * Q or  RbU * U
        assert len(qumap) == 2
        assert (len(coeffs) <= self.nmodes)
        assert qumap[0].size == self.npix, (self.npix, qumap[0].size)
        assert qumap[1].size == self.npix, (self.npix, qumap[1].size)
        eblm = self._rlm2eblm(coeffs)
        this_lmax = Alm.getlmax(eblm[0].size, -1)
        q, u = self.geom.synthesis(eblm, 2, this_lmax, this_lmax, self.sht_threads)
        qumap[0] *= q
        qumap[1] *= u


    def accum(self, qumap, coeffs):
        """Forward template operation

            Turns the input real harmonic *coeffs* to blm and send to Q, U.
            This plus-adds the input *qumap*

        """
        assert (len(coeffs) <= self.nmodes)
        eblm = self._rlm2eblm(coeffs)
        this_lmax = Alm.getlmax(eblm[0].size, -1)
        q, u = self.geom.synthesis(eblm, 2, this_lmax, this_lmax, self.sht_threads)
        qumap[0] += q
        qumap[1] += u


    def dot(self, qumap):
        """Backward template operation.

            Turns the input qu maps into the real harmonic coefficient B-modes up to lmax.
            This includes a factor npix / 4pi, as the transpose differs from the inverse by that factor

        """
        
        assert len(qumap) == 2
        assert qumap[0].size == self.npix and qumap[1].size == self.npix, ' '.join([str(qumap[1].shape), str(self.npix)])
        eblm = self.geom.adjoint_synthesis(qumap, 2, self.lmax, self.lmax, self.sht_threads, apply_weights=False)
        return self._eblm2rlm(eblm)


    def build_tnit(self, NiQQ_NiUU_NiQU):
        """Return the nmodes x nmodes matrix (T^t N^{-1} T )_{bl bl'}'

            For unit inverse noise matrices on the full-sky this is a diagonal matrix
            with diagonal :math:`N_{\rm pix} / (4\pi)`.

            If input inverse noise matrices are the inverse pixel noise in uK, this give 1/noise level ** 2 in uK-rad squared

        """
        if self.lib_dir is not None:
            return self._build_tnit('')
        if NiQQ_NiUU_NiQU.shape[0] == 3: #Here, QQ and UU may be different, but NiQU negligible
            NiQQ, NiUU, NiQU = NiQQ_NiUU_NiQU
            assert NiQU is None
        else: #Here, we assume that NiQQ = NiUU, and NiQU is negligible
            NiQQ, NiUU, NiQU = NiQQ_NiUU_NiQU, NiQQ_NiUU_NiQU, None
        tnit = np.zeros((self.nmodes, self.nmodes), dtype=float)
        for i, a in enumerate_progress(range(self.nmodes),
                                             False * 'filling template matrix'):  # Starts at ell = 2
            _NiQ = np.copy(NiQQ)  # Building Ni_{QX} R_bX
            _NiU = np.copy(NiUU)  # Building Ni_{UX} R_bX
            self.apply_qumode([_NiQ, _NiU], a)
            tnit[:, a] = self.dot([_NiQ, _NiU])
            tnit[a, :] = tnit[:, a]

        return tnit


    def _build_tnit(self, prefix=''):
        tnit = np.zeros((self.nmodes, self.nmodes), dtype=float)
        for i, a in enumerate_progress(range(self.nmodes), label='collecting Pmat rows'):
            fname = os.path.join(self.lib_dir, 'rows', prefix + 'row%05d.npy'%a)
            assert os.path.exists(fname), fname
            tnit[:, a]  = np.load(fname)
            tnit[a, :] = tnit[:, a]

        return tnit


    def _get_rows_mpi(self, NiQQ_NiUU_NiQU, prefix):
        """Produces and save all rows of the matrix for large matriz sizes

        """
        assert self.lib_dir is not None, 'cant do this without a lib_dir'
        if NiQQ_NiUU_NiQU.shape[0] == 3: #Here, QQ and UU may be different, but NiQU negligible
            NiQQ, NiUU, NiQU = NiQQ_NiUU_NiQU
            assert NiQU is None
        else: #Here, we assume that NiQQ = NiUU, and NiQU is negligible
            NiQQ, NiUU, NiQU = NiQQ_NiUU_NiQU[0], NiQQ_NiUU_NiQU[0], None
        assert self.nmodes <= 99999, 'ops, naming in the lines below'
        log.info("number of rows for tnit: {}".format(self.nmodes))
        if not os.path.exists(os.path.join(self.lib_dir, 'rows')):
            os.makedirs(os.path.join(self.lib_dir, 'rows'))
        for ai, a in enumerate_progress(range(self.nmodes)[mpi.rank::mpi.size], label='Calculating Pmat row'):
            fname = os.path.join(self.lib_dir, 'rows', prefix + 'row%05d.npy'%a)
            if not os.path.exists(fname):
                _NiQ = np.copy(NiQQ)  # Building Ni_{QX} R_bX
                _NiU = np.copy(NiUU)  # Building Ni_{UX} R_bX
                self.apply_qumode([_NiQ, _NiU], a)
                np.save(fname, self.dot([_NiQ, _NiU]))
                del _NiQ, _NiU


class template_dense(template_ebfilt):
    """
    Class for loading existing tniti matrix. Cannot be used for building it.
    """
    def __init__(self, lmax_marg:int, geom:utils_geom.Geom, sht_threads:int, _lib_dir=None, rescal=1.):
        assert os.path.exists(os.path.join(_lib_dir, 'tniti.npy')), os.path.join(_lib_dir, 'tniti.npy')
        super().__init__(lmax_marg, geom, sht_threads, _lib_dir=_lib_dir)
        self.rescal = rescal
        self._tniti = None # will load this when needed

    def hashdict(self):
        return {'lmax':self.lmax, 'rescal':self.rescal}

    def tniti(self):
        if self._tniti is None:
            self._tniti = read_map(os.path.join(self.lib_dir, 'tniti.npy')) * self.rescal
            log.debug("reading " +os.path.join(self.lib_dir, 'tniti.npy') )
            log.debug("Rescaling it with %.5f"%self.rescal)
        return self._tniti