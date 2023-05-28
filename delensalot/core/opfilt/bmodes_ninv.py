"""Module for harmonic curl modes template deprojection of a spin-2 field (e.g. B-modes)


"""
import os, sys
import numpy as np
import healpy as hp

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from plancklens import utils
from plancklens.qcinv import opfilt_pp

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


class template_bfilt(object):
    def __init__(self, lmax_marg:int, geom:utils_geom.Geom, sht_threads:int, _lib_dir=None):
        """
        Class for building tniti matrix.
        Here all B-modes up to lmax are set to infinite noise

            Args:
                lmax_marg: all B-mulitpoles up to and inclusive of lmax are marginalized
                geom: lenspyx geometry of SHTs
                sht_threads: number of OMP threads for SHTs
                _lib_dir: some stuff might be cached there in some cases


        """
        assert lmax_marg >= 2, lmax_marg
        self.lmax = lmax_marg
        self.nmodes = int((lmax_marg + 1) * lmax_marg + lmax_marg + 1 - 4)
        if not np.all(geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            log.info('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
            # old signature: "nrings"_a, "nph"_a, "ofs"_a, "stride"_a, "phi0"_a, "theta"_a, "wgt"_a
            # nr = geom.get_nrings()
            # old geom: geom_ = us.Geometry(nr, geom.nph.copy(), geom.ofs.copy(), 1, geom.phi0.copy(), geom.theta.copy(), np.ones(nr, dtype=float))
            # new signature: (self, thet:, phi0, nphi, ringstart, w)
            # new geom_
            geom_ = utils_geom.Geom(geom.theta.copy(), geom.phi0.copy(), geom.nph.copy(), geom.ofs.copy(), np.ones(len(geom.ofs), dtype=float))
        else:
            geom_ = geom
            # Does not seem to work without the 'copy'
        self.geom = geom_
        self.npix = geom_.npix()

        self.lib_dir = None
        if _lib_dir is not None and lmax_marg > 10: #just to avoid problems if user does not understand what is doing...
            if not os.path.exists(_lib_dir):
                os.makedirs(_lib_dir)
            self.lib_dir = _lib_dir

        sht_threads = sht_threads


    def hashdict(self):

        return {'lmax':self.lmax,}


    @staticmethod
    def get_nmodes(lmax):

        assert lmax >= 2, lmax
        return (lmax + 1) * lmax + lmax + 1 - 4


    @staticmethod
    def get_modelmax(mode):

        assert mode >= 0, mode
        nmodes = 0
        l = -1
        while nmodes - 1 < mode + 4:
            l += 1
            nmodes += 2 * l + 1

        return l


    @staticmethod
    def _rlm2blm(rlm):

        return rlm2alm(np.concatenate([np.zeros(4), rlm]))


    @staticmethod
    def _blm2rlm(blm):

        return alm2rlm(blm)[4:]


    def apply_qumode(self, qumap, mode):
        assert mode < self.nmodes, (mode, self.nmodes)
        assert len(qumap) == 2
        tcoeffs = np.zeros(self.get_nmodes(self.get_modelmax(mode)), dtype=float)
        tcoeffs[mode] = 1.0
        self.apply_qu(qumap, tcoeffs)


    def apply_qu(self, qumap, coeffs):  # RbQ  * Q or  RbU * U
        assert len(qumap) == 2
        assert (len(coeffs) <= self.nmodes)
        assert qumap[0].size == self.npix, (self.npix, qumap[0].size)
        assert qumap[1].size == self.npix, (self.npix, qumap[1].size)
        blm = self._rlm2blm(coeffs)
        elm = np.zeros_like(blm)

        this_lmax = Alm.getlmax(blm.size, -1)
        q, u = self.geom.alm2map_spin([elm, blm], 2, this_lmax, this_lmax. self.sht_threads)
        qumap[0] *= q
        qumap[1] *= u


    def accum(self, qumap, coeffs):
        """Forward template operation

            Turns the input real harmonic *coeffs* to blm and send to Q, U.
            This plus-adds the input *qumap*

        """
        assert (len(coeffs) <= self.nmodes)
        blm = self._rlm2blm(coeffs)
        elm = np.zeros_like(blm)
        this_lmax = Alm.getlmax(blm.size, -1)
        q, u = self.geom.alm2map_spin([elm, blm], 2, this_lmax, this_lmax, self.sht_threads)
        qumap[0] += q
        qumap[1] += u


    def dot(self, qumap):
        """Backward template operation.

            Turns the input qu maps into the real harmonic coefficient B-modes up to lmax.
            This includes a factor npix / 4pi, as the transpose differs from the inverse by that factor

        """
        
        assert len(qumap) == 2
        assert qumap[0].size == self.npix and qumap[1].size == self.npix, ' '.join([str(qumap[1].shape), str(self.npix)])
        blm = self.geom.map2alm_spin(qumap, 2, self.lmax, self.lmax, self.sht_threads)[1]

        return self._blm2rlm(blm) # Units weight transform


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


class template_dense(template_bfilt):
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
            log.info("reading " +os.path.join(self.lib_dir, 'tniti.npy') )
            log.info("Rescaling it with %.5f"%self.rescal)
        return self._tniti


# TODO this is merely a copy paste of the itercurv version. Replace with delensalot.bmodes_ninv.template_dense()
class eblm_filter_ninv(opfilt_pp.alm_filter_ninv):
    """Identical to *plancklens* polarization filter, but adding the $B$-marginalization possibility

        Note:
            n_inv is inverse pixel variance map (no volume factors or units)

            set bmarg_lib_dir only to calculate the rows of the template with mpi later on, for very large bmarg_lmax

            blm_range is only a way to approximately project out some modes,
            you dont want to mix this with bmarg_lmax which is exact template marginalisation

    """
    def __init__(self, geom, n_inv, b_transf, lmax_marg=0, zbounds=(-1., 1.), blm_range=(2, np.inf), _bmarg_lib_dir=None, _bmarg_rescal=1., sht_threads=8):
        super(eblm_filter_ninv, self).__init__(n_inv, b_transf)
        self.n_inv = self.get_ninv()
        self.nside = hp.npix2nside(len(self.n_inv[0]))
        if not ( (blm_range[0] <= 2) and (blm_range[1] >= (3 * self.nside - 1)) ):
            assert len(self.templates)  == 0, 'templates-cuts mixing not implemented'

        self.blm_range = blm_range
        self.templates = []
        if lmax_marg > 1:
            assert len(self.n_inv) == 1, 'implement if 3'
            self.templates.append(template_bfilt(lmax_marg=lmax_marg, geom=geom, sht_threads=sht_threads, _lib_dir=_bmarg_lib_dir))
        if len(self.templates) > 0:
            if _bmarg_lib_dir is not None and os.path.exists( os.path.join(_bmarg_lib_dir, 'tniti.npy')):
                log.info("Loading " + os.path.join(_bmarg_lib_dir, 'tniti.npy'))
                self.tniti = np.load(os.path.join(_bmarg_lib_dir, 'tniti.npy'))
                if _bmarg_rescal != 1.:
                    log.info("**** RESCALING tiniti with %.4f"%_bmarg_rescal)
                    self.tniti *= _bmarg_rescal
            else:
                log.info("Inverting template matrix:")
                tnit = self.templates[0].build_tnit((self.n_inv[0], self.n_inv[0], None))
                eigv, eigw = np.linalg.eigh(tnit)
                if not np.all(eigv > 0):
                    log.info('Negative or zero eigenvalues in template projection')
                eigv_inv = utils.cli(eigv)
                self.tniti = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))
                if _bmarg_lib_dir is not None and not os.path.exists(os.path.join(_bmarg_lib_dir, 'tniti.npy')):
                    np.save(os.path.join(_bmarg_lib_dir, 'tniti.npy'), self.tniti)
                    log.info("Cached " + os.path.join(_bmarg_lib_dir, 'tniti.npy'))

        self.zbounds = zbounds


    def apply_map(self, qumap):
        [qmap, umap] = qumap
        if len(self.n_inv) == 1:  # TT, QQ=UU
            if (self.blm_range[0] <= 2) and (self.blm_range[1] >= (3 * self.nside - 1)):
                qmap *= self.n_inv[0]
                umap *= self.n_inv[0]
                # tmap *= self.n_inv
                if len(self.templates) != 0:
                    coeffs = np.concatenate(([t.dot([qmap, umap]) for t in self.templates]))
                    coeffs = np.dot(self.tniti, coeffs)
                    pmodes = [np.zeros_like(qmap), np.zeros_like(umap)]
                    im = 0
                    for t in self.templates:
                        t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                        im += t.nmodes
                    pmodes[0] *= self.n_inv[0]
                    pmodes[1] *= self.n_inv[0]
                    qmap -= pmodes[0]
                    umap -= pmodes[1]
            else:
                log.info("apply_map: cuts %s %s"%(self.blm_range[0], self.blm_range[1]))
                elm, blm = lug.map2alm_spin(np.array([qmap, umap]), 2, lmax=min(3 * self.nside - 1, self.blm_range[1]), mmax=min(3 * self.nside - 1, self.blm_range[1]), nthreads=4)
                if self.blm_range[0] > 2: # approx taking out the low-ell B-modes
                    b_ftl = np.ones(hp.Alm.getlmax(blm.size) + 1, dtype=float)
                    b_ftl[:self.blm_range[0]] *= 0.
                    hp.almxfl(blm, b_ftl, inplace=True)

                q, u = lug.alm2map_spin(np.array([elm, blm]), 2, lmax=hp.Alm.getlmax(elm.size), mmax=hp.Alm.getlmax(elm.size), nthreads=4, zbounds=self.zbounds)
                qmap[:] = q * self.n_inv[0]
                umap[:] = u * self.n_inv[0]

        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            assert 0, 'implement template deproj.'
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0
