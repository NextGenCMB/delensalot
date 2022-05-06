"""Module for scalar field harmonic modes template deprojection


"""
import os
import numpy as np
import scarf
from lenscarf import utils_scarf as us
from lenscarf.utils_hp import Alm
from lenscarf.utils import enumerate_progress, read_map


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




class template_tfilt(object):
    def __init__(self, lmax_marg:int, geom:scarf.Geometry, sht_threads:int, _lib_dir=None):
        """Here all T-modes up to lmax are set to infinite noise

            Args:
                lmax_marg: all T-mulitpoles up to and inclusive of lmax are marginalized
                geom: scarf geometry of SHTs
                sht_threads: number of OMP threads for SHTs
                _lib_dir: some stuff might be cached there in some cases


        """
        assert lmax_marg >= 1, lmax_marg # monopole always assumed to be zero
        self.lmax = lmax_marg
        self.nmodes = (lmax_marg + 1) * lmax_marg + lmax_marg + 1 - 1
        if not np.all(geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            print('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
            nr = geom.get_nrings()
            geom_ = us.Geometry(nr, geom.nph.copy(), geom.ofs.copy(), 1, geom.phi0.copy(), geom.theta.copy(), np.ones(nr, dtype=float))
        else:
            geom_ = geom
            # Does not seem to work without the 'copy'
        sc_job = us.scarfjob()
        sc_job.set_geometry(geom_)
        sc_job.set_triangular_alm_info(lmax_marg, lmax_marg)
        sc_job.set_nthreads(sht_threads)

        self.sc_job = sc_job
        self.npix = us.Geom.npix(sc_job.geom)
        self.lib_dir = None
        if _lib_dir is not None and lmax_marg > 10: #just to avoid problems if user does not understand what is doing...
            if not os.path.exists(_lib_dir):
                os.makedirs(_lib_dir)
            self.lib_dir = _lib_dir

    def hashdict(self):
        return {'lmax':self.lmax, 'geom':us.Geom.hashdict(self.sc_job.geom)}

    @staticmethod
    def get_nmodes(lmax):
        assert lmax >= 1, lmax
        return (lmax + 1) * lmax + lmax + 1 - 1

    @staticmethod
    def get_modelmax(mode):
        assert mode >= 0, mode
        nmodes = 0
        l = -1
        while nmodes - 1 < mode + 1:
            l += 1
            nmodes += 2 * l + 1
        return l

    @staticmethod
    def _rlm2blm(rlm):
        return rlm2alm(np.concatenate([np.zeros(1), rlm]))

    @staticmethod
    def _blm2rlm(blm):
        return alm2rlm(blm)[1:]

    def apply_tmode(self, tmap, mode):
        assert mode < self.nmodes, (mode, self.nmodes)
        tcoeffs = np.zeros(self.get_nmodes(self.get_modelmax(mode)), dtype=float)
        tcoeffs[mode] = 1.0
        self.apply_t(tmap, tcoeffs)

    def apply_t(self, tmap, coeffs):  # RbQ  * Q or  RbU * U
        assert (len(coeffs) <= self.nmodes)
        assert tmap.size == self.npix, (self.npix, tmap.size)
        tlm = self._rlm2blm(coeffs)
        this_lmax = Alm.getlmax(tlm.size, -1)
        self.sc_job.set_triangular_alm_info(this_lmax, this_lmax)
        tmap *= self.sc_job.alm2map(tlm)

    def accum(self, tmap, coeffs):
        """Forward template operation

            Turns the input real harmonic *coeffs* to tlm and send to T map.
            This plus-adds the input *tmap*

        """
        assert (len(coeffs) <= self.nmodes)
        tlm = self._rlm2blm(coeffs)
        this_lmax = Alm.getlmax(tlm.size, -1)
        self.sc_job.set_triangular_alm_info(this_lmax, this_lmax)
        tmap += self.sc_job.alm2map(tlm)

    def dot(self, tmap):
        """Backward template operation.

            Turns the input map into the real harmonic coefficient T-modes up to lmax.
            This includes a factor npix / 4pi, as the transpose differs from the inverse by that factor

        """
        assert tmap.size == self.npix
        self.sc_job.set_triangular_alm_info(self.lmax, self.lmax)
        tlm = self.sc_job.map2alm(tmap)
        return self._blm2rlm(tlm) # Units weight transform

    def build_tnit(self, NiT):
        """Return the nmodes x nmodes matrix (T^t N^{-1} T )_{bl bl'}'

            For unit inverse noise matrices on the full-sky this is a diagonal matrix
            with diagonal :math:`N_{\rm pix} / (4\pi)`.

            If input inverse noise matrices are the inverse pixel noise in uK, this give 1/noise level ** 2 in uK-rad squared

        """
        if self.lib_dir is not None:
            return self._build_tnit('')
        tnit = np.zeros((self.nmodes, self.nmodes), dtype=float)
        for i, a in enumerate_progress(range(self.nmodes), False * 'filling template matrix'):  # Starts at ell = 1
            _NiT = np.copy(NiT)  # Building Ni_{QX} R_bX
            self.apply_tmode(_NiT, a)
            tnit[:, a] = self.dot(_NiT)
            tnit[a, :] = tnit[:, a]
        return tnit

    def _build_tnit(self, prefix=''):
        tnit = np.zeros((self.nmodes, self.nmodes), dtype=float)
        for i, a in enumerate_progress(range(self.nmodes), label='collecting Tmat rows'):
            fname = os.path.join(self.lib_dir, prefix + 'row%05d.npy'%a)
            assert os.path.exists(fname)
            tnit[:, a]  = np.load(fname)
            tnit[a, :] = tnit[:, a]
        return tnit


    def _get_rows_mpi(self, NiT, prefix):
        """Produces and save all rows of the matrix for large matriz sizes

        """
        from plancklens.helpers import mpi
        assert self.lib_dir is not None, 'cant do this without a lib_dir'
        assert self.nmodes <= 99999, 'ops, naming in the lines below'
        for a in range(self.nmodes)[mpi.rank::mpi.size]:
            fname = os.path.join(self.lib_dir, prefix + 'row%05d.npy'%a)
            if not os.path.exists(fname):
                _NiT = np.copy(NiT)  # Building Ni_{T} R_bT
                self.apply_tmode(_NiT, a)
                np.save(fname, self.dot(_NiT))


class template_dense(template_tfilt):
    def __init__(self, lmax_marg:int, geom:scarf.Geometry, sht_threads:int, _lib_dir=None, rescal=1.):
        """Dense harmonic mode template projection matrix instance

            Typically used with a precomputed matrix

            Args:
                lmax_marg: maximum projected multipole (inclusive)
                geom: scarf geometry of the data maps
                sht_threads: number of threads per SHT
                _lib_dir(optional): the instance will look for a matrix cached there if set
                rescal(optional): rescales the matrix by this factor. Useful if the original matrix was calculated using
                        a different overall noise level. The 'tniti' template matrix is proportional to the squared noise level


        """
        assert os.path.exists(os.path.join(_lib_dir, 'tniti.npy')), os.path.join(_lib_dir, 'tniti.npy')
        super().__init__(lmax_marg, geom, sht_threads, _lib_dir=_lib_dir)
        self.rescal = rescal
        self._tniti = None # will load this when needed

    def hashdict(self):
        return {'lmax':self.lmax, 'geom':us.Geom.hashdict(self.sc_job.geom),  'rescal':self.rescal}

    def tniti(self):
        if self._tniti is None:
            self._tniti = read_map(os.path.join(self.lib_dir, 'tniti.npy')) * self.rescal
            print("reading " +os.path.join(self.lib_dir, 'tniti.npy') )
            print("Rescaling it with %.5f"%self.rescal)
        return self._tniti