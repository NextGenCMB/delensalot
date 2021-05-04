import os
import numpy as np
import scarf
from lenscarf.utils_sht import st2mmax, lowpapprox

class scarfjob:
    r"""SHT job instance emulating existing ducc python bindings


        Note:
            the ordering of the rings is lost after the geometry objects definition!


    """

    def __init__(self):
        self.geom = None
        self.nthreads = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.lmax = None
        self.mmax = None

    def n_pix(self):  # !FIXME add this to scarf.cc
        return np.sum([self.geom.nph(i) for i in range(self.geom.nrings())])

    def set_healpix_geometry(self, nside):
        self.geom = scarf.healpix_geometry(nside, 1)

    def set_ecp_geometry(self, nlat, nlon, tbounds=(0., np.pi)):
        r"""Cylindrical grid equidistant in longitude and latitudes, between the provided co-latitude bounds

            Co-latitudes are :math:`\theta = i \frac{\pi}{Nt-1}, i=0,...,Nt-1`
            Longitudes are :math:`\phi = i \frac{2\pi}{Np}, i=0,...,Np-1`

            Not planning to use this for map2alm directions, weights not optimized


        """
        tbounds = np.sort(tbounds)
        tht = np.linspace(max(tbounds[0], 0), min(tbounds[1], np.pi), nlat)
        wt = np.ones(nlat, dtype=float) * (2 * np.pi / nlon * np.pi / (nlat - 1))
        wt[[0, -1]] *= 0.5
        phi0 = np.zeros(nlat, dtype=float)
        nph = nlon * np.ones(nlat, dtype=int)
        ofs = np.arange(nlat, dtype=int) * nlon
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0, tht, wt)

    def set_gauss_geometry(self, nlat, nlon):
        """standard Gauss-Legendre grid


        """
        tht = np.arccos(scarf.GL_xg(nlat))
        wt = scarf.GL_wg(nlat) * (2 * np.pi / nlon)
        phi0 = np.zeros(nlat, dtype=float)
        nph = nlon * np.ones(nlat, dtype=int)
        ofs = np.arange(nlat, dtype=int) * nlon
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0, tht, wt)

    def set_thingauss_geometry(self, lmax, smax, zbounds=(-1., 1.)):
        """Build a 'thinned' Gauss-Legendre geometry, using polar optimization to reduce the number of points away from the equator


            Args:
                nlat: number of GL latitude points
                lmax: band-limit (or desired band-limit) on the equator
                smax: maximal intended spin-value (this changes the m-truncation by an amount ~smax)
                zbounds: pixels outside of provided cos-colatitude bounds will be discarded

            Note:
                This saves memory but hardly any computational time


        """
        nlatf = lmax + 1  # full meridian GL points
        tht = np.arccos(scarf.GL_xg(nlatf))
        tb = np.sort(np.arccos(zbounds))
        p = np.where((tb[0] <= tht) & (tht <= tb[1]))

        tht = tht[p]
        wt = scarf.GL_wg(nlatf)[p]  # standard wg \cdot 2pi  n

        nlat = tht.size
        phi0 = np.zeros(nlat, dtype=float)
        mmax = np.minimum(np.maximum(st2mmax(smax, tht, lmax), st2mmax(-smax, tht, lmax)), np.ones(nlat) * lmax)
        nph = lowpapprox(np.ceil(2 * mmax + 1))
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0, tht, wt * (2 * np.pi / nph ))

    def set_nthreads(self, nthreads):
        self.nthreads = nthreads

    def set_triangular_alm_info(self, lmax, mmax):
        self.lmax = lmax
        self.mmax = mmax

    def alm2map(self, alm):
        return self.geom.alm2map(alm, self.lmax, self.mmax, self.nthreads, [-1., 1.])

    def map2alm(self, m):
        return self.geom.map2alm(m, self.lmax, self.mmax, self.nthreads, [-1., 1.])

    def alm2map_spin(self, gclm, spin):
        return self.geom.alm2map_spin(gclm, spin, self.lmax, self.mmax, self.nthreads, [-1., 1.])

    def map2alm_spin(self, m, spin):
        return self.geom.map2alm_spin(m, spin, self.lmax, self.mmax, self.nthreads, [-1., 1.])