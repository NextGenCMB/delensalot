import os
import numpy as np
import scarf
from lenscarf.utils_sht import st2mmax, lowprimes

class pbounds:
    """Class to regroup simple functions handling sky maps longitude truncation

            Args:
                pctr: center of interval in radians
                prange: full extent of interval in radians

            Note:
                2pi periodicity

    """
    def __init__(self, pctr:float, prange:float):
        assert prange >= 0., prange
        self.pctr = pctr % (2. * np.pi)
        self.hext = min(prange * 0.5, np.pi) # half-extent

    def __repr__(self):
        return "ctr:%.2f range:%.2f"%(self.pctr, self.hext * 2)

    def get_range(self):
        return 2 * self.hext

    def get_ctr(self):
        return self.pctr

    def contains(self, phs:np.ndarray):
        dph = (phs - self.pctr) % (2 * np.pi)  # points inside are either close to zero or 2pi
        return (dph <= self.hext) |( (2 * np.pi - dph) <= self.hext)



class Geom:
    """This collects simple static methods for scarf Geometry class


    """
    @staticmethod
    def npix(geom:scarf.Geometry):
        return np.sum(geom.nph)

    @staticmethod
    def phis(geom:scarf.Geometry, ir):
        assert ir < geom.get_nrings(), (ir, geom.get_nrings())
        nph = geom.get_nph(ir)
        return (geom.get_phi0(ir) + np.arange(nph) * (2 * np.pi  / nph)) % (2. * np.pi)

    @staticmethod
    def tbounds(geom:scarf.Geometry):
        return np.min(geom.theta), np.max(geom.theta)

    @staticmethod
    def rings2pix(geom:scarf.Geometry, rings:np.ndarray):
        return np.concatenate([geom.get_ofs(ir) + np.arange(geom.get_nph(ir), dtype=int) for ir in rings])

    @staticmethod
    def rings2phi(geom:scarf.Geometry, rings:np.ndarray):
        return np.concatenate([Geom.phis(geom, ir) for ir in rings])

    @staticmethod
    def rings2tht(geom: scarf.Geometry, rings: np.ndarray):
        return np.concatenate([geom.theta[ir] * np.ones(geom.nph[ir]) for ir in rings])

    @staticmethod
    def pbounds2pix(geom:scarf.Geometry, ir, pbs:pbounds):
        assert ir < geom.get_nrings(), (ir, geom.get_nrings())
        pixs = geom.get_ofs(ir) + np.arange(geom.get_nph(ir), dtype=int)
        return pixs[pbs.contains(Geom.phis(geom, ir))]

    @staticmethod
    def pbounds2npix(geom:scarf.Geometry, pbs:pbounds):
        if pbs.get_range() >= (2. * np.pi) : return Geom.npix(geom)
        npix = 0
        for ir in range(geom.get_nrings()):
            npix += np.sum(pbs.contains(Geom.phis(geom, ir))) #FIXME: hack
        return npix

    @staticmethod
    def pbdmap2map(geom:scarf.Geometry, m_bnd:np.ndarray, pbs:pbounds):#FIXME: hack
        """Converts a map defined on longitude cuts back to the input geometry with full longitude range

            Note:
                'inverse' to Geom.map2pbnmap

        """
        assert Geom.pbounds2npix(geom, pbs) == m_bnd.size, ('incompatible arrays size', (Geom.npix(geom), m_bnd.size))
        if pbs.get_range() >= (2. * np.pi) : return m_bnd
        m = np.zeros(Geom.npix(geom), dtype=m_bnd.dtype)
        start = 0
        for ir in range(geom.get_nrings()):
            pixs = Geom.pbounds2pix(geom, ir, pbs)
            m[pixs] = m_bnd[start:start + pixs.size]
            start += pixs.size
        return m

    @staticmethod
    def map2pbnmap(geom:scarf.Geometry, m:np.ndarray, pbs:pbounds):#FIXME: hack
        """Converts a map defined by the input geometry to a small array according to input longitude cuts

            Note:
                'inverse' to Geom.pbdmap2map

        """
        assert Geom.npix(geom) == m.size, ('incompatible arrays size', (Geom.npix(geom), m.size))
        if pbs.get_range() >= (2. * np.pi) : return m
        m_bnd = np.zeros(Geom.pbounds2npix(geom, pbs), dtype=m.dtype)
        start = 0
        for ir in range(geom.get_nrings()):
            pixs = Geom.pbounds2pix(geom, ir, pbs)
            m_bnd[start:start + pixs.size] = m[pixs]
            start += pixs.size
        return m_bnd

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

    @staticmethod
    def supported_geometries():
        return ['healpix', 'ecp','gauss', 'thingauss', 'pixel']

    def n_pix(self):
        return np.sum(self.geom.nph)

    def set_healpix_geometry(self, nside):
        self.geom = scarf.healpix_geometry(nside, 1)

    def set_ecp_geometry(self, nlat, nlon, phi_center=np.pi, tbounds=(0., np.pi)):
        r"""Cylindrical grid equidistant in longitude and latitudes, between the provided co-latitude bounds


            Args:
                nlat: number of latitude points
                nlon: number of longitude points
                phi_center: longitude of center of patch in radians (defaults to pi)
                tbounds: latitudes of the patch boudaries (in radians, defaults to (0, pi)

            Co-latitudes are :math:`\theta = i \frac{\pi}{Nt-1}, i=0,...,Nt-1`
            Longitudes are :math:`\phi = i \frac{2\pi}{Np}, i=0,...,Np-1`

            Not planning to use this for map2alm directions, weights not optimized


        """
        assert nlat >= 2, nlat
        tbounds = np.sort(tbounds)
        tht = np.linspace(max(tbounds[0], 0), min(tbounds[1], np.pi), nlat)
        wt = np.ones(nlat, dtype=float) * (2 * np.pi / nlon * np.pi / (nlat - 1))
        wt[[0, -1]] *= 0.5
        phi0s = (phi_center - np.pi) + np.zeros(nlat, dtype=float)
        nph = nlon * np.ones(nlat, dtype=int)
        ofs = np.arange(nlat, dtype=int) * nlon
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0s, tht, wt)

    def set_gauss_geometry(self, nlat, nlon):
        """Standard Gauss-Legendre grid


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
                lmax: band-limit (or desired band-limit) on the equator
                smax: maximal intended spin-value (this changes the m-truncation by an amount ~smax)
                zbounds: pixels outside of provided cos-colatitude bounds will be discarded

            Note:
                'thinning' saves memory but hardly any computational time for the same latitude range


        """
        nlatf = lmax + 1  # full meridian GL points
        tht = np.arccos(scarf.GL_xg(nlatf))
        tb = np.sort(np.arccos(zbounds))
        p = np.where((tb[0] <= tht) & (tht <= tb[1]))

        tht = tht[p]
        wt = scarf.GL_wg(nlatf)[p]
        nlat = tht.size
        phi0 = np.zeros(nlat, dtype=float)
        mmax = np.minimum(np.maximum(st2mmax(smax, tht, lmax), st2mmax(-smax, tht, lmax)), np.ones(nlat) * lmax)
        nph = lowprimes(np.ceil(2 * mmax + 1))
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0, tht, wt * (2 * np.pi / nph ))

    def set_pixel_geometry(self, tht:float or np.ndarray, phi:float or np.ndarray):
        """Single ring with two phis


        """
        #assert 0
        #FIXME: dont understand the result with a single phi, looks like needs at least two phis...
        if not np.isscalar(tht): tht = tht[0]
        if not np.isscalar(phi): phi = phi[0]

        assert 0 <= tht <= np.pi
        self.geom = scarf.Geometry(1, np.array([2]), np.array([0]), 1, np.array([phi %(2 * np.pi)]), np.array([tht]), np.array([1.]))

    def set_geometry(self, geom:scarf.Geometry):
        self.geom = geom

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