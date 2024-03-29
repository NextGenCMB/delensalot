from __future__ import annotations
import os, sys
import numpy as np
import scarf
from delensalot.core.helper.utils_sht import st2mmax, lowprimes
from delensalot.utils import clhash


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

    def __eq__(self, other:pbounds):
        return self.pctr == other.pctr and self.hext == other.hext

    def get_range(self):
        return 2 * self.hext

    def get_ctr(self):
        return self.pctr

    def contains(self, phs:np.ndarray):
        dph = (phs - self.pctr) % (2 * np.pi)  # points inside are either close to zero or 2pi
        return (dph <= self.hext) |( (2 * np.pi - dph) <= self.hext)

Geometry = scarf.Geometry

class pbdGeometry:
    def __init__(self, geom:Geometry, pbound:pbounds):
        """Scarf geometry with additional info on longitudinal cuts


        """
        self.geom = geom
        self.pbound = pbound



    def fsky(self):
        """Area of the sky covered by the pixelization"""
        return np.sum(self.geom.weight * self.geom.nph) / (4 * np.pi)


class Geom:
    """This collects simple static methods for scarf Geometry class


    """
    @staticmethod
    def npix(geom:scarf.Geometry):
        return int(np.sum(geom.nph))

    @staticmethod
    def fsky(geom:scarf.Geometry):
        """Area of the sky covered by the pixelization"""
        return np.sum(geom.weight * geom.nph) / (4 * np.pi)

    @staticmethod
    def phis(geom:scarf.Geometry, ir):
        assert ir < geom.get_nrings(), (ir, geom.get_nrings())
        nph = geom.get_nph(ir)
        return (geom.get_phi0(ir) + np.arange(nph) * (2 * np.pi  / nph)) % (2. * np.pi)

    @staticmethod
    def tbounds(geom:scarf.Geometry):
        return np.min(geom.theta), np.max(geom.theta)

    @staticmethod
    def rings2pix(geom:scarf.Geometry, rings:np.ndarray[int]):
        return np.concatenate([geom.get_ofs(ir) + np.arange(geom.get_nph(ir), dtype=int) for ir in rings])

    @staticmethod
    def pix2ang(geom: scarf.Geometry, pixs: np.ndarray[int] or list[int]):
        """Convert pixel indices to latitude and longitude (not terribly fast)"""
        s_pixs = np.argsort(pixs)
        thts, phis = np.empty(s_pixs.shape, dtype=float), np.empty(s_pixs.shape, float)
        sorted_iofs = np.argsort(geom.ofs)
        i, this_pix = 0, pixs[s_pixs[0]]
        this_ring = 0
        for next_ring in sorted_iofs:
            while this_pix < geom.ofs[next_ring]:
                thts[s_pixs[i]] = geom.theta[this_ring]
                phis[s_pixs[i]] = geom.phi0[this_ring] + (2 * np.pi) * (this_pix - geom.ofs[this_ring]) / geom.nph[this_ring]
                i += 1
                if i == len(pixs):
                    return thts, phis
                this_pix = pixs[s_pixs[i]]
            this_ring = next_ring
        assert 0, 'invalid inputs'

    @staticmethod
    def rings2phi(geom:scarf.Geometry, rings:np.ndarray[int]):
        return np.concatenate([Geom.phis(geom, ir) for ir in rings])

    @staticmethod
    def rings2tht(geom: scarf.Geometry, rings: np.ndarray[int]):
        return np.concatenate([geom.theta[ir] * np.ones(geom.nph[ir]) for ir in rings])


    @staticmethod
    def merge(geomlist:list[scarf.Geometry]):
        """Concatenates a list of pizelizations to a larger ones. Does not check for latitudes overlaps.

        """
        thts = np.concatenate([geom.theta  for geom in geomlist])
        nph  = np.concatenate([geom.nph  for geom in geomlist])
        phi0 = np.concatenate([geom.phi0  for geom in geomlist])
        wt   = np.concatenate([geom.weight for geom in geomlist])
        npixs = np.cumsum([Geom.npix(geom) for geom in geomlist])
        ofs  = np.concatenate([geom.ofs + npix - npixs[0] for geom, npix in zip(geomlist, npixs)])
        new_geom = scarf.Geometry(thts.size, nph, ofs, 1, phi0, thts, wt)
        assert Geom.npix(new_geom) ==  npixs[-1]
        return new_geom

    @staticmethod
    def pbounds2pix(geom:scarf.Geometry, ir, pbs:pbounds):
        """Returns pixels (unsorted) of geometry maps within longitude bounds

        """
        assert ir < geom.get_nrings(), (ir, geom.get_nrings())
        if pbs.get_range() >= (2. * np.pi) :
            return geom.ofs[ir] + np.arange(geom.nph[ir], dtype=int)
        jmin = geom.nph[ir] * (((pbs.pctr - pbs.hext - geom.phi0[ir]) / (2 * np.pi)) % 1.)
        jmax = jmin + geom.nph[ir] * pbs.get_range() / (2. * np.pi)
        return geom.ofs[ir] + np.arange(int(np.ceil(jmin)), int(np.floor(jmax)) + 1) % geom.nph[ir]

    @staticmethod
    def pbounds2npix(geom:scarf.Geometry, pbs:pbounds):
        """Returns total number of pixels within longitudinal bounds

        """
        if pbs.get_range() >= (2. * np.pi) : return Geom.npix(geom)
        jmins = geom.nph * (((pbs.pctr - pbs.hext - geom.phi0) / (2 * np.pi)) % 1.)
        jmaxs = jmins + geom.nph * (pbs.get_range() / (2. * np.pi))
        return np.sum(np.floor(jmaxs).astype(int) - np.ceil(jmins).astype(int)) + jmaxs.size


    @staticmethod
    def pbdmap2map(geom:scarf.Geometry, m_bnd:np.ndarray, pbs:pbounds):
        """Converts a map defined on longitude cuts back to the input geometry with full longitude range

            Note:
                'inverse' to Geom.map2pbnmap

        """
        assert Geom.pbounds2npix(geom, pbs) == m_bnd.size, ('incompatible arrays size', (Geom.npix(geom), m_bnd.size))
        if pbs.get_range() >= (2. * np.pi) :
            return m_bnd
        jmins = geom.nph * ( ((pbs.pctr - pbs.hext - geom.phi0) / (2 * np.pi)) % 1. )
        jmaxs = np.floor(jmins + geom.nph * (pbs.get_range() / (2. * np.pi))).astype(int)
        jmins = np.ceil(jmins).astype(int)
        m = np.zeros(Geom.npix(geom), dtype=m_bnd.dtype)
        start = 0
        for ir in np.argsort(geom.ofs):
            pixs = geom.ofs[ir] + np.arange(jmins[ir], jmaxs[ir] + 1)%geom.nph[ir]
            m[pixs] = m_bnd[start:start + pixs.size]
            start += pixs.size
        return m


    @staticmethod
    def map2pbnmap(geom:scarf.Geometry, m:np.ndarray, pbs:pbounds):
        """Converts a map defined by the input geometry to a small array according to input longitude cuts

            Note:
                'inverse' to Geom.pbdmap2map

        """
        assert Geom.npix(geom) == m.size, ('incompatible arrays size', (Geom.npix(geom), m.size))
        if pbs.get_range() >= (2. * np.pi) :
            return m
        jmins = geom.nph * (((pbs.pctr - pbs.hext - geom.phi0) / (2 * np.pi)) % 1.)
        jmaxs = np.floor(jmins + geom.nph * (pbs.get_range() / (2. * np.pi))).astype(int)
        jmins = np.ceil(jmins).astype(int)
        m_bnd = np.empty(np.sum(jmaxs - jmins) + jmaxs.size, dtype=m.dtype)
        start = 0
        for ir in np.argsort(geom.ofs):
            pixs = geom.ofs[ir] + np.arange(jmins[ir], jmaxs[ir] + 1)%geom.nph[ir]
            m_bnd[start:start + pixs.size] = m[pixs]
            start += pixs.size
        return m_bnd

    @staticmethod
    def get_healpix_geometry(nside:int, zbounds:tuple[float, float]=(-1., 1.)):
        hp_geom = scarf.healpix_geometry(nside, 1)
        tbounds = np.arccos(zbounds)
        if zbounds[0] > -1. or zbounds[1] < 1.:
            ri, = np.where( (hp_geom.theta >= tbounds[1]) & (hp_geom.theta <= tbounds[0]))
            nph = hp_geom.nph[ri]
            ofs = hp_geom.ofs[ri] - np.min(hp_geom.ofs[ri]) if ri.size > 0 else np.array([])
            geom = scarf.Geometry(ri.size, nph, ofs, 1, hp_geom.phi0[ri], hp_geom.theta[ri], hp_geom.weight[ri])
        else:
            geom = hp_geom
        return geom

    @staticmethod
    def get_thingauss_geometry(lmax:int, smax:int, zbounds:tuple[float, float]=(-1., 1.)):
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
        return scarf.Geometry(nlat, nph, ofs, 1, phi0, tht, wt * (2 * np.pi / nph ))

    @staticmethod
    def get_ecp_geometry(nlat:int, nlon:int, phi_center:float=np.pi, tbounds:tuple[float, float]=(0., np.pi)):
        assert nlat >= 2, nlat
        tbounds = np.sort(tbounds)
        tht = np.linspace(max(tbounds[0], 0), min(tbounds[1], np.pi), nlat)
        wt = np.ones(nlat, dtype=float) * (2 * np.pi / nlon * np.pi / (nlat - 1))
        wt[[0, -1]] *= 0.5
        phi0s = (phi_center - np.pi) + np.zeros(nlat, dtype=float)
        nph = nlon * np.ones(nlat, dtype=int)
        ofs = np.arange(nlat, dtype=int) * nlon
        return scarf.Geometry(nlat, nph, ofs, 1, phi0s, tht, wt)


    @staticmethod
    def get_pixel_geometry(tht:float or np.ndarray, phi:float or np.ndarray):
        """Single pixel geom


        """
        #FIXME: dont understand the result with a single phi, looks like needs at least two phis...
        if not np.isscalar(tht): tht = tht[0]
        if not np.isscalar(phi): phi = phi[0]

        assert 0 <= tht <= np.pi
        return scarf.Geometry(1, np.array([2]), np.array([0]), 1, np.array([phi %(2 * np.pi)]), np.array([tht]), np.array([1.]))

    @staticmethod
    def hashdict(geom:scarf.Geometry):
        """Returns a hash dictionary from scarf geometry

        """
        arrs = [geom.theta, geom.nph, geom.ofs, geom.weight, geom.phi0]
        labs = ['theta', 'nph', 'ofs', 'weight', 'phi0']
        typs = [np.float16, int, int, np.float16, np.float16]
        return {lab : clhash(arr, dtype=typ) for arr, lab, typ in zip(arrs, labs, typs)}

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

    def set_healpix_geometry(self, nside, zbounds=(-1,1.)):
        self.geom = Geom.get_healpix_geometry(nside, zbounds=zbounds)

    def set_ecp_geometry(self, nlat, nlon, phi_center=np.pi, tbounds=(0., np.pi)):
        r"""Cylindrical grid equidistant in longitude and latitudes, between the provided co-latitude bounds


            Args:
                nlat: number of latitude points
                nlon: number of longitude points
                phi_center: longitude of center of patch in radians (defaults to pi)
                tbounds: latitudes of the patch boudaries (in radians, defaults to (0, pi)

            Co-latitudes are :math:`\theta = i \frac{\pi}{Nt-1}, i=0,...,Nt-1`
            Longitudes are :math:`\phi = i \frac{2\pi}{Np}, i=0,...,Np-1`

            Not planning to use this for map2alm directions, weights not optimized.


        """
        self.geom = Geom.get_ecp_geometry(nlat, nlon,phi_center=phi_center, tbounds=tbounds)

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
        self.geom = Geom.get_thingauss_geometry(lmax, smax, zbounds=zbounds)

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