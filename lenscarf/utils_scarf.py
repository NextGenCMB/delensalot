import os
import numpy as np
import scarf
from lenscarf.utils_sht import st2mmax, lowpapprox


class Geom:
    @staticmethod
    def npix(geom:scarf.Geometry):
        return np.sum([geom.nph(ir) for ir in range(geom.nrings())]) #:FIXME hack

    @staticmethod
    def phis(geom:scarf.Geometry, ir):
        nph = geom.nph(ir)
        return (geom.phi0(ir) + np.arange(nph) * (2 * np.pi  / nph)) % (2. * np.pi)

    @staticmethod
    def tbounds(geom:scarf.Geometry): # FIXME hack
        tmin = np.inf
        tmax = 0.
        for ir in range(geom.nrings()):
            tht = geom.theta(ir)
            if tht > tmax:
                tmax = tht
            if tht < tmin:
                tmin = tht
        return tmin, tmax



    @staticmethod
    def pbounds2pix(geom:scarf.Geometry, ir, pbounds): #FIXME hack
        if abs(pbounds[1] - pbounds[0]) >= (2 * np.pi):
            return geom.ofs(ir) + np.arange(geom.nph(ir)), Geom.phis(geom, ir)
        pbounds = np.array(pbounds) % (2. * np.pi)
        pixs = geom.ofs(ir) + np.arange(geom.nph(ir), dtype=int)
        phis = Geom.phis(geom, ir)
        print(pbounds)
        if pbounds[1] >= pbounds[0]:
            idc =  (phis >= pbounds[0]) & (phis <= pbounds[1])
        else:
            idc =  (phis <= pbounds[0]) | (phis >= pbounds[1])
        return pixs[idc], phis[idc]

    @staticmethod
    def pbounds2npix(geom:scarf.Geometry, pbounds): #FIXME hack
        if abs(pbounds[1] - pbounds[0]) >= (2 * np.pi):
            return Geom.npix(geom)
        pbounds = np.array(pbounds) % (2 * np.pi)
        npix = 0
        ordr = pbounds[1] >= pbounds[0]
        for ir in range(geom.nrings()):
            phis = Geom.phis(geom, ir)
            if ordr:
                npix += np.sum((phis >= pbounds[0]) & (phis <= pbounds[1]))
            else:
                npix += np.sum((phis <= pbounds[0]) | (phis >= pbounds[1]))
        return npix


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
        return ['healpix', 'ecp','gauss', 'thingauss']

    def n_pix(self):  # !FIXME add this to scarf.cc
        return np.sum([self.geom.nph(i) for i in range(self.geom.nrings())])

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
        tbounds = np.sort(tbounds)
        tht = np.linspace(max(tbounds[0], 0), min(tbounds[1], np.pi), nlat)
        wt = np.ones(nlat, dtype=float) * (2 * np.pi / nlon * np.pi / (nlat - 1))
        wt[[0, -1]] *= 0.5
        phi0s = (phi_center - np.pi) + np.zeros(nlat, dtype=float)
        nph = nlon * np.ones(nlat, dtype=int)
        ofs = np.arange(nlat, dtype=int) * nlon
        self.geom = scarf.Geometry(nlat, nph, ofs, 1, phi0s, tht, wt)

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