import scarf

from lenscarf.skypatch import skypatch
from lenscarf import interpolators as itp
from lenscarf.utils_remapping import d2ang
from lenscarf import cachers
from lenscarf.utils_hp import getlmax
from lenscarf.utils_scarf import Geom, pbounds as pbs
from lenscarf.fortran import remapping as fremap
import numpy as np

class deflection:
    def __init__(self, scarf_geometry:scarf.Geometry, targetres_amin, p_bounds, dlm, fftw_threads, scarf_threads,
                 cacher:cachers.cacher = cachers.cacher_none(), clm=None):
        """

                p_bounds: tuple with longitude cuts info in the form of (patch center, patch extent), both in radians


                scarf_geometry: points where the deflected is calculated. Resolution independent of the ECP patch


        #FIXME: allow non-trivial mmax?

        """
        assert (p_bounds[1] > 0), p_bounds

        # --- interpolation of spin-1 deflection on the desired area and resolution
        tht_bounds = Geom.tbounds(scarf_geometry)
        assert (0. <= tht_bounds[0] < tht_bounds[1] <= np.pi), tht_bounds
        #self.sky_patch = skypatch(tht_bounds, p_bounds, targetres_amin, pole_buffers=3)
        self.dlm = dlm
        self.clm = clm
        self.d1 = None # -- this might be instantiated later if needed
        self.cacher = cacher
        self.geom = scarf_geometry

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = tht_bounds
        self._pbds = pbs(p_bounds[0], p_bounds[1])  # (patch ctr, patch extent)
        self._resamin = targetres_amin
        self._sht_tr = scarf_threads
        self._fft_tr = fftw_threads

    def _build_interpolator(self, glm, spin, clm=None, mmax=None):
        bufamin = 30.
        print("***instantiating spin-%s interpolator with %s amin buffers"%(spin, bufamin))
        # putting a d = 0.01 ~ 30 arcmin buffer which should be way more than enough
        buf = bufamin/ 180 / 60 * np.pi
        tbds = [max(self._tbds[0] - buf, 0.), min(np.pi, self._tbds[1] + buf)]
        sintmin = np.min(np.sin(self._tbds))
        prange = min(self._pbds.get_range() + 2 * buf / sintmin if sintmin > 0 else 2 * np.pi, 2 * np.pi)
        buffered_patch = skypatch(tbds, (self._pbds.get_ctr(), prange), self._resamin, pole_buffers=3)
        return itp.bicubic_ecp_interpolator(spin, glm, buffered_patch, self._sht_tr, self._fft_tr, clm=clm, mmax=mmax)

    def _init_d1(self):
        if self.d1 is None:
            self.d1 = self._build_interpolator(self.dlm, 1)

    def _fwd_angles(self):
        """Builds deflected angles for the forawrd deflection field for the pixels inside the patch


        """
        fn = 'fwdangles'
        if not self.cacher.is_cached(fn):
            nrings = self.geom.nrings()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            lmax, mmax = (getlmax(self.dlm.size), getlmax(self.dlm.size)) #FIXME: allow non-trivial mmax?
            clm = np.zeros_like(self.dlm) if self.clm is None else self.clm
            red, imd = self.geom.alm2map_spin([self.dlm, clm], 1, lmax, mmax, self._sht_tr, [-1., 1.])
            thp_phip= np.zeros( (2, npix), dtype=float)
            startpix = 0
            for ir in range(nrings):
                pixs, phis = Geom.pbounds2pix(self.geom, ir, self._pbds)
                thtp_, phip_ = d2ang(red[pixs], imd[pixs], self.geom.theta(ir), phis, int(np.round(self.geom.cth(ir))))
                sli = slice(startpix, startpix + len(pixs))
                thp_phip[0, sli] = thtp_
                thp_phip[1, sli] = phip_
                startpix += len(pixs)
            self.cacher.cache(fn, thp_phip)
            assert startpix == npix, (startpix, npix)
            return thp_phip
        return self.cacher.load(fn)

    def _bwd_angles(self):
        self._init_d1()
        (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = self.d1.get_spline_info()
        nrings = self.geom.get_nrings()
        npix = Geom.pbounds2npix(self.geom, self._pbds)
        rediimdi = np.zeros((2, npix), dtype=float)
        from plancklens import utils #FIXME
        startpix = 0
        for i, ir in utils.enumerate_progress(range(nrings)):
            pixs, phis = Geom.pbounds2pix(self.geom, ir, self._pbds)
            thts = np.ones(len(pixs)) * self.geom.get_theta(ir)
            redi, imdi = fremap.remapping.solve_pixs(re_f , im_f, thts, phis, tht0, phi0, t2grid, p2grid)
            sli = slice(startpix, startpix + len(pixs))
            rediimdi[0, sli] = redi
            rediimdi[1, sli] = imdi
            startpix += len(pixs)
        assert startpix == npix, (startpix, npix)
        return rediimdi

    def lensgclm(self, glm, spin, nside, lmax_out, backwards=False, clm=None):
        interpjob = self._build_interpolator(glm, spin, clm=clm)
        hp_rings = self.sky_patch.hp_rings(nside)
        hp_pixs, start_hpchunk = self.sky_patch.hp_pixels(nside)

        # NB: could perform here unchecked interpolation, gain of 10-15% so not mindblowing
        hp_job = get_trhealpix_sharpjob(nside, hp_rings, lmax_out)
        hp_job.set_nthreads(self._sht_tr)
        hp_job.set_triangular_alm_info(lmax_out, lmax_out) #--- mmax?
        #FIXME: shall we store defl. in grid units only?
        if backwards:
            thtn, phin = self._build_healpix_bwd_deflangles(nside)
        else:
            thtn, phin = self._build_healpix_fwd_deflangles(nside)
        lenm = np.zeros((1 + (spin > 0), hp_job.n_pix()), dtype=float)
        lenm[:, hp_pixs] = interpjob.eval(thtn, phin)
        #FIXME: polarization rotation
        # --- going back to alm:
        if spin > 0:
            gclm_len = hp_job.map2alm_spin(lenm, spin)
        else:
            gclm_len = hp_job.map2alm(lenm.squeeze())

        return gclm_len