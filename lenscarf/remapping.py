from lenscarf.skypatch import skypatch
from lenscarf import interpolators as itp
from lenscarf import utils_remapping
from lenscarf import cachers
from lenscarf.utils_hp import getlmax
from lenscarf.fortran import remapping as fremap
import numpy as np

class deflection:
    def __init__(self, scarf_geometry, targetres_amin, tht_bounds, p_bounds, dlm, fftw_threads, scarf_threads,
                 cacher=cachers.cacher_none(), clm=None):
        """

        """
        assert (0. <= tht_bounds[0] < tht_bounds[1] <= np.pi), tht_bounds
        assert (p_bounds[0] < p_bounds[1]) and (p_bounds[1] - p_bounds[0]) <= (2 * np.pi), p_bounds

        # --- interpolation of spin-1 deflection on the desired area and resolution

        self.sky_patch = skypatch(tht_bounds, p_bounds, targetres_amin, pole_buffers=3)
        self.dlm = dlm
        self.clm = clm
        self.d1 = None # -- this might be instantiated later if needed
        self.cacher = cacher
        self.scarf_geometry = scarf_geometry

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = tht_bounds
        self._pbds = p_bounds
        self._resamin = targetres_amin
        self._sht_tr = scarf_threads
        self._fft_tr = fftw_threads

    def _build_interpolator(self, glm, spin, clm=None):
        bufamin = 30.
        print("***instantiating spin-%s interpolator with %s amin buffers"%(spin, bufamin))
        # putting a d = 0.01 ~ 30 arcmin buffer which should be way more than enough
        buf = bufamin/ 180 / 60 * np.pi
        tbds = [max(self._tbds[0] - buf, 0.), min(np.pi, self._tbds[1] + buf)]
        pbds =  [self._pbds[0] - buf, self._pbds[1] + buf]
        if pbds[1] - pbds[0] >= 2 * np.pi:
            pctr = np.mean(self._pbds)
            pbds = [pctr - np.pi, pctr + np.pi]
        return itp.bicubic_ecp_interpolator(spin, glm, tbds, self._resamin, self._sht_tr, self._fft_tr,  p_bounds=pbds, clm=clm)

    def _init_d1(self):
        if self.d1 is None:
            self.d1 = self._build_interpolator(self.dlm, 1)

    def _fwd_angles(self):
        """Builds deflected angles for the forawrd deflection field for the pixels of a healpix map inside the patch


        """
        fn = 'fwdangles'
        if not self.cacher.is_cached(fn):
            geo = self.scarf_geometry
            nrings = geo.nrings()
            npix = np.sum([geo.nph(i) for i in range(nrings)])
            stride = 1 #FIXME: can get stride from geom object?
            lmax, mmax = (getlmax(self.dlm.size), getlmax(self.dlm.size)) #FIXME: allow non-trivial mmax?
            clm = np.zeros_like(self.dlm) if self.clm is None else self.clm
            red, imd = geo.alm2map_spin([self.dlm, clm], 1, lmax, mmax, self._sht_tr, [-1., 1.])
            thp_phip= np.zeros( (2, npix), dtype=float)
            for ir in range(nrings):
                nph = geo.nph(ir)
                phis = geo.phi0(ir) + np.arange(nph) * ((2 * np.pi) / nph)
                sli = slice(geo.ofs(ir), geo.ofs(ir) + nph, stride)
                thtp_, phip_ = utils_remapping.d2ang(red[sli], imd[sli], geo.theta(ir), phis, int(np.round(geo.cth(ir))))
                thp_phip[0, sli] = thtp_
                thp_phip[1, sli] = phip_
            self.cacher.cache(fn, thp_phip)
            return thp_phip
        return self.cacher.load(fn)

    def _bwd_angles(self):
        self._init_d1()
        tht0 = self.d1._buf_t_bounds[0]
        phi0 = self.d1._ecp_pctr - np.pi + self.d1._phir_min
        t2grid = self.d1._trescal
        p2grid = self.d1._prescal
        geo = self.scarf_geometry
        nrings = geo.nrings()
        stride = 1
        npix = np.sum([geo.nph(i) for i in range(nrings)]) #:FIXME
        rediimdi = np.zeros((2, npix), dtype=float)
        from plancklens import utils #FIXME
        for i, ir in utils.enumerate_progress(range(nrings)):
            nph = geo.nph(ir)
            phis = geo.phi0(ir) + np.arange(nph) * ((2 * np.pi) / nph)
            sli = slice(geo.ofs(ir), geo.ofs(ir) + nph, stride)
            thts = np.ones(nph) * geo.theta(ir)
            redi, imdi = fremap.remapping.solve_pixs(self.d1._re_f , self.d1._im_f, thts, phis, tht0, phi0, t2grid, p2grid)
            rediimdi[0, sli] = redi
            rediimdi[1, sli] = imdi
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