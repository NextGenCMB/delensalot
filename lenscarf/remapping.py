"""This module contains the main interpolators


"""
from lenscarf.utils import timer
from lenscarf.utils_hp import getlmax
import numpy as np
import pyfftw
from omppol.src import bicubic #:FIXME

class spinfield_interpolator:
    """Spin-weighted bicubic spline interpolator on a patch of the sky.

        Args:
            spin: spin-weight of the map to interpolate
            glm: gradient-mode of the map to interpolate (healpix/py format)
            t_bounds:  co-latitude bounds (rad) of the patch to do the interpolation on
            targetres_amin: target resolution of the interpolation operation
            sht_threads: numbers of OMP threads to perform shts with (libsharp / ducc)
            fftw_threads: numbers of threads for FFT's
            p_bounds: longitude bounds (defaults to (0, 2pi))
            clm: curl mode of map to interpolate, if relevant

        Note:
            The first and last pixels (of the first dimension) match the input colatitude bounds exactly
            The longitude bounds (in the second dimension) will be buffered by some small amount and will not match pixels exactly

        Note:
            On first instantiation pyfftw might spend some extra time calculating a FFT plan

    """
    def __init__(self, spin, glm, t_bounds, targetres_amin, sht_threads, fftw_threads,
                 p_bounds=(0., 2 * np.pi), clm=None, mmax=None, pole_buffers=3):
        # Defines latitude grid from input bounds:
        assert spin >= 0, spin
        assert 0<= t_bounds[0] < t_bounds[1] <= np.pi, t_bounds
        assert p_bounds[0] < p_bounds[1], p_bounds
        assert (p_bounds[1] - p_bounds[0]) <= 2 * np.pi, p_bounds

        tim = timer(True, prefix='spinfield interpolator')
        th1, th2 = t_bounds
        patch = skypatch(t_bounds, p_bounds, targetres_amin, pole_buffers=pole_buffers)
        ecp_nt, ecp_nph = patch.ecp_ring_ntphi()
        nt_buf_n, nt_buf_s = patch.nt_buffers_n, patch.nt_buffers_s
        ecp_nt_nobuf = ecp_nt - nt_buf_n - nt_buf_s
        ecp_thts = th1 + (th2 - th1) / (ecp_nt_nobuf - 1.) * np.arange(ecp_nt_nobuf, dtype=float)
        ecp_pctr = np.mean(p_bounds)
        imin, imax = patch.ecp_resize(ecp_nph)
        print(ecp_nt_nobuf, ecp_nt, imin, imax)

        lmax = getlmax(glm.size)
        # ------ Build custom libsharp job with patch center on the middle of ECP map
        ecp_job = sht.sharpjob_d()
        ecp_job.set_triangular_alm_info(lmax, lmax if mmax is None else mmax)
        ecp_job.set_nthreads(sht_threads)

        ecp_ofs = ecp_nph * np.arange(ecp_nt_nobuf, dtype=int)
        ecp_wei = np.ones(ecp_nt_nobuf, dtype=float)
        ecp_ps = (ecp_pctr - np.pi) * np.ones(ecp_nt_nobuf, dtype=float)
        ecp_nphi = ecp_nph * np.ones(ecp_nt_nobuf, dtype=int)
        ecp_job.set_standard_geometry(ecp_nt_nobuf, ecp_nphi, ecp_thts, ecp_ps, ecp_ofs, ecp_wei)
        tim.add('sharp ecp job setup')
        # ----- calculation of the map to interpolate
        if spin > 0:
            ecp_m_resized = np.zeros((ecp_nt_nobuf + nt_buf_n + nt_buf_s, imax - imin + 1), dtype=complex)
            ecp_m = ecp_job.alm2map_spin([glm, clm if clm is not None else np.zeros_like(glm)], spin)
            ecp_m_resized[nt_buf_n:ecp_nt_nobuf + nt_buf_n] = (ecp_m[0] +  1j* ecp_m[1]).reshape( (ecp_nt_nobuf, ecp_nph))[:, imin:imax+1]
            tmp_shape = ecp_m_resized.shape
            ftype = complex
        else:
            ecp_m_resized = np.zeros((ecp_nt_nobuf + nt_buf_n + nt_buf_s, imax - imin + 1), dtype=float)
            ecp_m_resized[nt_buf_n:ecp_nt_nobuf + nt_buf_n] = ecp_job.alm2map(glm).reshape( (ecp_nt_nobuf, ecp_nph))[:, imin:imax+1]
            tmp_shape = (ecp_m_resized.shape[0], ecp_m_resized.shape[1] // 2 + 1)
            ftype = float
        for i in range(nt_buf_n):
            ecp_m_resized[i] = ecp_m_resized[2 * nt_buf_n - i]
        for i in range(nt_buf_s):
            ecp_m_resized[ecp_nt_nobuf + nt_buf_n + i] = ecp_m_resized[ecp_nt_nobuf + nt_buf_n - 1 - i]

        tim.add('alm2map lmax%s, resiz %s->%s'%(lmax, ecp_nph, ecp_m_resized.shape[1]))


        f = pyfftw.empty_aligned(ecp_m_resized.shape, dtype=ftype)
        tmp = pyfftw.empty_aligned(tmp_shape, dtype=complex)
        fft2 = pyfftw.FFTW(f, tmp, axes=(0, 1), direction='FFTW_FORWARD', threads=fftw_threads)
        ifft2 = pyfftw.FFTW(tmp, f, axes=(0, 1), direction='FFTW_BACKWARD', threads=fftw_threads)
        tim.add('fftw planning')

        fft2(pyfftw.byte_align(ecp_m_resized))
        tim.add('fftw fwd')

        # ----- bicucic prefiltering
        #6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nphi)) + 4.)
        wt = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(tmp_shape[0])) + 4.)
        wp = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(ecp_m_resized.shape[1])) + 4.)[:tmp_shape[1]]

        ifft2(tmp * np.outer(wt, wp))
        tim.add('bicubic prefilt, fftw bwd')

        dt = (th2 - th1) / (ecp_nt_nobuf - 1.)
        buf_t_bounds = [t_bounds[0] - nt_buf_n * dt, t_bounds[1] + nt_buf_s * dt]

        self._re_f = np.require(f.real, dtype=np.float64)
        self._im_f = np.require(f.imag, dtype=np.float64) if spin > 0 else None


        phir_min = imin * 2 * np.pi / ecp_nph
        phir_max = imax * 2 * np.pi / ecp_nph

        self._phir_min = phir_min
        self._phir_max = phir_max
        self._ecp_pctr = ecp_pctr
        self._prescal =  ( (imax - imin) / (phir_max - phir_min) )
        self._trescal =  ((ecp_m_resized.shape[0] - 1) / (buf_t_bounds[1] - buf_t_bounds[0]))
        self._buf_t_bounds = buf_t_bounds

        self.spin = spin
        self.tim = tim
        self.shape = self._re_f.shape
        self.patch = patch

        #temp
        #self.ma = ecp_m_resized
        print(tim)

    def phi2grid(self, phi):
        return (phi - self._ecp_pctr + np.pi - self._phir_min)%(2. * np.pi) * self._prescal

    def tht2grid(self, tht):
        return (tht - self._buf_t_bounds[0]) * self._trescal

    def eval_ongrid(self, t_grid, p_grid):
        if self.spin > 0:
            re_len = bicubic.deflect_omp(self._re_f, t_grid, p_grid)
            im_len = bicubic.deflect_omp(self._im_f, t_grid, p_grid)
            return re_len, im_len
        return bicubic.deflect_omp(self._re_f, t_grid, p_grid)

    def eval_ongrid_unchkd(self, t_grid, p_grid):
        """This can give somewhat faster interpolation (by ~20%) but can crash if too close (3 pixels) from boundaries


        """
        if self.spin > 0:
            re_len = bicubic.deflect_omp_unchkd(self._re_f, t_grid, p_grid)
            im_len = bicubic.deflect_omp_unchkd(self._im_f, t_grid, p_grid)
            return re_len, im_len
        return bicubic.deflect_omp_unchkd(self._re_f, t_grid, p_grid)

    def ecp_angles(self):
        """Returns colatitude and longitude (in rad) of the ecp map regular grid pixels


        """
        th1, th2 = self._buf_t_bounds
        ecp_nt, ecp_np = self._re_f.shape
        ecp_tht = th1 + (th2 - th1) / (ecp_nt - 1.) * np.arange(ecp_nt, dtype=float)
        ecp_phi = self._phir_min + np.arange(ecp_np, dtype=float) / (ecp_np - 1) * (self._phir_max - self._phir_min)
        ecp_phi += self._ecp_pctr - np.pi
        return ecp_tht, ecp_phi

    def __call__(self, tht, phi):
        return self.eval(tht, phi)

    def eval(self, tht, phi):
        """Returns interpolated spin field on input colatitude and longitude (in rad)


        """
        assert len(tht) == len(phi), (len(tht), len(phi))
        return self.eval_ongrid(self.tht2grid(tht),  self.phi2grid(phi))

    def eval_ders(self, tht, phi):
        """Crude estimate of derivatives (could be optimized a lot if needed)

        """
        tgrid = self.tht2grid(tht)
        pgrid = self.phi2grid(phi)
        dfdt = np.array(self.eval_ongrid(tgrid + 1, pgrid)) -np.array(self.eval_ongrid(tgrid - 1, pgrid) )
        dfdp = np.array(self.eval_ongrid(tgrid, pgrid + 1)) -np.array(self.eval_ongrid(tgrid, pgrid - 1) )
        return dfdt * 0.5 * self._trescal, dfdp * 0.5 * self._prescal


#spin, glm, t_bounds, targetres_amin, sht_threads, fftw_threads,
 #                p_bounds=(0., 2 * np.pi), clm=None, mmax=None

def lensgclm(targetres_amin, tht_bounds, p_bounds, hp_nside, glm, dlm, spin, lmax_len,
             cacher=cachers.cacher_mem(), fftw_threads=4, sht_threads=4, clm=None, mmax=None):
    """This lens a alm array on a patch of the curved-sky proceeding as follows:

        The alms are projected onto a ECP grid centred on the patch center, with colatitude extend as specified by the input.
        The ECP grid has minimal resolution as specified in input.
        If possible the full range longitude ECP map is restriced to a smaller subset.
        Bicubic spline interpolation if performed on this ECP map on a healpix map truncated to the relevant pixels
        (The cost of this is moderate and dominated by two FFT's of the ECP map size)
        Then the truncated map is sent back to harmonic domain with a cut-sky libsharp job.

        If not present in 'cacher' the new angles implied by dlm will be cached for further use
        (cached in grid units)

    """
    assert (0. <= tht_bounds[0] < tht_bounds[1] <= np.pi), tht_bounds
    assert (p_bounds[0] < p_bounds[1]) and (p_bounds[1] - p_bounds[0]) <= (2 * np.pi), p_bounds
    assert spin >= 0
    tim = timer(True)
    interpjob = spinfield_interpolator(spin, glm, tht_bounds, targetres_amin, sht_threads, fftw_threads,
                 p_bounds=p_bounds, clm=clm, mmax=mmax)
    tim += interpjob.tim
    tim.add('***total interpolator setup')
    # --- interpolation post-filtering
    hp_rings = interpjob.patch.hp_rings(hp_nside)
    hp_pixs, start_hpchunk = interpjob.patch.hp_pixels(hp_nside)
    tim.add('hp pixels and rings')

    fntp = 'dtdp'  # FIXME
    if not cacher.is_cached(fntp):
        # ----- Builds deflected position in ECP grid units (from 0 to nt and 0 to nphi, suitable for the bicubic prefilter)
        from itercurv.remapping.utils import d2ang
        #FIXME: use new d2ang differentiating poles
        lmax_dlm = hp.Alm.getlmax(dlm.size)

        job_hp_d1 = get_trhealpix_sharpjob(hp_nside, hp_rings, lmax_dlm)
        job_hp_d1.set_triangular_alm_info(lmax_dlm, lmax_dlm)
        job_hp_d1.set_nthreads(sht_threads)
        redimd = job_hp_d1.alm2map_spin([dlm, np.zeros_like(dlm)], 1)[:, hp_pixs]
        tht, phi = hp.pix2ang(hp_nside, hp_pixs + start_hpchunk)
        costn, phin = d2ang(redimd[0], redimd[1], np.cos(tht), phi, verbose=True)
        thtn = np.arccos(costn)
        del phi, tht, redimd
        phir_grid = interpjob.phi2grid(phin)
        thtn_grid = interpjob.tht2grid(thtn)
        #FIXME: unchecked?
        cacher.cache(fntp, np.array([thtn_grid, phir_grid]))
        tim.add('spin 1 d calc')

    # NB: could perform here unchecked interpolation, gain of 10-15% so not mindblowing
    hp_job = get_trhealpix_sharpjob(hp_nside, hp_rings, lmax_len, mmax=lmax_len if mmax is None else mmax)
    hp_job.set_nthreads(sht_threads)
    hp_job.set_triangular_alm_info(lmax_len, lmax_len if mmax is None else mmax)
    dt, dp = cacher.load(fntp)
    lenm = np.zeros((1 + (spin > 0), hp_job.n_pix()), dtype=float)
    lenm[:, hp_pixs] = interpjob.eval_ongrid(dt, dp)
    tim.add('interp %s pix' % hp_pixs.size)

    # --- going back to alm:
    if spin > 0:
        gclm_len = hp_job.map2alm_spin(lenm, spin)
    else:
        gclm_len = hp_job.map2alm(lenm.squeeze())
    tim.add('lenmap2alm lmax%s'%lmax_len)
    print(tim)
    return gclm_len

def lensgclm_band_ducc(sharpjob_fwd, sharpjob_bwd, nt, nphi, spin, thtnew, phinew, gclm, pyfftwt=4):
    #FIXME: thtnew, phinew to grid units
    tim = timer(True)
    assert sharpjob_fwd.n_pix() == nt * nphi
    assert sharpjob_bwd.n_pix() == thtnew.size and sharpjob_bwd.n_pix() == phinew.size, \
        ((sharpjob_fwd.n_pix(), thtnew.size),'not necessary we could spline only the center patch. '
        'However the gain in time is super low unless the ffts are also cut and it simplifies the filling of the map')
    assert gclm.ndim == 2 and gclm[0].size == sharpjob_fwd.n_alm()
    tim.add('asserts')
    sharpjob_fwd.set_nthreads(4)
    m = sharpjob_fwd.alm2map_spin(gclm, spin)
    tim.add('glm2map')
    #FIXME: could resize the map to something smaller prior to bicuic prefilter for interp
    w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(nt)) + 4.)
    w1 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(nphi)) + 4.)
    if pyfftwt <= 0: # f: filtmap to interpolate
        f = np.fft.fft2( (m[0] + 1j* m[1]).reshape( (nt, nphi)))
        tim.add('map2fvtm')
        f = np.fft.ifft2(f * np.outer(w0, w1))
        tim.add('fvtm2fmap')
    else: #pyfftw with 4 threads seems faster than ducc ffts with same threads
        f= pyfftw.empty_aligned((nt, nphi), dtype=complex)
        tmp = pyfftw.empty_aligned((nt, nphi), dtype=complex)
        fft2 = pyfftw.FFTW(f, tmp, axes=(0, 1), direction='FFTW_FORWARD', threads=pyfftwt)
        ifft2 = pyfftw.FFTW(tmp, f, axes=(0, 1), direction='FFTW_BACKWARD', threads=pyfftwt)

        f[:] = pyfftw.byte_align( (m[0] + 1j* m[1]).reshape( (nt, nphi ) ))
        fft2()
        tim.add('map2fvtm')
        ifft2(tmp * np.outer(w0, w1))
        tim.add('fvtm2fmap')
    #FIXME: can get rid of bounds checks in deflect, since we are conservative in englobing this stuff
    #FIXME: can also only interpolate only the relevant bit by feedin in some strides
    lenre = bicubic.deflect_omp(f.real, thtnew, phinew)
    lenim = bicubic.deflect_omp(f.imag, thtnew, phinew)
    tim.add('interp')
    tim.checkpoint(' gclm2lenmap')

    # going back to alm:
    gclm_len = sharpjob_bwd.map2alm_spin([lenre, lenim], 2)
    tim.add('lenmap2alm')
    print(tim)
    return gclm_len
