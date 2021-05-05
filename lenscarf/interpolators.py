"""This module contains the main curved-sky spin-field interpolators


"""
from lenscarf.utils import timer
from lenscarf.utils_hp import getlmax
from lenscarf.skypatch import skypatch
from lenscarf.utils_scarf import scarfjob
import numpy as np
import pyfftw
from omppol.src import bicubic #:FIXME

class bicubic_ecp_interpolator:
    """Spin-weighted bicubic spline interpolator on a patch of the sky.

        Args:
            spin: spin-weight of the map to interpolate
            glm: gradient-mode of the map to interpolate (healpix/py format)
            t_bounds:  co-latitude bounds (rad) of the patch to do the interpolation on
            targetres_amin: target resolution of the interpolation operation
            sht_threads: numbers of OMP threads to perform shts with (scarf / ducc)
            fftw_threads: numbers of threads for FFT's
            clm: curl mode of map to interpolate, if relevant

        Note:
            The first and last pixels (of the first dimension) match the input colatitude bounds exactly
            The longitude bounds (in the second dimension) will be buffered by some small amount and will not match pixels exactly

        Note:
            On first instantiation pyfftw spends some extra time calculating a FFT plan

    """
    def __init__(self, spin, glm, patch:skypatch, sht_threads, fftw_threads, clm=None, mmax=None):
        # Defines latitude grid from input bounds:
        assert spin >= 0, spin
        #assert 0<= t_bounds[0] < t_bounds[1] <= np.pi, t_bounds
        #assert p_bounds[0] < p_bounds[1], p_bounds
        #assert (p_bounds[1] - p_bounds[0]) <= 2 * np.pi, p_bounds

        tim = timer(True, prefix='spinfield bicubic interpolator')
        # ---- We build an ECP patch on the specified lat and lon bounds
        ecp_nt, ecp_nph = patch.ecp_ring_ntphi()
        # North and south pol buffers, if relevant (need a periodic patch)
        nt_buf_n, nt_buf_s = patch.nt_buffers_n, patch.nt_buffers_s
        ecp_nt_nobuf = ecp_nt - nt_buf_n - nt_buf_s
        imin, imax = patch.ecp_resize(ecp_nph)

        lmax = getlmax(glm.size)
        # ------ Build custom scarf job with patch center on the middle of ECP map
        ecp_job = scarfjob()
        ecp_job.set_triangular_alm_info(lmax, lmax if mmax is None else mmax)
        ecp_job.set_nthreads(sht_threads)
        ecp_job.set_ecp_geometry(ecp_nt_nobuf, ecp_nph, phi_center=patch.pbounds[0], tbounds=patch.tbounds)
        tim.add('scarf ecp job setup')
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
        for i in range(nt_buf_n): # north pole buffers if relevant
            ecp_m_resized[i] = ecp_m_resized[2 * nt_buf_n - i]
        for i in range(nt_buf_s): # south pole buffers if relevant
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
        wt = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(tmp_shape[0])) + 4.)
        wp = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(ecp_m_resized.shape[1])) + 4.)[:tmp_shape[1]]

        ifft2(tmp * np.outer(wt, wp))
        tim.add('bicubic prefilt, fftw bwd')

        dt = (patch.tbounds[1] - patch.tbounds[0]) / (ecp_nt_nobuf - 1.)
        buf_t_bounds = [patch.tbounds[0] - nt_buf_n * dt, patch.tbounds[1] + nt_buf_s * dt]

        self._re_f = np.require(f.real, dtype=np.float64)
        self._im_f = np.require(f.imag, dtype=np.float64) if spin > 0 else None


        phir_min = imin * 2 * np.pi / ecp_nph
        phir_max = imax * 2 * np.pi / ecp_nph

        #FIXME: do I need all this stuff?
        self._phir_min = phir_min
        self._phir_max = phir_max
        self._ecp_pctr = patch.pbounds[0]
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

    def get_spline_info(self):
        """Returns splining info that could be passed to optimized code

            Note:
                Do not change any of the outputs

        """
        tht0 = self._buf_t_bounds[0]
        phi0 = self._ecp_pctr - np.pi + self._phir_min
        t2grid = self._trescal
        p2grid = self._prescal
        return (tht0, t2grid), (phi0, p2grid), (self._re_f, self._im_f)

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