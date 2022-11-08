"""This module contains the main curved-sky spin-field interpolators

# TODO flexible fft module
"""
from lenscarf.utils import timer
from lenscarf.utils_hp import Alm
from lenscarf.skypatch import skypatch

from lenscarf.utils_scarf import scarfjob, Geom
import numpy as np
import pyfftw
from lenscarf.fortran import bicubic

class bicubic_ecp_interpolator:
    """Spin-weighted bicubic spline interpolator on a patch of the sky.

        Args:
            spin: spin-weight of the map to interpolate
            gclm: gradient and curl mode of the map to interpolate (healpix/py format). For spin-0 1-d array
            mmax: maxmimal m defining the glm layout (defaults to lmax if None)
            patch: skypatch instance defining the region boundaries and sampling resolution
            sht_threads: numbers of OMP threads to perform shts with (scarf / ducc)
            fftw_threads: numbers of threads for FFT's
            fftw_flags: fftw planning flags (see pyfftw doc and note below)
            ns_symmetrize: if set, will also interpolate the patch symmetric wrt the equator
                            (can be used to speed up SHTs)


        Note:
            The first and last pixels (of the first dimension) match the input colatitude bounds exactly
            The longitude bounds (in the second dimension) will be buffered by some small amount and will not match pixels exactly


        Note:
            On first instantiation pyfftw might spend some extra time calculating a FFT plan. This depends on fftw_flags.
            A rule of thumb might be to use ('FFTW_ESTIMATE',) if this instance is going to be used only a few times,
            in which case no time is spent on the planning, and ('FFTW_MEASURE',) if this interpolator is going to be used again and again.
            This typically leads to ~30 sec planning time for twice faster transforms in many configurations


    """
    def __init__(self, spin:int, gclm:np.ndarray or list, mmax:int or None, patch:skypatch, sht_threads:int,
                 fftw_threads:int, fftw_flags=('FFTW_ESTIMATE',),
                 verbose=False, ns_symmetrize=False):
        # Defines latitude grid from input bounds:
        assert spin >= 0, spin
        tim = timer(True, prefix='spinfield bicubic interpolator')
        # ---- We build an ECP patch on the specified lat and lon bounds
        ecp_nt, ecp_nph = patch.ecp_ring_ntphi()
        # North and south pol buffers, if relevant (need a periodic patch)
        nt_buf_n, nt_buf_s = patch.nt_buffers_n, patch.nt_buffers_s
        ecp_nt_nobuf = ecp_nt - nt_buf_n - nt_buf_s
        imin, imax = patch.ecp_resize(ecp_nph)

        lmax = Alm.getlmax( (gclm[0] if spin > 0 else gclm).size, mmax)
        if mmax is None : mmax = lmax
        # ------ Build custom scarf job with patch center on the middle of ECP map
        ecp_job = scarfjob()
        ecp_job.set_triangular_alm_info(lmax, mmax)
        ecp_job.set_nthreads(sht_threads)
        if ns_symmetrize and np.abs(np.max(patch.tbounds[1]) - (np.pi - np.max(patch.tbounds[0]))) > 1e-14:# else already symmetric
            assert patch.tbounds[1] <= np.pi * 0.5 or patch.tbounds[0] >= np.pi * 0.5, 'patch crosses equator, you probably did not mean to use this in conjonction with ns_symmetrize'
            # We will have two ECP maps, but one extended SHT geometry
            ecp_geom1 = Geom.get_ecp_geometry(ecp_nt_nobuf, ecp_nph, phi_center=patch.pbounds[0], tbounds=patch.tbounds)
            ecp_geom2 = Geom.get_ecp_geometry(ecp_nt_nobuf, ecp_nph, phi_center=patch.pbounds[0], tbounds=(np.pi - patch.tbounds[1], np.pi - patch.tbounds[0]))
            ecp_job.set_geometry(Geom.merge([ecp_geom1, ecp_geom2]))
            npix = Geom.npix(ecp_job.geom)
            slices = [slice(0, npix // 2), slice(npix//2, npix)]
            self.singlepatch = False
        else:
            # One ECP map and SHT geometry
            ecp_job.set_ecp_geometry(ecp_nt_nobuf, ecp_nph, phi_center=patch.pbounds[0], tbounds=patch.tbounds)
            npix = Geom.npix(ecp_job.geom)
            slices = [slice(0, npix)]
            self.singlepatch = True

        tim.add('scarf ecp job setup')
        # ----- calculation of the map to interpolate
        self._re_f = []
        self._im_f = []
        dt = np.abs((patch.tbounds[1] - patch.tbounds[0]) / (ecp_nt_nobuf - 1.))
        # SHT:
        ecp_m = ecp_job.alm2map_spin(gclm, spin) if  spin > 0 else ecp_job.alm2map(gclm)
        for isli, slic in enumerate(slices):
            tbuf_n = nt_buf_n if isli == 0 else nt_buf_s
            tbuf_s = nt_buf_s if isli == 0 else nt_buf_n
            if spin > 0:
                ecp_m_resized = np.zeros((ecp_nt_nobuf + tbuf_n + tbuf_s, imax - imin + 1), dtype=complex)
                ecp_m_resized[tbuf_n:ecp_nt_nobuf + tbuf_n] = (ecp_m[0,slic] +  1j* ecp_m[1,slic]).reshape( (ecp_nt_nobuf, ecp_nph))[:, imin:imax+1]
                tmp_shape = ecp_m_resized.shape
                ftype = complex
            else:
                ecp_m_resized = np.zeros((ecp_nt_nobuf + tbuf_n + tbuf_s, imax - imin + 1), dtype=float)
                ecp_m_resized[tbuf_n:ecp_nt_nobuf + tbuf_n] = ecp_m[slic].reshape( (ecp_nt_nobuf, ecp_nph))[:, imin:imax+1]
                tmp_shape = (ecp_m_resized.shape[0], ecp_m_resized.shape[1] // 2 + 1)
                ftype = float
            for i in range(tbuf_n): # north pole buffers if relevant
                ecp_m_resized[i] = ecp_m_resized[2 * tbuf_n - i]
            for i in range(tbuf_s): # south pole buffers if relevant
                ecp_m_resized[ecp_nt_nobuf + tbuf_n + i] = ecp_m_resized[ecp_nt_nobuf + tbuf_n - 1 - i]

            tim.add('alm2map lmax%s, resiz %s->%s'%(lmax, ecp_nph, ecp_m_resized.shape[1]))


            f = pyfftw.empty_aligned(ecp_m_resized.shape, dtype=ftype)
            tmp = pyfftw.empty_aligned(tmp_shape, dtype=complex)
            fft2 = pyfftw.FFTW(f, tmp, axes=(0, 1), direction='FFTW_FORWARD', threads=fftw_threads, flags=fftw_flags)
            ifft2 = pyfftw.FFTW(tmp, f, axes=(0, 1), direction='FFTW_BACKWARD', threads=fftw_threads, flags=fftw_flags)
            tim.add('fftw planning')

            fft2(pyfftw.byte_align(ecp_m_resized))
            tim.add('fftw fwd')

            # ----- bicucic prefiltering
            wt = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(tmp_shape[0])) + 4.)
            wp = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(ecp_m_resized.shape[1])) + 4.)[:tmp_shape[1]]

            ifft2(tmp * np.outer(wt, wp))
            tim.add('bicubic prefilt, fftw bwd')

            self._re_f.append(np.require(f.real, dtype=np.float64))
            self._im_f.append(np.require(f.imag, dtype=np.float64) if spin > 0 else None)


        phir_min = imin * 2 * np.pi / ecp_nph
        phir_max = imax * 2 * np.pi / ecp_nph

        # do I need all this stuff?
        self._phir_min = phir_min
        self._phir_max = phir_max
        self._ecp_pctr = patch.pbounds[0]
        self._prescal =  ( (imax - imin) / (phir_max - phir_min) )
        self._buf_t_bounds = (patch.tbounds[0] - nt_buf_n * dt, patch.tbounds[1] + nt_buf_s * dt)
        self._trescal =  ((self._re_f[0].shape[0] - 1) / (self._buf_t_bounds[1] - self._buf_t_bounds[0]))

        self.spin = spin
        self.tim = tim
        self.shape = self._re_f[0].shape
        self.patch = patch

        #temp
        self.ecpjob = ecp_job
        if verbose:
            print(tim)

    def phi2grid(self, phi):
        return (phi - self._ecp_pctr + np.pi - self._phir_min)%(2. * np.pi) * self._prescal

    def tht2grid(self, tht):
        if self.singlepatch:
            return (tht - self._buf_t_bounds[0]) * self._trescal
        assert tht.size%2 == 0
        ret = np.empty_like(tht)
        ret[:tht.size//2] = tht[0:tht.size//2] - self._buf_t_bounds[0]
        ret[tht.size//2:] = tht[tht.size//2:] -  (np.pi - self._buf_t_bounds[1])
        return ret * self._trescal


    def eval_ongrid(self, t_grid, p_grid):
        if self.singlepatch:
            if self.spin > 0:
                return bicubic.deflect_omp(self._re_f[0], t_grid, p_grid), bicubic.deflect_omp(self._im_f[0], t_grid, p_grid)
            return bicubic.deflect_omp(self._re_f[0], t_grid, p_grid)
        Nt = t_grid.size
        assert Nt % 2 == 0
        slices = (slice(0, Nt // 2), slice(Nt // 2, Nt))
        assert len(self._re_f) == 2
        if self.spin > 0:
            ret_re = np.empty_like(t_grid)
            ret_im = np.empty_like(t_grid)
            for ref, imf, slic in zip(self._re_f, self._im_f, slices):
                ret_re[slic] = bicubic.deflect_omp(ref, t_grid[slic], p_grid[slic])
                ret_im[slic] = bicubic.deflect_omp(imf, t_grid[slic], p_grid[slic])
            return ret_re, ret_im
        ret_re = np.empty_like(t_grid)
        for ref, slic in zip(self._re_f, slices):
            ret_re[slic] = bicubic.deflect_omp(ref, t_grid[slic], p_grid[slic])
        return ret_re

    def eval_ongrid_unchkd(self, t_grid, p_grid):
        """This can give somewhat faster interpolation (by ~20%) but can crash if too close (3 pixels) from boundaries


        """
        assert self.singlepatch, 'implement ns_symm version'
        if self.spin > 0:
            re_len = bicubic.deflect_omp_unchkd(self._re_f[0], t_grid, p_grid)
            im_len = bicubic.deflect_omp_unchkd(self._im_f[0], t_grid, p_grid)
            return re_len, im_len
        return bicubic.deflect_omp_unchkd(self._re_f[0], t_grid, p_grid)

    def ecp_angles(self):
        """Returns colatitude and longitude (in rad) of the ecp map regular grid pixels


        """
        th1, th2 = self._buf_t_bounds
        ecp_nt, ecp_np = self._re_f[0].shape
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


class polar_interpolator:
    def __init__(self, spin, glm, patch:skypatch, sht_threads, fftw_threads, clm=None, mmax=None):
        pass