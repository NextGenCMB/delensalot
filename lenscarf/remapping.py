import scarf

from lenscarf.skypatch import skypatch
from lenscarf import interpolators as itp
from lenscarf.utils_remapping import d2ang
from lenscarf import cachers
from lenscarf.utils import timer
from lenscarf.utils_hp import Alm
from lenscarf.utils_scarf import Geom, pbounds as pbs
from lenscarf.fortran import remapping as fremap
import numpy as np

class deflection:
    def __init__(self, scarf_geometry:scarf.Geometry, targetres_amin, p_bounds:tuple, dlm, fftw_threads, scarf_threads,
                 cacher:cachers.cacher = cachers.cacher_none(), clm=None, mmax=None):
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
        lmax = Alm.getlmax(dlm.size, mmax)
        if mmax is None: mmax = lmax

        self.dlm = dlm
        self.clm = clm

        self.lmax_dlm = lmax
        self.mmax_dlm = mmax
        self.d1 = None # -- this might be instantiated later if needed
        self.cacher = cacher
        self.geom = scarf_geometry

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = tht_bounds
        self._pbds = pbs(p_bounds[0], p_bounds[1])  # (patch ctr, patch extent)
        self._resamin = targetres_amin
        self._sht_tr = scarf_threads
        self._fft_tr = fftw_threads
        self.tim = timer(True)

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
            nrings = self.geom.get_nrings()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            clm = np.zeros_like(self.dlm) if self.clm is None else self.clm
            red, imd = self.geom.alm2map_spin([self.dlm, clm], 1, self.lmax_dlm, self.mmax_dlm, self._sht_tr, [-1., 1.])
            thp_phip= np.zeros( (2, npix), dtype=float)
            startpix = 0
            for ir in range(nrings):
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if pixs.size > 0:
                    phis = Geom.phis(self.geom, ir)[self._pbds.contains(Geom.phis(self.geom, ir))]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.get_theta(ir) * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(red[pixs], imd[pixs], thts , phis, int(np.round(self.geom.get_cth(ir))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip[0, sli] = thtp_
                    thp_phip[1, sli] = phip_
                    startpix += len(pixs)
            self.cacher.cache(fn, thp_phip)
            assert startpix == npix, (startpix, npix)
            return thp_phip
        return self.cacher.load(fn)

    def _bwd_angles(self): #FIXME: feed the full map at once
        self._init_d1()
        (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = self.d1.get_spline_info()
        nrings = self.geom.get_nrings()
        npix = Geom.pbounds2npix(self.geom, self._pbds)
        rediimdi = np.zeros((2, npix), dtype=float)
        from plancklens import utils #FIXME
        startpix = 0
        buft = 1e20 # not too sure why this fails with a few degrees buffer
        for i, ir in utils.enumerate_progress(range(nrings)):
            pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
            phis = self._pbds.contains(Geom.phis(self.geom, ir))
            tht = self.geom.get_theta(ir)
            this_n = (tht - tht0) * t2grid # this ring in grid unit
            slt = slice(max( int(np.rint(this_n - buft)), 0), min( int(np.rint(this_n  + buft)), re_f.shape[0]))
            print(this_n, slt)
            redi, imdi = fremap.remapping.solve_pixs(re_f[slt,:] , im_f[slt,:], np.ones(len(pixs)) * tht, phis, tht0, phi0, t2grid, p2grid)
            sli = slice(startpix, startpix + len(pixs))
            rediimdi[0, sli] = redi
            rediimdi[1, sli] = imdi
            startpix += len(pixs)
        assert startpix == npix, (startpix, npix)
        return rediimdi

    def _bwd_magn(self):
        pass
        #M = None
        #for i, iband in utils.enumerate_progress(bands, label='Caching backw. magnification'):
        #    fname = 'bwd_detm_band%02d'%iband
        #    if not self.cacher.is_cached(fname):
        #        if M is None:
        #            plm_i, olm_i = self.ang2dlm(hp.Alm.getlmax(plm.size) + 100, bwd=True)
        #            k, (g1, g2), w = rpu.get_kappa_gamma_omega(plm_i, self.nside_lens, olm=olm_i, zbounds=self.zbounds)
        #            M = (1. - k) ** 2 - g1 ** 2 - g2 ** 2 + w ** 2
        #            del k, g1, g2, w
        #        self.cacher.cache(fname, M[self._get_pix(iband)])

    def lensgclm(self, glm, spin, lmax_out, backwards=False, clm=None, mmax=None, mmax_out=None):

        if mmax_out is None: mmax_out = lmax_out
        interpjob = self._build_interpolator(glm, spin, clm=clm, mmax=mmax)
        self.tim.add('glm spin %s lmax %s interpolator setup'%(spin, Alm.getlmax(glm.size, mmax)))
        # NB: could perform here unchecked interpolation, gain of 10-15% so not mindblowing
        thtn, phin = self._bwd_angles() if backwards else self._fwd_angles()
        self.tim.add('getting angles')

        npix = Geom.npix(self.geom)
        if thtn.size == npix:
            lenm = interpjob.eval(thtn, phin)
            self.tim.add('interpolation')

        else: # need to patch up the pixels. how to best do that?
            lenm_ = interpjob.eval(thtn, phin)
            self.tim.add('interpolation')
            lenm = np.zeros(npix if spin == 0 else (2, npix), dtype=float)
            start = 0
            for ir in range(self.geom.get_nrings()):
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if spin == 0:
                    lenm[pixs] = lenm_[start:start + pixs.size]
                else:
                    lenm[0, pixs] = lenm_[0][start:start + pixs.size]
                    lenm[1, pixs] = lenm_[1][start:start + pixs.size]
                start += pixs.size
            self.tim.add('truncated array filling')

        #FIXME: polarization rotation
        # --- going back to alm:
        if spin > 0:
            gclm_len = self.geom.map2alm_spin(lenm, spin, lmax_out, mmax_out, self._sht_tr, [-1.,1.])
        else:
            gclm_len = self.geom.map2alm(lenm, lmax_out, mmax_out, self._sht_tr, [-1.,1.])
        self.tim.add('map2alm spin %s lmaxout %s'%(spin, lmax_out))
        print(self.tim)
        return gclm_len