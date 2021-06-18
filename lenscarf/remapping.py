import scarf

from lenscarf.skypatch import skypatch
from lenscarf import interpolators as itp
from lenscarf.utils_remapping import d2ang, ang2d
from lenscarf import cachers
from lenscarf.utils import timer, clhash
from lenscarf.utils_hp import Alm
from lenscarf.utils_scarf import Geom, pbounds as pbs, scarfjob
from lenscarf.fortran import remapping as fremap
from lenscarf import utils_dlm
import numpy as np

class deflection:
    def __init__(self, scarf_geometry:scarf.Geometry, targetres_amin, p_bounds:tuple, dlm, fftw_threads, scarf_threads,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, mmax=None, verbose=True):
        """Deflection field object than can be used to lens several maps with forward or backward deflection

            Args:
                scarf_geometry: scarf.Geometry object holding info on the delfection operation pixelzation etc
                targetres_amin: float, desired interpolation resolution in arcmin
                p_bounds: tuple with longitude cuts info in the form of (patch center, patch extent), both in radians
                dlm: deflection-field alm array, gradient mode (:math:`\sqrt{L(L+1)}\phi_{LM}`)
                fftw_threads: number of threads for FFTWs transforms (other than the ones in SHTs)
                scarf_threads: number of threads for the SHTs scarf-ducc based calculations
                cacher: cachers.cacher instance allowing if desired caching of several pieces of info;
                        Useless if only one maps is intended to be deflected, but useful if more.
                dclm: deflection-field alm array, curl mode (if relevant)
                mmax: maximal m of the dlm / dclm arrays, if different from lmax


        """
        assert (p_bounds[1] > 0), p_bounds
        # --- interpolation of spin-1 deflection on the desired area and resolution
        tht_bounds = Geom.tbounds(scarf_geometry)
        assert (0. <= tht_bounds[0] < tht_bounds[1] <= np.pi), tht_bounds
        #self.sky_patch = skypatch(tht_bounds, p_bounds, targetres_amin, pole_buffers=3)
        lmax = Alm.getlmax(dlm.size, mmax)
        if mmax is None: mmax = lmax
        if cacher is None: cacher = cachers.cacher_none()

        self.dlm = dlm
        self.dclm = dclm

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
        self.tim = timer(True, prefix='deflection instance')
        self.verbose = verbose

    def hashdict(self):
        return {'lensgeom':Geom.hashdict(self.geom), 'resamin':self._resamin, 'pbs':self._pbds,
               'dlm':clhash(self.dlm.real), 'dclm': None if self.dclm is None else clhash(self.dclm.real)}


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
            self.d1 = self._build_interpolator(self.dlm, 1, clm=self.dclm, mmax=self.mmax_dlm)

    def _fwd_angles(self):
        """Builds deflected angles for the forawrd deflection field for the pixels inside the patch


        """
        fn = 'fwdang'
        if not self.cacher.is_cached(fn):
            nrings = self.geom.get_nrings()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self._sht_tr, [-1., 1.])
            thp_phip= np.zeros( (2, npix), dtype=float)
            startpix = 0
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if pixs.size > 0:
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.get_theta(ir) * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(red[pixs], imd[pixs], thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip[0, sli] = thtp_
                    thp_phip[1, sli] = phip_
                    startpix += len(pixs)
            self.cacher.cache(fn, thp_phip)
            assert startpix == npix, (startpix, npix)
            return thp_phip
        return self.cacher.load(fn)

    def _bwd_ring_angles(self, ir): # Single ring inversion, for test purposes
        self.tim.reset_t0()
        self._init_d1()
        (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = self.d1.get_spline_info()
        pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
        if pixs.size > 0:
            phis = Geom.phis(self.geom, ir)[[pixs - self.geom.ofs[ir]]]
            assert phis.size == pixs.size
            thts = self.geom.get_theta(ir) * np.ones(pixs.size)
            redi, imdi = fremap.remapping.solve_pixs(re_f, im_f, thts, phis, tht0, phi0, t2grid, p2grid)
            bwdang = d2ang(redi, imdi, thts, phis, int(np.rint(np.cos(self.geom.theta(ir)))))
            return bwdang
        else:
            return np.array([[],[]])

    def _bwd_angles(self):
        fn = 'bwdang'
        if not self.cacher.is_cached(fn):
            self.tim.reset_t0()
            self._init_d1()
            (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = self.d1.get_spline_info()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            nrings = self.geom.get_nrings()
            # build full angles arrays
            thts = np.empty(npix, dtype=float)
            phis = np.empty(npix, dtype=float)
            starts = np.zeros(nrings + 1, dtype=int)
            for i, ir in enumerate(np.argsort(self.geom.ofs)): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                starts[ir + 1] = starts[ir] + pixs.size
                if pixs.size > 0:
                    thts[starts[ir] : starts[ir + 1]] = self.geom.get_theta(ir) * np.ones(pixs.size)
                    phis[starts[ir] : starts[ir + 1]] = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
            # Perform deflection inversion with Fortran code
            redi, imdi = fremap.remapping.solve_pixs(re_f, im_f, thts, phis, tht0, phi0, t2grid, p2grid)
            bwdang = np.zeros((2, npix), dtype=float)
            # Inverse deflection to inverse angles
            # FIXME: cache here grid units etc
            for i, ir in enumerate(np.argsort(self.geom.ofs)): # We must follow the ordering of scarf position-space map
                vt = int(np.rint(np.cos(self.geom.theta[ir])))
                sli = slice(starts[ir], starts[ir + 1])
                bwdang[:, sli] = d2ang(redi[sli], imdi[sli], thts[sli], phis[sli], vt)
            self.cacher.cache(fn, bwdang)
            self.tim.add('bwd angles (full map)')
            return bwdang
        return self.cacher.load(fn)

    def _fwd_polrot(self):
        fn = 'fwdpolrot'
        if not self.cacher.is_cached(fn):
            self.tim.reset_t0()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self._sht_tr, [-1., 1.])
            d = np.sqrt(red ** 2 + imd ** 2)
            gamma = np.zeros(npix, dtype=float)
            startpix = 0
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if pixs.size > 0:
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size
                    sli = slice(startpix, startpix + len(pixs))
                    startpix += len(pixs)
                    assert 0 < self.geom.theta[ir] < np.pi, 'Fix this'
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    gamma[sli]  = np.arctan2(imd[pixs], red[pixs])
                    gamma[sli] -= np.arctan2(imd[pixs], d[pixs] * np.sin(d[pixs]) * cot + red[pixs] * np.cos(d[pixs]))
            assert startpix == npix, (startpix, npix)
            self.cacher.cache(fn,gamma)
            self.tim.add('polrot, fwd')
            return gamma
        return self.cacher.load(fn)

    def _fwd_magn(self):
        scjob = scarfjob()
        scjob.set_geometry(self.geom)
        scjob.set_triangular_alm_info(self.lmax_dlm, self.mmax_dlm)
        scjob.set_nthreads(self._sht_tr)
        return Geom.map2pbnmap(self.geom, utils_dlm.dlm2M(scjob, self.dlm, self.dclm), self._pbds)
        
    def _bwd_magn(self):
        """Builds inverse deflection magnification determinant


        """
        fn = 'bwdmagn'
        if not self.cacher.is_cached(fn):
            self.tim.reset_t0()
            scjob = scarfjob()
            scjob.set_geometry(self.geom)
            scjob.set_triangular_alm_info(self.lmax_dlm, self.mmax_dlm)
            scjob.set_nthreads(self._sht_tr)
            thti, phii = self._bwd_angles()
            redimd = np.zeros((2, Geom.npix(scjob.geom)), dtype=float)
            start = 0
            for it in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, it, self._pbds)
                if pixs.size > 0:
                    phis = Geom.phis(self.geom, it)[pixs - self.geom.ofs[it]]
                    sli = slice(start, start+pixs.size)
                    redimd[:, pixs] = ang2d(thti[sli], self.geom.theta[it] * np.ones(pixs.size), phii[sli] -phis)
                    start += pixs.size
            assert start == thti.size
            self.tim.add('collecting red imd for Mi')
            dlm, dclm = scjob.map2alm_spin(redimd, 1)
            Mi = Geom.map2pbnmap(self.geom, utils_dlm.dlm2M(scjob, dlm, dclm), self._pbds)
            self.tim.add('Mi SHTs')
            self.cacher.cache(fn, Mi)
        return self.cacher.load(fn)

    def _bwd_polrot(self):
        """Builds inverse deflection polarisation rotation angles


        """
        fn = 'bwdpolrot'
        if not self.cacher.is_cached(fn):
            self.tim.reset_t0()
            scjob = scarfjob()
            scjob.set_geometry(self.geom)
            scjob.set_triangular_alm_info(self.lmax_dlm, self.mmax_dlm)
            scjob.set_nthreads(self._sht_tr)
            thti, phii = self._bwd_angles()
            gamma = np.zeros(Geom.pbounds2npix(self.geom, self._pbds), dtype=float)
            start = 0
            for it in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, it, self._pbds)
                if pixs.size > 0:
                    phis = Geom.phis(self.geom, it)[pixs - self.geom.ofs[it]]
                    sli = slice(start, start+pixs.size)
                    red, imd = ang2d(thti[sli], self.geom.theta[it] * np.ones(pixs.size), phii[sli] -phis)
                    assert 0 < self.geom.theta[it] < np.pi, 'Fix this'
                    cot = np.cos(self.geom.theta[it]) / np.sin(self.geom.theta[it])
                    d = np.sqrt(red ** 2 + imd ** 2)
                    gamma[sli]  = np.arctan2(imd, red)
                    gamma[sli] -= np.arctan2(imd, d * np.sin(d) * cot + red * np.cos(d))
                    start += pixs.size

            assert start == thti.size
            self.tim.add('bwd polrot')
            self.cacher.cache(fn, gamma)
        return self.cacher.load(fn)

    def lensgclm(self, glm, spin, lmax_out, backwards=False, clm=None, mmax=None, mmax_out=None):
        #TODO: save only grid angles and full phase factor
        if mmax_out is None: mmax_out = lmax_out
        self.tim.reset_t0()
        interpjob = self._build_interpolator(glm, spin, clm=clm, mmax=mmax)
        self.tim.add('glm spin %s lmax %s interpolator setup'%(spin, Alm.getlmax(glm.size, mmax)))
        thtn, phin = self._bwd_angles() if backwards else self._fwd_angles()
        self.tim.add('getting angles')

        lenm_pbded = interpjob.eval(thtn, phin)
        self.tim.add('interpolation')
        if spin == 0:
            if backwards:
                lenm_pbded *= self._bwd_magn()
                self.tim.add('det Mi')
            lenm =  Geom.pbdmap2map(self.geom, lenm_pbded, self._pbds)
        else:
            gamma = self._bwd_polrot if backwards else self._fwd_polrot
            lenm_pbded = np.exp(1j * spin * gamma()) * (lenm_pbded[0] + 1j * lenm_pbded[1])
            self.tim.add('pol rot')
            if backwards:
                lenm_pbded *= self._bwd_magn()
                self.tim.add('det Mi')
            lenm = [Geom.pbdmap2map(self.geom, lenm_pbded.real, self._pbds),
                    Geom.pbdmap2map(self.geom, lenm_pbded.imag, self._pbds)]
        self.tim.add('truncated array filling')
        if spin > 0:
            gclm_len = self.geom.map2alm_spin(lenm, spin, lmax_out, mmax_out, self._sht_tr, [-1.,1.])
        else:
            gclm_len = self.geom.map2alm(lenm, lmax_out, mmax_out, self._sht_tr, [-1.,1.])
        self.tim.add('map2alm spin %s lmaxout %s nrings %s'%(spin, lmax_out, self.geom.get_nrings()))
        if self.verbose:
            print(self.tim)
        return gclm_len