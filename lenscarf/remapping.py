from __future__ import annotations

import time
import numpy as np

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from lenscarf.skypatch import skypatch
from lenscarf import interpolators as itp
from lenscarf.utils_remapping import d2ang, ang2d
from lenscarf import cachers
from lenscarf.utils import timer, clhash
from lenscarf.utils_hp import Alm, alm2cl, alm_copy
from lenscarf.utils_scarf import Geom, scarfjob, pbdGeometry
from lenscarf.fortran import remapping as fremap
from lenscarf import utils_dlm
import scarf

class deflection:
    def __init__(self, scarf_pbgeometry:pbdGeometry, targetres_amin, dglm,
                 mmax_dlm:int or None, fftw_threads:int, scarf_threads:int,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, verbose=False, fftw_flags=('FFTW_ESTIMATE',)):
        """Deflection field object than can be used to lens several maps with forward or backward deflection

            Args:
                scarf_pbgeometry: scarf.Geometry object holding info on the deflection operation pixelization
                targetres_amin: float, desired interpolation resolution in arcmin
                dglm: deflection-field alm array, gradient mode (:math:`\sqrt{L(L+1)}\phi_{LM}`)
                fftw_threads: number of threads for FFTWs transforms (other than the ones in SHTs)
                scarf_threads: number of threads for the SHTs scarf-ducc based calculations
                cacher: cachers.cacher instance allowing if desired caching of several pieces of info;
                        Useless if only one maps is intended to be deflected, but useful if more.
                dclm: deflection-field alm array, curl mode (if relevant)
                mmax_dlm: maximal m of the dlm / dclm arrays, if different from lmax


        """
        lmax = Alm.getlmax(dglm.size, mmax_dlm)
        if mmax_dlm is None: mmax_dlm = lmax
        if cacher is None: cacher = cachers.cacher_none()


        # std deviation of deflection:
        s2_d = np.sum(alm2cl(dglm, dglm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1) ) / (4 * np.pi)
        if dclm is not None:
            s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1) ) / (4 * np.pi)
        sig_d = np.sqrt(s2_d)
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%(sig_d/np.pi * 180 * 60))
        log.info(" Deflection std %.2e amin"%(sig_d / np.pi * 180 * 60))
        self.sig_d = sig_d
        self.dlm = dglm
        self.dclm = dclm

        self.lmax_dlm = lmax
        self.mmax_dlm = mmax_dlm
        self.d1 = None # -- this might be instantiated later if needed
        self.cacher = cacher
        self.pbgeom = scarf_pbgeometry
        self.geom = scarf_pbgeometry.geom

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = Geom.tbounds(scarf_pbgeometry.geom)
        self._pbds = scarf_pbgeometry.pbound  # (patch ctr, patch extent)
        self._resamin = targetres_amin
        self.sht_tr = scarf_threads
        self._fft_tr = fftw_threads
        self.fftw_flags = fftw_flags

        self.tim = timer(True, prefix='deflection instance')
        self.verbose = verbose

    def hashdict(self):
        return {'lensgeom':Geom.hashdict(self.geom), 'resamin':self._resamin, 'pbs':self._pbds,
               'dlm':clhash(self.dlm.real), 'dclm': None if self.dclm is None else clhash(self.dclm.real)}

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        """Returns a deflection instance for another deflection field and cacher with same parameters than self


        """
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.pbgeom, self._resamin, dlm[0], mmax_dlm, self._fft_tr, self.sht_tr, cacher, dlm[1],
                          verbose=self.verbose)

    def change_geom(self, pbgeom:pbdGeometry, cacher:cachers.cacher or None=None):
        """Returns a deflection instance with a different position-space geometry

                Args:
                    pbgeom: new pbounded-scarf geometry
                    cacher: cacher instance if desired


        """
        return deflection(pbgeom, self._resamin, self.dlm, self.mmax_dlm, self._fft_tr, self.sht_tr, cacher, self.dclm,
                          verbose=self.verbose)

    def _build_interpolator(self, gclm, mmax:int or None, spin:int):
        bufamin = 30.
        if self.verbose: print("***instantiating spin-%s interpolator with %s amin buffers"%(spin, bufamin))
        # putting a d = 0.01 ~ 30 arcmin buffer which should be way more than enough
        buf = bufamin/ 180 / 60 * np.pi
        srted_tht = np.sort(self.geom.theta)
        assert self._tbds == (srted_tht[0], srted_tht[-1]), (self._tbds)
        symmetric = np.all(np.abs(srted_tht - (np.pi - srted_tht)[::-1]) < 1e-14)
        largegap = np.min(np.abs(srted_tht - np.pi * 0.5)) > buf
        if symmetric and largegap:
            tbds = (max(self._tbds[0] - buf, 0.), srted_tht[len(srted_tht)//2 - 1] + buf)
            assert tbds[1] < 0.5 * np.pi, tbds
            assert np.all(np.argsort(self.geom.ofs) == np.argsort(self.geom.theta)), 'I believe we need the rings to be ordered'
            if self.verbose:
                print("building a symmetric pair of interpolators")
        else:
            tbds = (max(self._tbds[0] - buf, 0.), min(np.pi, self._tbds[1] + buf))
        sintmin = np.min(np.sin(self._tbds))
        prange = min(self._pbds.get_range() + 2 * buf / sintmin if sintmin > 0 else 2 * np.pi, 2 * np.pi)
        buffered_patch = skypatch(tbds, (self._pbds.get_ctr(), prange), self._resamin, pole_buffers=3)
        return itp.bicubic_ecp_interpolator(spin, gclm, mmax, buffered_patch, self.sht_tr, self._fft_tr,
                                            ns_symmetrize=symmetric * largegap, verbose=self.verbose, fftw_flags=self.fftw_flags)

    def _init_d1(self):
        if self.d1 is None and self.sig_d > 0.:
            gclm = [self.dlm, np.zeros_like(self.dlm) if self.dclm is None else self.dclm]
            self.d1 = self._build_interpolator(gclm, self.mmax_dlm, 1)

    def fill_map(self, functions:list[callable], dtype=float):
        """Iterates over rings to produce output maps functions of the deflections

            Args:
                functions: list of callable each with arguments red, imd, and theta
                dtype: dtype of the output maps

            Returns:
                (len(functions), npix)-shaped array (squeezed)

        """
        startpix = 0
        npix = Geom.pbounds2npix(self.geom, self._pbds)
        dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
        red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
        m = np.zeros((len(functions),npix), dtype=dtype)
        cost, sint = np.cos(self.geom.theta), np.sin(self.geom.theta)
        for ir in np.argsort(self.geom.ofs):  # We must follow the ordering of scarf position-space map
            pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
            if pixs.size > 0:
                phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                assert phis.size == pixs.size, (phis.size, pixs.size)
                sli = slice(startpix, startpix + len(pixs))
                for ifu, func in enumerate(functions):
                    m[ifu, sli] = func(red[pixs], imd[pixs], self.geom.theta[ir], cost[ir], sint[ir])
                startpix += len(pixs)
        assert startpix == npix, (startpix, npix)
        return m.squeeze()

    def _fwd_angles(self):
        """Builds deflected angles for the forward deflection field for the pixels inside the patch


        """
        fn = 'fwdang'
        if not self.cacher.is_cached(fn):
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
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

            t0 = time.time()
            self.tim.reset_t0()
            self._init_d1()
            #TODO: this will for the (unexpected here) ns_symmetrized interpolator cases
            (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = self.d1.get_spline_info()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            nrings = self.geom.get_nrings()
            # build full angles arrays
            thts = np.empty(npix, dtype=float)
            phis = np.empty(npix, dtype=float)
            starts = np.zeros(nrings + 1, dtype=int)
            for i, ir in enumerate(np.argsort(self.geom.ofs)):  # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                starts[i + 1] = starts[i] + pixs.size
                if pixs.size > 0:
                    thts[starts[i]: starts[i + 1]] = self.geom.get_theta(ir) * np.ones(pixs.size)
                    phis[starts[i]: starts[i + 1]] = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
            # Perform deflection inversion with Fortran code
            redi, imdi = fremap.remapping.solve_pixs(re_f, im_f, thts, phis, tht0, phi0, t2grid, p2grid)
            bwdang = np.zeros((2, npix), dtype=float)
            # Inverse deflection to inverse angles
            # FIXME: cache here grid units etc
            for i, ir in enumerate(np.argsort(self.geom.ofs)): # We must follow the ordering of scarf position-space map
                vt = int(np.rint(np.cos(self.geom.theta[ir])))
                sli = slice(starts[i], starts[i + 1])
                bwdang[:, sli] = d2ang(redi[sli], imdi[sli], thts[sli], phis[sli], vt)
            self.cacher.cache(fn, bwdang)
            log.info('bwd angles calc, %.2e secs'%(time.time() - t0))
            self.tim.add('bwd angles (full map)')
            return bwdang
        return self.cacher.load(fn)

    def _fwd_polrot(self):
        fn = 'fwdpolrot'
        if not self.cacher.is_cached(fn):
            self.tim.reset_t0()
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
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
        scjob.set_nthreads(self.sht_tr)
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
            scjob.set_nthreads(self.sht_tr)
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
            scjob.set_nthreads(self.sht_tr)
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

    def gclm2lenpixs(self, gclm:np.ndarray or list, mmax:int or None, spin:int, pixs:np.ndarray[int], backwards:bool, nomagn=False):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation

            Note:
                The number of pixels must be small here, otherwise way too slow

            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so make take some time.

        """
        thts, phis = self._bwd_angles()[:, pixs] if backwards else self._fwd_angles()[:, pixs]
        nph = 2 * np.ones(thts.size, dtype=int) # I believe at least 2 points per ring if using scarf
        ofs = 2 * np.arange(thts.size, dtype=int)
        wt = np.ones(thts.size)
        geom = scarf.Geometry(thts.size, nph, ofs, 1, phis.copy(), thts.copy(), wt) #copy necessary as this goes to C
        #thts.size, nph, ofs, 1, phi0, thts, wt
        if abs(spin) > 0:
            lmax = Alm.getlmax(gclm[0].size, mmax)
            if mmax is None: mmax = lmax
            QU = geom.alm2map_spin(gclm, spin, lmax, mmax, self.sht_tr, [-1., 1.])[:, 0::2]
            gamma = self._bwd_polrot()[pixs] if backwards else self._fwd_polrot()[pixs]
            QU = np.exp(1j * spin * gamma) * (QU[0] + 1j * QU[1])
            if backwards and not nomagn:
                QU *= self._bwd_magn()[pixs]
            return QU.real, QU.imag
        lmax = Alm.getlmax(gclm.size, mmax)
        if mmax is None: mmax = lmax
        T = geom.alm2map(gclm, lmax, mmax, self.sht_tr, [-1., 1.])[0::2]
        if backwards and not nomagn:
            T *= self._bwd_magn()[pixs]
        return T

    def gclm2lenmap(self, gclm:np.ndarray or list, mmax:int or None, spin, backwards:bool, nomagn=False):
        if self.sig_d <= 0:
            if abs(spin) > 0:
                lmax = Alm.getlmax(gclm[0].size, mmax)
                if mmax is None: mmax = lmax
                return self.geom.alm2map_spin(gclm, spin, lmax, mmax, self.sht_tr, [-1., 1.])
            else:
                lmax = Alm.getlmax(gclm.size, mmax)
                if mmax is None: mmax = lmax
                return self.geom.alm2map(gclm, lmax, mmax, self.sht_tr, [-1., 1.])
        # TODO: consider saving only grid angles and full phase factor
        self.tim.reset_t0()
        interpjob = self._build_interpolator(gclm, mmax, spin)
        self.tim.add('glm spin %s lmax %s interpolator setup' % (
        spin, Alm.getlmax((gclm[0] if abs(spin) > 0 else gclm).size, mmax)))
        thtn, phin = self._bwd_angles() if backwards else self._fwd_angles()
        self.tim.add('getting angles')

        lenm_pbded = interpjob.eval(thtn, phin)
        self.tim.add('interpolation')
        if spin == 0:
            if backwards and not nomagn:
                lenm_pbded *= self._bwd_magn()
                self.tim.add('det Mi')
            lenm = Geom.pbdmap2map(self.geom, lenm_pbded, self._pbds)
        else:
            gamma = self._bwd_polrot if backwards else self._fwd_polrot
            lenm_pbded = np.exp(1j * spin * gamma()) * (lenm_pbded[0] + 1j * lenm_pbded[1])
            self.tim.add('pol rot')
            if backwards and not nomagn:
                lenm_pbded *= self._bwd_magn()
                self.tim.add('det Mi')
            lenm = [Geom.pbdmap2map(self.geom, lenm_pbded.real, self._pbds),
                    Geom.pbdmap2map(self.geom, lenm_pbded.imag, self._pbds)]
        self.tim.add('truncated array filling')
        return lenm

    def lensgclm(self, gclm:np.ndarray or list, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, nomagn=False):
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0: # no actual deflection
            if spin == 0:
                return alm_copy(gclm, mmax, lmax_out, mmax_out)
            glmret = alm_copy(gclm[0], mmax, lmax_out, mmax_out)
            return np.array([glmret, alm_copy(gclm[1], mmax, lmax_out, mmax_out) if gclm[1] is not None else np.zeros_like(glmret)])
        self.tim.reset_t0()
        lenm = self.gclm2lenmap(gclm, mmax, spin, backwards, nomagn=nomagn)
        self.tim.add('gclm2lenmap, total')
        if mmax_out is None: mmax_out = lmax_out
        if spin > 0:
            gclm_len = self.geom.map2alm_spin(lenm, spin, lmax_out, mmax_out, self.sht_tr, [-1., 1.])
        else:
            gclm_len = self.geom.map2alm(lenm, lmax_out, mmax_out, self.sht_tr, [-1., 1.])
        self.tim.add('map2alm spin %s lmaxout %s nrings %s'%(spin, lmax_out, self.geom.get_nrings()))
        if self.verbose:
            print(self.tim)
        return gclm_len