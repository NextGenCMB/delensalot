"""Simulation module


"""
import os
from lenspyx.utils_hp import synalm
import numpy as np
from numpy.random import default_rng
from plancklens.helpers import cachers
from lenspyx.wigners import wigners
from lenspyx.remapping import utils_geom
from scipy.interpolate import UnivariateSpline as spl
from lenspyx.utils_hp import almxfl, alm2cl

rng = default_rng() # random number generator
def k0(z): # convenience fit to empty beam convergence for 0.3 <= z <= 4, see https://arxiv.org/pdf/1105.3980.pdf Eq 38
    assert 0.3 <= z <= 4, z
    return 0.008 * z + 0.029 * z * z - 0.0079 * z ** 3 + 0.00065 * z ** 4

def calc_clln(cldd, lmax, npts=None, fill_mono=True):
    r"""From spectrum of delta field, returns spectrum of log 1 + delta for lognormal field statistics

        For lognormal fields hold

            :math:`\xi_{LN}(\hat n) = \ln (1 + \xi_{\delta})`

        Args:
            cldd: power spectrum of overdensity field
            lmax: spectrum of logdensity field is calculated down to this lmax
            npts(optional): number of points in GL quadrature
            fill_mono(optional): An exact lognormal field cant have cldd[0] = 0. This will interpolate from higher ell if set

        Returns:
            spectrum of log-field and variance of log-field

    """
    lmax_d  = len(cldd) - 1
    if npts is None:
        npts = min((lmax_d + lmax) // 2 + 1, 10000)
    if cldd[0] == 0 and fill_mono:
        print("(calc_cln: filling in cldd monopole)")
        assert np.all(cldd[1:10] > 0)
        cldd[0] = np.exp(spl(np.arange(1, 10, dtype=float), np.log(cldd[1:10]), k=3, s=0, ext=0)(0.))

    s2 =  wigners.wignerpos(cldd, np.array([0.]), 0, 0)
    print('calc_cln: root variance of the delta field is %.5f'%np.sqrt(s2))
    tht, wg = wigners.get_thgwg(npts)
    cl_lnd = wigners.wignercoeff(np.log(1. + wigners.wignerpos(cldd, tht, 0, 0)) * wg, tht, 0, 0, lmax)
    s2_lnd = np.sum(cl_lnd * (2 * np.arange(lmax + 1) + 1)) /(4 * np.pi)
    return cl_lnd, s2_lnd


class lneasy_sims:
    def __init__(self, cldd:np.ndarray, lmax_ln:int,
                 geom:utils_geom.Geom or None=None,
                 cacher_sims:cachers.cacher or None=None,
                 cacher_cls :cachers.cacher or None=None,
                 sht_threads:int or None=None, mmax:int or None=None,
                 nside:int=2048, kappa0=None):
        r"""Class to produce simple lognormal fields on the sphere

            Args:
                cldd (ndarray): overdensity 2d power spectrum
                geom(optional, lenscarf Geometry object): curved-sky pixelization (defaults fullsky healpix nside 2048)
                cacher_sims(optional): set this (eg to cachers.cacher_npy) to save the sims. By default does not cache.
                cacher_cls(optional): set this (eg to cachers.cacher_npy) to save the sims spectra. By default does not cache.
                sht_threads(optional): number of open_mp threads for sht transformds (defaults to OMP_NUM_THREADS or 8)
                mmax(optional): can cut mmax if close to the pole

        """
        if sht_threads is None:
            sht_threads = int(os.environ.get('OMP_NUM_THREADS', 0))
        if cacher_sims is None:
            cacher_sims = cachers.cacher_none()
        if cacher_cls is None:
            cacher_cls = cachers.cacher_none()
        if geom is None:
            geom = utils_geom.Geom.get_healpix_geometry(nside)
        if mmax is None:
            mmax = lmax_ln

        mmax = min(mmax, lmax_ln)
        self.nside = nside
        self.lmax_ln = lmax_ln
        self.mmax_ln = mmax

        self.cacher_sims = cacher_sims
        self.cacher_cls = cacher_cls

        # calculates spectrum of log-density field:
        cl_lnd, s2_lnd = calc_clln(cldd, lmax_ln)
        if np.any(cl_lnd < 0):
            ii = np.where(cl_lnd < 0.)[0]
            print("I see %s negative spectra values above multipole %s "%(len(ii), ii[0]))
            print("Setting all these to zero, but you might want to check whether this makes sense")

        self.cl_lnd = np.maximum(cl_lnd, 0.)
        self.cl_dd = self.calc_cldd(3 * nside - 1) # This recalculates the prediction for the density spectra
        self.mean_lnd = - 0.5 * s2_lnd  # For zero mean delta field must have <ln 1 + d> = - 1/2 < ln(1 + d)^2>
        self.s2_lnd = s2_lnd

        self.geom = geom
        self.tr = sht_threads

        self.kappa0 = kappa0

    def calc_cldd(self, lmax, npts=None):
        lmax_lnd = len(self.cl_lnd) - 1
        if npts is None: # We choose here the number of GL points scuh that to 2nd order in xi_A the integration is exact
            npts = min((lmax_lnd + lmax_lnd + lmax) // 2 + 1, 10000)
        tht, wg = wigners.get_thgwg(npts)
        cl_dd = wigners.wignercoeff( (np.exp(wigners.wignerpos(self.cl_lnd, tht, 0, 0)) - 1) * wg, tht, 0, 0, lmax)
        return cl_dd

    def get_skewness(self):
        """Theoretical skewness of one-point pdf


        """
        s2a = self.s2_lnd
        return (np.exp(s2a) + 2) * np.sqrt(np.exp(s2a) - 1)


    def get_sim(self, idx):
        """Produces a sim (or loads it if precomputed and cached)


        """
        fn = 'sim_%04d'%int(idx)
        if not self.cacher_sims.is_cached(fn):
            ln_d = synalm(self.cl_lnd, self.lmax_ln, self.mmax_ln)
            delta = np.exp(self.geom.synthesis(ln_d, 0, self.lmax_ln, self.mmax_ln, nthreads=self.tr) + self.mean_lnd) - 1
            self.cacher_sims.cache(fn, delta)
            return delta
        return self.cacher_sims.load(fn)

    def get_sim_plm(self, idx, lmax_plm=None):
        """Returns lensing potential according to kappa = delta = kappa/kappa0 = e^lnd - 1


        """
        assert self.kappa0 is not None
        if lmax_plm is None:
            lmax_plm = self.lmax_ln
        mmax_plm = lmax_plm
        d = self.get_sim(idx)
        dl = self.geom.adjoint_synthesis(d.copy(), 0, lmax_plm, mmax_plm, self.tr).squeeze()
        p2k = 0.5 * np.arange(lmax_plm + 1, dtype=float) * np.arange(1, lmax_plm + 2, dtype=float)
        ftl = np.zeros(lmax_plm + 1, dtype=float)
        ftl[1:] = 1./p2k[1:]
        almxfl(dl, ftl * self.kappa0, lmax_plm, True)
        ls = np.arange(lmax_plm + 1)
        sd = np.sqrt(np.sum(alm2cl(dl, dl, lmax_plm,  mmax_plm, lmax_plm)[ls] * ls * (ls + 1) * (2 * ls + 1.)) / (4 * np.pi))
        print('deflection rms %.3f amin'%(sd / np.pi * 180 * 60))
        return dl

    def get_sim_pt(self, idx):
        """Same sim but in leading order PT

           `math`: \delta(x) = e^A(x) - 1 \sim A(x) + \frac 12  A(x)^2


        """
        A= np.log(self.get_sim(idx) + 1.)
        return A + 0.5 * A * A


    def get_sim_cl(self, idx):
        """Gets (or load) full-sky power spectrum of the corresponding sim


         """
        fn = 'cl_%04d'%int(idx)
        if not self.cacher_cls.is_cached(fn):
            cl = self.geom.adjoint_synthesis(self.get_sim(idx).copy(), 0, self.lmax_ln, self.mmax_ln, nthreads=self.tr)
            self.cacher_cls.cache(fn, cl)
            return cl
        return self.cacher_cls.load(fn)