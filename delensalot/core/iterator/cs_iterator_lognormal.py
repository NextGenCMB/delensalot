"""Module for curved-sky iterative lensing estimation

    Version revised on March 2023

        Among the changes:
            * delensalot'ed this with great improvements in execution time
            * novel and more stable way of calculating the delfection angles and inverses
            * optionally change main variable from plm to klm or dlm with expected better behavior ?
            * rid of alm2rlm which was just wasting a little bit of time and loads of memory
            * abstracted bfgs with cacher and dot_op



    #FIXME: loading of total gradient seems mixed up with loading of quadratic gradient...
    #TODO: make plm0 possibly a path?
    #FIXME: Chh = 0 not resulting in 0 estimate
"""

import os
from os.path import join as opj
import shutil
import time
import sys
import numpy as np

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from plancklens.qcinv import multigrid

import lenspyx.remapping.utils_geom as utils_geom
from lenspyx.remapping.utils_geom import pbdGeometry, pbounds
from lenspyx.lensing import get_geom

from delensalot.utils import cli, read_map
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utility import utils_qe

from delensalot.core import cachers
from delensalot.core.opfilt import opfilt_base
from delensalot.core.iterator import bfgs, steps

from . import cs_iterator as csit

alm2rlm = lambda alm : alm # get rid of this
rlm2alm = lambda rlm : rlm


@log_on_start(logging.INFO, " Start of prt_time()")
@log_on_end(logging.INFO, " Finished prt_time()")
def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    log.info("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label)
    return

typs = ['T', 'QU', 'TQU']


class iterator_pertmf(csit.iterator_pertmf):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf_resp:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, mf0=None, kappa0=None, muG=None, clG=None, **kwargs):
        """Lensing map iterator

            The bfgs hessian updates are called 'hlm's and are either in plm, dlm or klm space

            Args:
                h: 'k', 'd', 'p' if bfgs updates act on klm's, dlm's or plm's respectively
                pp_h0: the starting hessian estimate. (cl array, ~ 1 / N0 of the lensing potential)
                cpp_prior: fiducial lensing potential spectrum used for the prior term
                cls_filt (dict): dictionary containing the filter cmb unlensed spectra (here, only 'ee' is required)
                k_geom: lenspyx geometry for once-per-iterations operations (like checking for invertibility etc, QE evals...)
                stepper: custom calculation of NR-step
                wflm0(optional): callable with Wiener-filtered CMB map search starting point

        """

        super(csit.iterator_pertmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        assert mf_resp.ndim == 1 and mf_resp.size > self.lmax_qlm, mf_resp.shape
        if mf0 is not None: 
            assert self.lmax_qlm == Alm.getlmax(mf0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(mf0.size, self.lmax_qlm))
            self.cacher.cache('mf', almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))
        self.p_mf_resp = mf_resp

        nside = 4096
        geominfo = ('healpix', {'nside': nside})
        #geominfo_defl = ('thingauss', {'lmax': 4200 + 300, 'smax': 2})
        self.q_geom = get_geom(geominfo)

        self.ffi = ninv_filt.ffi

        ells = np.arange(0, self.hh_h0.size, 1)
        factor = ells*(ells+1)/2
        self.hh_h0 *= factor**2. #transform in kappa space, and assume this is ok for the lognormal Gaussian field

        #kappa0 = 1.0272441232149767
        #ymu = 0.023049319395827456
        #cly = np.loadtxt(direc+"clgaussian.txt")

        """kappa0 = 0.8869370911600852
        ymu = -0.12403122348608665
        cly = np.loadtxt(direc+"clgaussian_5120.txt")
        """

        kappa0 = kappa0
        ymu = muG
        cly = clG

        cly[:1] *= 0.

        self.cly = cly
        self.kappa0 = kappa0
        self.ymu = ymu #mean of Gaussian filed

        """    #incr is in y map
        geominfo = ('healpix', {'nside': nside})
        ninvjob_geometry_new = get_geom(geominfo)
        kappa0 = 1.0272441232149767
        ymu = 0.023049319395827456
        ells = np.arange(0, self.lmax_qlm+1, 1)
        factor = ells*(ells+1)/2
        klm0 = almxfl(plm0, factor, mmax = self.mmax_qlm, inplace = False) #k0 estimate
        kmap = ninvjob_geometry_new.alm2map(klm0, self.lmax_qlm, self.mmax_qlm, 6, (-1., 1.)) #kmap estimate from harmonic to real space
        y = np.log(kmap+kappa0) #take log of shifted kappa map to obtain the y field
        ylm = ninvjob_geometry_new.map2alm(y, lself.max_qlm, self.mmax_qlm, 6, (-1., 1.))
        plm0 = ylm
        print("Done with ylm")
        """

    
    def get_hlm(self, itr, key):
        """Loads current estimate """
        print("Loading hlm, iteration", itr)
        if itr < 0:
            return np.zeros(Alm.getsize(self.lmax_qlm, self.mmax_qlm), dtype=complex)
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        fn = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}[key.lower()], self.h, itr)
        if self.cacher.is_cached(fn):
            return self.cacher.load(fn)
        return self._sk2plm(itr)

    def phi_to_kappa(self, phi_lm):
        lmax = Alm.getlmax(phi_lm.size, None)
        ells = np.arange(0, lmax+1, 1)
        factor = ells*(ells+1)/2
        return almxfl(phi_lm, factor, lmax, False)
    

    def kappa_to_phi(self, kappa_lm):
        ells = np.arange(0, self.lmax_qlm+1, 1)
        factor = ells*(ells+1)/2
        return almxfl(kappa_lm, cli(factor), self.mmax_qlm, False)
    
    def alm2map(self, alm):
        return self.q_geom.alm2map(alm.copy(), self.lmax_qlm, self.mmax_qlm, self.ffi.sht_tr, (-1., 1.))
    
    def map2alm(self, map):
        return self.q_geom.map2alm(map.copy(), self.lmax_qlm, self.mmax_qlm, self.ffi.sht_tr, (-1., 1.))
    
    def kappa_shifted(self, kappa):
        return kappa+self.kappa0
    
    def kappa_to_y_real(self, kappa):
        y = np.log(self.kappa_shifted(kappa))-self.ymu
        return y
    
    def y_to_kappa_real(self, y):
        kappa = np.exp(y)-self.kappa0
        return kappa
    
    def load_gradpri(self, itr, key):
        """
        Log-normal gradient with respect to p_lm
        """

        assert key in ['p'], key + ' not implemented'
        assert self.is_iter_done(itr -1 , key)
        ret = self.get_hlm(itr, key)
        ret = almxfl(ret, cli(self.cly), self.mmax_qlm, False)
        return ret
    

    def get_y(self, itr, key):

        assert key in ['p'], key + ' not implemented'
        assert self.is_iter_done(itr -1 , key)
        ret = self.get_hlm(itr, key)

        kappa_lm = self.phi_to_kappa(ret)
        kappa = self.alm2map(kappa_lm)

        return np.log(self.kappa_shifted(kappa))
    

    @log_on_start(logging.INFO, "calc_gradlik(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "calc_gradlik(it={itr}, key={key}) finished")
    def calc_gradlik(self, itr, key, iwantit=False):
        """Computes the quadratic part of the gradient for plm iteration 'itr'

        """
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self._is_qd_grad_done(itr, key) or iwantit:
            assert key in ['p'], key + '  not implemented'
            dlm = self.get_hlm(itr - 1, key)

            #now this is in reality the y field, so you have to get kappa
            yreal = self.alm2map(dlm)
            kappa = self.y_to_kappa_real(yreal)
            kappa_lm = self.map2alm(kappa)
            dlm = self.kappa_to_phi(kappa_lm)

            self.hlm2dlm(dlm, True)
            ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
            self.filter.set_ffi(ffi)
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter)
            if self._usethisE is not None:
                if callable(self._usethisE):
                    log.info("iterator: using custom WF E")
                    soltn = self._usethisE(self.filter, itr)
                else:
                    assert 0, 'dont know what to do this with this E input'
            else:
                soltn, it_soltn = self.load_soltn(itr, key)
                if it_soltn < itr - 1:
                    soltn *= self.soltn_cond
                    mchain.solve(soltn, self.dat_maps, dot_op=self.filter.dot_op())
                    fn_wf = 'wflm_%s_it%s' % (key.lower(), itr - 1)
                    log.info("caching "  + fn_wf)
                    self.wf_cacher.cache(fn_wf, soltn)
                else:
                    log.info("Using cached WF solution at iter %s "%itr)

            t0 = time.time()
            if ffi.pbgeom.geom is self.k_geom and ffi.pbgeom.pbound == pbounds(0., 2 * np.pi):
                # This just avoids having to recalculate angles on a new geom etc
                q_geom = ffi.pbgeom
            else:
                q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
            G, C = self.filter.get_qlms(self.dat_maps, soltn, q_geom)
            almxfl(G if key.lower() == 'p' else C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
            log.info('get_qlms calculation done; (%.0f secs)'%(time.time() - t0))
            if itr == 1: #We need the gradient at 0 and the yk's to be able to rebuild all gradients
                fn_lik = '%slm_grad%slik_it%03d' % (self.h, key.lower(), 0)
                self.cacher.cache(fn_lik, -G if key.lower() == 'p' else -C)
            return -G if key.lower() == 'p' else -C
    

    @log_on_start(logging.INFO, "iterate(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "iterate(it={itr}, key={key}) finished")
    def iterate(self, itr, key):
        """Performs iteration number 'itr'

            This is done by collecting the gradients at level iter, and the lower level potential

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self.is_iter_done(itr, key):
            assert self.is_iter_done(itr - 1, key), 'previous iteration not done'
            self.logger.on_iterstart(itr, key, self)
            # Calculation in // of lik and det term :
            glm_like  = self.calc_gradlik(itr, key)
            glm_det = self.calc_graddet(itr, key)

            glm = glm_det+glm_like

            glm_kappa = self.kappa_to_phi(glm) #gradient in kappa
            glm_kappa_real = self.alm2map(glm_kappa)

            ylm = self.get_hlm(itr-1, key)
            y = self.alm2map(ylm)
            glm_y_real = glm_kappa_real*np.exp(y) #gradient in y
            glm = self.map2alm(glm_y_real)

            glm_pri = self.load_gradpri(itr - 1, key) #prior in y
            glm += glm_pri
            almxfl(glm, self.cly > 0, self.mmax_qlm, True) 

            
            self.build_incr(itr, key, glm)
            del glm

            self.logger.on_iterdone(itr, key, self)
            if self.tidy > 2:  # Erasing deflection databases
                if os.path.exists(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr))):
                    shutil.rmtree(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr)))


    def load_gradquad(self, k, key):
        fn = '%slm_grad%slik_it%03d' % (self.h, key.lower(), k)
        result = self.cacher.load(fn)
        glm_kappa = self.kappa_to_phi(result) #gradient in kappa
        glm_kappa_real = self.alm2map(glm_kappa)
        ylm = self.get_hlm(k, key)
        y = self.alm2map(ylm)
        glm_y_real = glm_kappa_real*np.exp(y) #gradient in y

        return self.map2alm(glm_y_real)
    
    @log_on_start(logging.INFO, "load_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "load_graddet(it={itr}, key={key}) finished")
    def load_graddet(self, itr, key):
        assert self.h == 'p', 'check this line is ok for other h'
        mf = almxfl(self.get_hlm(itr - 1, key), self.p_mf_resp * self._h2p(self.lmax_qlm), self.mmax_qlm, False)
        if self.cacher.is_cached('mf'):
            mf += self.cacher.load('mf')
        return mf
    

    def get_kappa(self, ylm, kappa0):
        y = self.alm2map(ylm)
        return self.kappa_shifted(np.exp(y), -kappa0)

    def get_plm(self, ylm, kappa0):
        kappa = self.get_kappa(ylm, kappa0)
        return self.kappa_to_phi(self.map2alm(kappa))
        

    def load_gradient(self, itr, key):
        """Loads the total gradient at iteration iter.

                All necessary alm's must have been calculated previously

        """
        if itr == 0:
            g = self.load_gradpri(0, key)
            g += self.load_graddet(0, key)
            g += self.load_gradquad(0, key)
            return g
        return self._yk2grad(itr)


    @log_on_start(logging.INFO, "build_incr(it={it}, key={key}) started")
    @log_on_end(logging.INFO, "build_incr(it={it}, key={key}) finished")
    def build_incr(self, it, key, gradn):
        """Search direction

           BGFS method with 'self.NR method' BFGS updates to the hessian.
            Initial hessian are built from N0s.

            :param it: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
            :param key: 'p' or 'o'
            :param gradn: current estimate of the gradient (alm array)
            :return: increment for next iteration (alm array)

            s_k = x_k+1 - x_k = - H_k g_k
            y_k = g_k+1 - g_k
        """
        assert it > 0, it
        k = it - 2
        yk_fname = 'rlm_yn_%s_%s' % (k, key)
        if k >= 0 and not self.hess_cacher.is_cached(yk_fname):  # Caching hessian BFGS yk update :
            yk = alm2rlm(gradn - self.load_gradient(k, key))
            self.hess_cacher.cache(yk_fname, yk)
        k = it - 1
        BFGS = self.get_hessian(k, key)  # Constructing L-BFGS hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = 'rlm_sn_%s_%s' % (k, key)
        if not self.hess_cacher.is_cached(sk_fname):
            log.info("calculating descent direction" )
            t0 = time.time()
            incr = BFGS.get_mHkgk(alm2rlm(gradn), k)
            incr = alm2rlm(self.stepper.build_incr(incr, it))
            self.hess_cacher.cache(sk_fname, incr)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert self.hess_cacher.is_cached(sk_fname), sk_fname


