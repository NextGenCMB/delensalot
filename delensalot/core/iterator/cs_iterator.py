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
import shutil, time, sys
import numpy as np

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import lenspyx.remapping.utils_geom as utils_geom
from lenspyx.remapping.utils_geom import pbdGeometry, pbounds
from lenspyx.remapping.deflection_028 import rtype

from delensalot.utils import cli, read_map, enumerate_progress
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utility import utils_qe

from delensalot.core import cachers
from delensalot.core.cg import multigrid
from delensalot.core.opfilt import opfilt_base
from delensalot.core.iterator import bfgs, steps


@log_on_start(logging.DEBUG, " Start of prt_time()")
@log_on_end(logging.DEBUG, " Finished prt_time()")
def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    log.info("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label)
    return



class qlm_iterator(object):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict,
                 ninv_filt:opfilt_base.alm_filter_wl,
                 k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep,
                 logger=None,
                 NR_method=100, tidy=0, verbose=True, soltn_cond=True, wflm0=None, _usethisE=None, 
                 no_lensing_precond=False, no_lensing_dense=None):
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
                no_lensing_precond: if True, the preconditioner will not include lensing
                no_lensing_dense: if True, the dense part of the preconditioner will not include lensing, defaults to None to match no_lensing_precond
        """
        assert h in ['k', 'p', 'd']
        lmax_qlm, mmax_qlm = lm_max_dlm
        lmax_filt, mmax_filt = ninv_filt.lmax_sol, ninv_filt.mmax_sol

        if mmax_qlm is None: mmax_qlm = lmax_qlm

        self.h = h

        self.lib_dir = lib_dir
        self.cacher = cachers.cacher_npy(lib_dir)
        self.hess_cacher = cachers.cacher_npy(opj(self.lib_dir, 'hessian'))
        self.wf_cacher = cachers.cacher_npy(opj(self.lib_dir, 'wflms'))
        self.blt_cacher = cachers.cacher_npy(opj(self.lib_dir, 'BLT/'))
        if logger is None:
            from delensalot.core.iterator import loggers
            logger = loggers.logger_norms(opj(lib_dir, 'history_increment.txt'))
        self.logger = logger

        self.chain_descr = chain_descr
        self.opfilt = sys.modules[ninv_filt.__module__] # filter module containing the ch-relevant info
        self.stepper = stepper
        self.soltn_cond = soltn_cond

        self.dat_maps = np.array(dat_maps)

        self.chh = cpp_prior[:lmax_qlm+1] * self._p2h(lmax_qlm) ** 2
        self.hh_h0 = cli(pp_h0[:lmax_qlm + 1] * self._h2p(lmax_qlm) ** 2 + cli(self.chh))  #~ (1/Cpp + 1/N0)^-1
        self.hh_h0 *= (self.chh > 0)
        self.lmax_qlm = lmax_qlm
        self.mmax_qlm = mmax_qlm

        self.NR_method = NR_method
        self.tidy = tidy
        self.verbose = verbose

        self.cls_filt = cls_filt
        self.lmax_filt = lmax_filt
        self.mmax_filt = mmax_filt

        self.filter = ninv_filt
        self.k_geom = k_geom
        # Defining a trial newton step length :

        self.wflm0 = wflm0
        plm_fname = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}['p'], self.h, 0)
        if not self.cacher.is_cached(plm_fname):
            self.cacher.cache(plm_fname, almxfl(read_map(plm0), self._p2h(self.lmax_qlm), self.mmax_qlm, False))
        self.logger.startup(self)

        self._usethisE = _usethisE
        self.no_lensing_precond = no_lensing_precond
        self.no_lensing_dense = no_lensing_dense 


    def _p2h(self, lmax):
        if self.h == 'p':
            return np.ones(lmax + 1, dtype=float)
        elif self.h == 'k':
            return 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
        elif self.h == 'd':
            return np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2), dtype=float)
        else:
            assert 0, self.h + ' not implemented'

    def _h2p(self, lmax): return cli(self._p2h(lmax))

    def hlm2dlm(self, hlm, inplace):
        if self.h == 'd':
            return hlm if inplace else hlm.copy()
        if self.h == 'p':
            h2d = np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float))
        elif self.h == 'k':
            h2d = cli(0.5 * np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float)))
        else:
            assert 0, self.h + ' not implemented'
        if inplace:
            almxfl(hlm, h2d, self.mmax_qlm, True)
        else:
            return  almxfl(hlm, h2d, self.mmax_qlm, False)


    def dlm2hlm(self, dlm, inplace):
        if self.h == 'd':
            return dlm if inplace else dlm.copy()
        if self.h == 'p':
            d2h = cli(np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float)))
        elif self.h == 'k':
            d2h = 0.5 * np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float))
        else:
            assert 0, self.h + ' not implemented'
        if inplace:
            almxfl(dlm, d2h, self.mmax_qlm, True)
        else:
            return  almxfl(dlm, d2h, self.mmax_qlm, False)

    def _sk2plm(self, itr):
        sk_fname = lambda k: 'rlm_sn_%s_%s' % (k, 'p')
        rlm = self.cacher.load('phi_%slm_it000'%self.h)
        for i in range(itr):
            rlm += self.hess_cacher.load(sk_fname(i))
        return rlm

    def _yk2grad(self, itr):
        yk_fname = lambda k: 'rlm_yn_%s_%s' % (k, 'p')
        rlm = self.load_gradient(0, 'p')
        for i in range(itr):
            rlm += self.hess_cacher.load(yk_fname(i))
        return rlm

    def is_iter_done(self, itr, key):
        """Returns True if the iteration 'itr' has been performed already and False if not

        """
        if itr <= 0:
            return self.cacher.is_cached('%s_%slm_it000' % ({'p': 'phi', 'o': 'om'}[key], self.h))
        sk_fname = lambda k: 'rlm_sn_%s_%s' % (k, 'p')
        return self.hess_cacher.is_cached(sk_fname(itr - 1)) #FIXME

    def _is_qd_grad_done(self, itr, key):
        if itr <= 0:
            return self.cacher.is_cached('%slm_grad%slik_it%03d' % (self.h, key.lower(), 0))
        yk_fname = lambda k: 'rlm_yn_%s_%s' % (k, 'p')
        for i in range(itr):
            if not self.hess_cacher.is_cached(yk_fname(i)):
                return False
        return True


    @log_on_start(logging.DEBUG, "get_template_blm(it={it}) started")
    @log_on_end(logging.DEBUG, "get_template_blm(it={it}) finished")
    def get_template_blm(self, it, it_e, lmaxb=1024, lmin_plm=1, elm_wf:None or np.ndarray=None, dlm_mod=None, perturbative=False, k='p_p', pwithn1=False, plm=None):
        """Builds a template B-mode map with the iterated phi and input elm_wf

            Args:
                it: iteration index of lensing tracer
                it_e: iteration index of E-tracer (use it_e = it + 1 for matching lensing and E-templates)
                elm_wf: Wiener-filtered E-mode (healpy alm array), if not an iterated solution (it_e will ignored if set)
                lmin_plm: the lensing tracer is zeroed below lmin_plm
                lmaxb: the B-template is calculated up to lmaxb (defaults to lmax elm_wf)
                perturbative: use pertubative instead of full remapping if set (may be useful for QE)

            Returns:
                blm healpy array

            Note:
                It can be a real lot better to keep the same L range as the iterations

        """
        cache_cond = (lmin_plm >= 1) and (elm_wf is None)
        if cache_cond:
            fn_blt = 'blt_p%03d_e%03d_lmax%s'%(it, it_e, lmaxb)
            if dlm_mod is None:
                pass
            else:
                fn_blt += '_dlmmod' * dlm_mod.any()
            fn_blt += 'perturbative' * perturbative
            fn_blt += '_wN1' * pwithn1
        
        if cache_cond and self.blt_cacher.is_cached(fn_blt) :
            return self.blt_cacher.load(fn_blt)
        if elm_wf is None:
            assert k in ['p', 'p_p'], "Need to have computed the WF for polarization E in terations"
            if it_e > 0:
                e_fname = 'wflm_%s_it%s' % ('p', it_e - 1)
                assert self.wf_cacher.is_cached(e_fname)
                elm_wf = self.wf_cacher.load(e_fname)
            elif it_e == 0:
                elm_wf = self.wflm0()
            else:
                assert 0,'dont know what to do with it_e = ' + str(it_e)
        if len(elm_wf) == 2 and k == 'p':
            elm_wf = elm_wf[1]
        assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt, "{}, {}, {}, {}".format(elm_wf.size, self.mmax_filt, Alm.getlmax(elm_wf.size, self.mmax_filt), self.lmax_filt)
        mmaxb = lmaxb

        if plm is None:
            dlm = self.get_hlm(it, 'p', pwithn1)
        else:
            dlm = plm
        # subtract field from phi
        if dlm_mod is not None:
            dlm = dlm - dlm_mod

        self.hlm2dlm(dlm, inplace=True)
        almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
        if perturbative: # Applies perturbative remapping
            get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            geom, sht_tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
            d1_c = np.empty((geom.npix(),), dtype=elm_wf.dtype)
            d1_r = d1_c.view(rtype[d1_c.dtype]).reshape((d1_c.size, 2)).T  # real view onto complex array
            geom.synthesis(dlm, 1, self.lmax_qlm, self.mmax_qlm, sht_tr, map=d1_r, mode='GRAD_ONLY')
            dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dlens_c = -0.5 * ((d1_c.conj()) * dp + d1_c * dm)
            dlens_r = dlens_c.view(rtype[dlens_c.dtype]).reshape((dlens_c.size, 2)).T  # real view onto complex array
            del dp, dm, d1_c
            blm = geom.adjoint_synthesis(dlens_r, 2, lmaxb, mmaxb, sht_tr)[1]
        else: # Applies full remapping (this will re-calculate the angles)
            ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm)
            blm = ffi.lensgclm(elm_wf, self.mmax_filt, 2, lmaxb, mmaxb)[1]

        if cache_cond:
            self.blt_cacher.cache(fn_blt, blm)

        return blm


    def get_lik(self, itr, cache=False):
        """Returns the components of -2 ln p where ln p is the approximation to the posterior"""
        #FIXME: hack, this assumes this is the no-BB pol iterator 'iso' lik with no mf.  In general the needed map is the filter's file calc_prep output
        fn = 'lik_itr%04d'%itr
        if not self.cacher.is_cached(fn):
            e_fname = 'wflm_%s_it%s' % ('p', itr)
            assert self.wf_cacher.is_cached(e_fname), 'cant do lik, Wiener-filtered delensed CMB not available'
            elm_wf = self.wf_cacher.load(e_fname)
            self.filter.set_ffi(self._get_ffi(itr))
            elm = self.opfilt.calc_prep(read_map(self.dat_maps), self.cls_filt, self.filter, self.filter.ffi.sht_tr)
            l2p = 2 * np.arange(self.filter.lmax_sol + 1) + 1
            lik_qd = -np.sum(l2p * alm2cl(elm_wf, elm, self.filter.lmax_sol, self.filter.mmax_sol, self.filter.lmax_sol))
            # quadratic cst term : (X^d N^{-1} X^d)
            dat_copy = np.copy(read_map(self.dat_maps))
            self.filter.apply_map(dat_copy)
            # This only works for 'eb iso' type filters...
            l2p = 2 * np.arange(self.filter.lmax_len + 1) + 1
            lik_qdcst  = np.sum(l2p * alm2cl(dat_copy[0], self.dat_maps[0], self.filter.lmax_len, self.filter.mmax_len, self.filter.lmax_len))
            lik_qdcst += np.sum(l2p * alm2cl(dat_copy[1], self.dat_maps[1], self.filter.lmax_len, self.filter.mmax_len, self.filter.lmax_len))
            # Prior term
            hlm = self.get_hlm(itr, 'p')
            chh = alm2cl(hlm, hlm, self.lmax_qlm, self.mmax_qlm, self.lmax_qlm)
            l2p = 2 * np.arange(self.lmax_qlm + 1) + 1
            lik_pri = np.sum(l2p * chh * cli(self.chh))
            # det part
            lik_det = 0. # assumed constant here, should fix this for simple cases like constant MFs
            if cache:
                self.cacher.cache(fn, np.array([lik_qdcst, lik_qd, lik_det, lik_pri]))
            return  np.array([lik_qdcst, lik_qd, lik_det, lik_pri])
        return self.cacher.load(fn)

    def _get_ffi(self, itr):
        dlm = self.get_hlm(itr, 'p')
        self.hlm2dlm(dlm, inplace=True)
        ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
        return ffi

    def get_hlm(self, itr, key, pwithn1=False):
        """Loads current estimate """
        if itr < 0:
            return np.zeros(Alm.getsize(self.lmax_qlm, self.mmax_qlm), dtype=complex)
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if pwithn1:
            fn = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}[key.lower()], self.h, itr)
        else:
            fn = '%s_%slm_it%03d_wN1' % ({'p': 'phi', 'o': 'om'}[key.lower()], self.h, itr)
        if self.cacher.is_cached(fn):
            return self.cacher.load(fn)
        return self._sk2plm(itr)


    def load_soltn(self, itr, key):
        """Load starting point for the conjugate gradient inversion.

        """
        assert key.lower() in ['p', 'o']
        for i in np.arange(itr - 1, -1, -1):
            fname = 'wflm_%s_it%s' % (key.lower(), i)
            if self.wf_cacher.is_cached(fname):
                return self.wf_cacher.load(fname), i
        if callable(self.wflm0):
            return self.wflm0(), -1
        # TODO: for MV this need a change
        return np.zeros((1, Alm.getsize(self.lmax_filt, self.mmax_filt)), dtype=complex).squeeze(), -1


    def load_graddet(self, itr, key):
        fn= '%slm_grad%sdet_it%03d' % (self.h, key.lower(), itr)
        return self.cacher.load(fn)

    def load_gradpri(self, itr, key):
        """Compared to formalism of the papers, this returns -g_LM^{PR}"""
        assert key in ['p'], key + ' not implemented'
        assert self.is_iter_done(itr -1 , key)
        ret = self.get_hlm(itr, key)
        almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        return ret

    def load_gradquad(self, k, key):
        fn = '%slm_grad%slik_it%03d' % (self.h, key.lower(), k)
        return self.cacher.load(fn)

    def load_gradient(self, itr, key):
        """Loads the total gradient at iteration iter.

                All necessary alm's must have been calculated previously
                Compared to formalism of the papers, this returns -g_LM^{tot}
        """
        if itr == 0:
            g  = self.load_gradpri(0, key)
            g += self.load_graddet(0, key)
            g += self.load_gradquad(0, key)
            return g
        return self._yk2grad(itr)

    def calc_norm(self, qlm):
        return np.sqrt(np.sum(alm2cl(qlm, qlm, self.lmax_qlm, self.mmax_qlm, self.lmax_qlm)))


    @log_on_start(logging.DEBUG, "get_hessian(k={k}, key={key}) started")
    @log_on_end(logging.DEBUG, "get_hessian(k={k}, key={key}) finished")
    def get_hessian(self, k, key):
        """Inverse hessian that will produce phi_iter.


        """
        # Zeroth order inverse hessian :
        apply_H0k = lambda rlm, kr: almxfl(rlm, self.hh_h0, self.lmax_qlm, False)
        apply_B0k = lambda rlm, kr: almxfl(rlm, cli(self.hh_h0), self.lmax_qlm, False)
        lp1 = 2 * np.arange(self.lmax_qlm + 1) + 1
        dot_op = lambda rlm1, rlm2: np.sum(lp1 * alm2cl(rlm1, rlm2, self.lmax_qlm, self.mmax_qlm, self.lmax_qlm))
        BFGS_H = bfgs.BFGS_Hessian(self.hess_cacher, apply_H0k, {}, {}, dot_op,
                                   L=self.NR_method, verbose=self.verbose, apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for k_ in range(np.max([0, k - BFGS_H.L]), k):
            BFGS_H.add_ys('rlm_yn_%s_%s' % (k_, key), 'rlm_sn_%s_%s' % (k_, key), k_)
        return BFGS_H


    @log_on_start(logging.DEBUG, "build_incr(it={it}, key={key}) started")
    @log_on_end(logging.DEBUG, "build_incr(it={it}, key={key}) finished")
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
            yk = gradn - self.load_gradient(k, key)
            self.hess_cacher.cache(yk_fname, yk)
        k = it - 1
        BFGS = self.get_hessian(k, key)  # Constructing L-BFGS hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = 'rlm_sn_%s_%s' % (k, key)
        if not self.hess_cacher.is_cached(sk_fname):
            log.debug("calculating descent direction" )
            t0 = time.time()
            incr = BFGS.get_mHkgk(gradn, k)
            incr = self.stepper.build_incr(incr, it)
            self.hess_cacher.cache(sk_fname, incr)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert self.hess_cacher.is_cached(sk_fname), sk_fname


    @log_on_start(logging.DEBUG, "iterate(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "iterate(it={itr}, key={key}) finished")
    def iterate(self, itr, key):
        """Performs iteration number 'itr'

            This is done by collecting the gradients at level iter, and the lower level potential
            
            Compared to formalism of the papers, the total gradient is -g_{LM}^{Tot}. 
            These are the gradients of -ln posterior that we should minimize
        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self.is_iter_done(itr, key):
            assert self.is_iter_done(itr - 1, key), 'previous iteration not done'
            self.logger.on_iterstart(itr, key, self)
            # Calculation in // of lik and det term :
            glm  = self.calc_gradlik(itr, key)
            glm += self.calc_graddet(itr, key)
            glm += self.load_gradpri(itr - 1, key)
            almxfl(glm, self.chh > 0, self.mmax_qlm, True) # kills all modes where prior is set to zero
            self.build_incr(itr, key, glm)
            del glm
            self.logger.on_iterdone(itr, key, self)
            if self.tidy > 2:  # Erasing deflection databases
                if os.path.exists(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr))):
                    shutil.rmtree(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr)))


    @log_on_start(logging.DEBUG, "calc_gradlik(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_gradlik(it={itr}, key={key}) finished")
    def calc_gradlik(self, itr, key, iwantit=False):
        """Computes the quadratic part of the gradient for plm iteration 'itr'
        Compared to formalism of the papers, this returns -g_LM^{QD}
        Args:
            itr: iteration index
            key: 'p' or 'o'
            iwantit: if True, forces the calculation of the gradient and return it
        """
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self._is_qd_grad_done(itr, key) or iwantit:
            assert key in ['p'], key + '  not implemented'
            dlm = self.get_hlm(itr - 1, key)
            self.hlm2dlm(dlm, True)
            ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
            self.filter.set_ffi(ffi)
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter, no_lensing_precond=self.no_lensing_precond, no_lensing_dense=self.no_lensing_dense)
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

    @log_on_start(logging.DEBUG, "calc_graddet(it={itr}, key={key}) started, subclassed")
    @log_on_end(logging.DEBUG, "calc_graddet(it={itr}, key={key}) finished, subclassed")
    def calc_graddet(self, itr, key):
        """Compared to formalism of the papers, this should return +g_LM^{MF}"""
        assert 0, 'subclass this'

class iterator_splitlik_cstmf(qlm_iterator):
    """Split lensing iterator
    dat_maps is a tuple containing two sets of maps, one for each data split
    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:tuple, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_splitlik_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        
        assert len(self.dat_maps) == 2, "Data maps must be a tuple of two sets of maps"
        assert self.dat_maps[0].shape == self.dat_maps[1].shape, "Data maps must have the same shape"
        assert self.lmax_qlm == Alm.getlmax(plm0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(plm0.size, self.lmax_qlm))
        self.cacher.cache('mf', almxfl(plm0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))

    def calc_gradlik(self, itr, key, iwantit=False):
        # TODO: Implement here the double CG search, one for each data set, and then get the qlms, each with a different data and sum the two
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key
        if not self._is_qd_grad_done(itr, key) or iwantit:
            assert key in ['p'], key + '  not implemented'
            dlm = self.get_hlm(itr - 1, key)
            self.hlm2dlm(dlm, True)
            ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
            self.filter.set_ffi(ffi)
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter, no_lensing_precond=self.no_lensing_precond, no_lensing_dense=self.no_lensing_dense)
            # if self._usethisE is not None:
            #     if callable(self._usethisE):
            #         log.info("iterator: using custom WF E")
            #         soltn = self._usethisE(self.filter, itr)
            #     else:
            #         assert 0, 'dont know what to do this with this E input'
            _soltn = []
            for datset in [0, 1]:
                log.info("Grad like: Computing WF solution for datasplit %s at iter %s " % (datset, itr))
                soltn, it_soltn = self.load_soltn(itr, key +'_'+str(datset))
                if it_soltn < itr - 1:
                    soltn *= self.soltn_cond
                    
                    mchain.solve(soltn, self.dat_maps[0], dot_op=self.filter.dot_op())
                    fn_wf = 'wflm_%s_%s_it%s' % (key.lower(), datset, itr - 1)
                    log.info("caching "  + fn_wf)
                    self.wf_cacher.cache(fn_wf, soltn)
                else:
                    log.info("Using cached WF solution at iter %s "%itr)
                _soltn.append(soltn)

            t0 = time.time()
            if ffi.pbgeom.geom is self.k_geom and ffi.pbgeom.pbound == pbounds(0., 2 * np.pi):
                # This just avoids having to recalculate angles on a new geom etc
                q_geom = ffi.pbgeom
            else:
                q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
            G1, C1 = self.filter.get_qlms(self.dat_maps[0], _soltn[0], q_geom, alm_wf_leg2=_soltn[1])
            G2, C2 = self.filter.get_qlms(self.dat_maps[1], _soltn[1], q_geom, alm_wf_leg2=_soltn[0])
            G = (G1 + G2) / 2.
            C = (C1 + C2) / 2. 
            almxfl(G if key.lower() == 'p' else C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
            log.info('get_qlms calculation done; (%.0f secs)'%(time.time() - t0))
            if itr == 1: #We need the gradient at 0 and the yk's to be able to rebuild all gradients
                fn_lik = '%slm_grad%slik_it%03d' % (self.h, key.lower(), 0)
                self.cacher.cache(fn_lik, -G if key.lower() == 'p' else -C)
            return -G if key.lower() == 'p' else -C
 
    @log_on_start(logging.DEBUG, "load_graddet(it={k}, key={key}) started")
    @log_on_end(logging.DEBUG, "load_graddet(it={k}, key={key}) finished")
    def load_graddet(self, k, key):
        return self.cacher.load('mf')

    @log_on_start(logging.DEBUG, "calc_graddet(it={k}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={k}, key={key}) finished")
    def calc_graddet(self, k, key):
        return self.cacher.load('mf')
            
class iterator_cstmf(qlm_iterator):
    """Constant mean-field
    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        assert self.lmax_qlm == Alm.getlmax(mf0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(mf0.size, self.lmax_qlm))
        self.cacher.cache('mf', almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))


    @log_on_start(logging.DEBUG, "load_graddet(it={k}, key={key}) started")
    @log_on_end(logging.DEBUG, "load_graddet(it={k}, key={key}) finished")
    def load_graddet(self, k, key):
        return self.cacher.load('mf')

    @log_on_start(logging.DEBUG, "calc_graddet(it={k}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={k}, key={key}) finished")
    def calc_graddet(self, k, key):
        return self.cacher.load('mf')


class iterator_pertmf(qlm_iterator):
    """Mean field isotropic response applied to current estimate

            A constant term can also be added ('mf0')

    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf_resp:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, mf0=None, **kwargs):
        super(iterator_pertmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        assert mf_resp.ndim == 1 and mf_resp.size > self.lmax_qlm, mf_resp.shape
        if mf0 is not None: 
            assert self.lmax_qlm == Alm.getlmax(mf0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(mf0.size, self.lmax_qlm))
            self.cacher.cache('mf', almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))
        self.p_mf_resp = mf_resp

    @log_on_start(logging.DEBUG, "load_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "load_graddet(it={itr}, key={key}) finished")
    def load_graddet(self, itr, key):
        assert self.h == 'p', 'check this line is ok for other h'
        mf = almxfl(self.get_hlm(itr - 1, key), self.p_mf_resp * self._h2p(self.lmax_qlm), self.mmax_qlm, False)
        if self.cacher.is_cached('mf'):
            mf += self.cacher.load('mf')
        return mf

    @log_on_start(logging.DEBUG, "calc_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={itr}, key={key}) finished")
    def calc_graddet(self, itr, key):
        assert self.h == 'p', 'check this line is ok for other h'
        mf = almxfl(self.get_hlm(itr - 1, key), self.p_mf_resp * self._h2p(self.lmax_qlm), self.mmax_qlm, False)
        if self.cacher.is_cached('mf'):
            mf += self.cacher.load('mf')
        return mf


class iterator_predmf(qlm_iterator):
    """Mean field predicted from kappa MAP

    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, resp_mf:np.ndarray, cls_unl:dict, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_predmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)

        self.cls_unl = cls_unl
        self.resp_mf = resp_mf 
    
    @log_on_start(logging.DEBUG, "load_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "load_graddet(it={itr}, key={key}) finished")
    def load_graddet(self, itr, key):
        assert self.h == 'p', 'check this line is ok for other h'
        plm = self.get_hlm(itr - 1, key)
        return self.filter.get_qlms_mf_pred(plm, self.cls_unl, self.resp_mf)


    @log_on_start(logging.DEBUG, "calc_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={itr}, key={key}) finished")
    def calc_graddet(self, itr, key):
        assert self.h == 'p', 'check this line is ok for other h'
        plm = self.get_hlm(itr - 1, key)
        return self.filter.get_qlms_mf_pred(plm, self.cls_unl, self.resp_mf)


class iterator_simf_mcs(qlm_iterator):
    """Monte-Carlo evaluation of mean-field

    args: 
        mc_sims: array with the number of sims to use for MF estimate, each index of the array is the index of the iteration
        sub_nolensing: if True, subtract the MF estimated with the same phases but without the lensing, reducing the Monte Carlo variance
        mf_cmb_phas: phases of CMB sims for the MF estimate
        mf_noise_phas: phases of noise maps for the MF estimate
        shift_phas: if False, all MF are estimated with the same pahse at all iterations,
                    if Turem, changes the phases used for MF estimate between iterations 
    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf_key:int, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, mc_sims:np.ndarray=None, sub_nolensing:bool=False, 
                 mf_cmb_phas=None, mf_noise_phas=None, shift_phas=False, 
                 mf_lowpass_filter= lambda ell: np.ones(len(ell), dtype=float), **kwargs):
        super(iterator_simf_mcs, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        self.mf_key = mf_key
        self.sub_nolensing = sub_nolensing
        self.mc_sims = mc_sims
        self.mf_cmb_phas = mf_cmb_phas
        self.mf_noise_phas = mf_noise_phas 
        self.shift_phas = shift_phas
        self.mf_lowpass_filter = mf_lowpass_filter
        
        print(f'Iterator MF key: {self.mf_key}')

    @log_on_start(logging.DEBUG, "load_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "load_graddet(it={itr}, key={key}) finished")
    def load_graddet(self, itr, key, get_all_mcs=False, A_dlm=1., mc_sims = None, return_half_mean=False):
        if mc_sims is None:
            mc_sims = self.mc_sims
        
        assert self.is_iter_done(itr - 1, key)
        # assert itr > 0, itr
        if itr == 0:
            return 0

        assert key in ['p'], key + '  not implemented'
        try: 
            mc_sims[itr]
        except IndexError:
            print(f'No MC sims defined for MF estimate at iter {itr}')  
            return 0 
        
        if mc_sims[itr] == 0:
            print(f"No MF subtraction is performed for iter {itr}")
            return 0

        if self.shift_phas:
            sim0 = np.sum(mc_sims[:itr])
        else:
            sim0 = 0 
        mcs = np.arange(sim0, sim0 + mc_sims[itr])

        print(f"****Iterator: MF estimated for iter {itr} with sims {mcs} ****")
        
        # Gmf = self.get_grad_mf(itr, key, mcs, nolensing=False, A_dlm=A_dlm)

        if return_half_mean is True:
            Gmf1 = self.get_grad_mf(itr, key, mcs[::2], nolensing=False, A_dlm=A_dlm)
            Gmf2 = self.get_grad_mf(itr, key, mcs[1::2], nolensing=False, A_dlm=A_dlm)
            if self.sub_nolensing:
                Gmf1_nl = self.get_grad_mf(itr, key, mcs[::2], nolensing=True, A_dlm=A_dlm)
                Gmf1 -= Gmf1_nl
                Gmf2_nl = self.get_grad_mf(itr, key, mcs[1::2], nolensing=True, A_dlm=A_dlm)
                Gmf2 -= Gmf2_nl
            return Gmf1, Gmf2
        else:
            Gmf = self.get_grad_mf(itr, key, mcs, nolensing=False, A_dlm=A_dlm)

            if self.sub_nolensing:
                Gmf_nl = self.get_grad_mf(itr, key, mcs, nolensing=True, A_dlm=A_dlm)
                Gmf -= Gmf_nl
            return Gmf
        # return Gmfs if get_all_mcs else np.mean(Gmfs, axis=0)


    @log_on_start(logging.DEBUG, "calc_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={itr}, key={key}) finished")
    def calc_graddet(self, itr, key, get_all_mcs=False, A_dlm=1., mc_sims=None, return_half_mean=False):
        if mc_sims is None:
            mc_sims = self.mc_sims

        assert self.is_iter_done(itr - 1, key)
        # assert itr > 0, itr
        if itr == 0:
            return 0

        assert key in ['p'], key + '  not implemented'
        try: 
            mc_sims[itr]
        except IndexError:
            print(f'No MC sims defined for MF estimate at iter {itr}')  
            return 0 
        
        if mc_sims[itr] == 0:
            print(f"No MF subtraction is performed for iter {itr}")
            return 0

        if self.shift_phas:
            sim0 = np.sum(mc_sims[:itr])
        else:
            sim0 = 0 
        mcs = np.arange(sim0, sim0 + mc_sims[itr])

        print(f"****Iterator: MF estimated for iter {itr} with sims {mcs} ****")

        if return_half_mean is True:
            Gmf1 = self.get_grad_mf(itr, key, mcs[::2], nolensing=False, A_dlm=A_dlm)
            Gmf2 = self.get_grad_mf(itr, key, mcs[1::2], nolensing=False, A_dlm=A_dlm)
            if self.sub_nolensing:
                Gmf1_nl = self.get_grad_mf(itr, key, mcs[::2], nolensing=True, A_dlm=A_dlm)
                Gmf1 -= Gmf1_nl
                Gmf2_nl = self.get_grad_mf(itr, key, mcs[1::2], nolensing=True, A_dlm=A_dlm)
                Gmf2 -= Gmf2_nl
            return Gmf1, Gmf2
        else:
            Gmf = self.get_grad_mf(itr, key, mcs, nolensing=False, A_dlm=A_dlm)

            if self.sub_nolensing:
                Gmf_nl = self.get_grad_mf(itr, key, mcs, nolensing=True, A_dlm=A_dlm)
                Gmf -= Gmf_nl
            return Gmf


    def get_grad_mf(self, itr:int, key:str, mcs:np.ndarray, nolensing:bool=False, A_dlm=1., debug_prints=False):
        """Returns the QD gradients for a set of simulations, and cache the results
        
        Args:
            itr: iteration number
            key: field key 
            mcs: array with the simulation indices
            nolensing: estimate the gradient without lensing (useful to subtract sims with same phase but no lensing to reduce variance)
            A_dlm: change the lensing deflection amplitude (useful for tests)
        Returns: 
            G_mf: the mean of the gradients estimate for the set of simulations
        
        """
        mf_cacher = cachers.cacher_npy(opj(self.lib_dir, f'mf_sims_itr{itr:03d}'))
        if debug_prints:
            print(mf_cacher.lib_dir)
        fn_qlm = lambda this_idx : f'qlm_mf{self.mf_key}_sim_{this_idx:04d}' + '_nolensing' * nolensing + f'_Adlm{A_dlm}' * (A_dlm!=1.) 

        if nolensing:
            dlm = np.zeros(Alm.getsize(self.lmax_qlm, self.mmax_qlm), dtype='complex128')
        else:
            dlm = self.get_hlm(itr - 1, key) * A_dlm
            self.hlm2dlm(dlm, True)

        ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
        self.filter.set_ffi(ffi)
        mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter)
        # t0 = time.time()
        q_geom = self.filter.ffi.pbgeom

        _Gmfs = np.zeros(Alm.getsize(ffi.lmax_dlm, ffi.mmax_dlm), dtype='complex128')
        for i, idx in enumerate_progress(mcs, label='Getting MF sims' + ' no lensing' * nolensing):
            if debug_prints:
                print(fn_qlm(idx))
            if not mf_cacher.is_cached(fn_qlm(idx)):
                #!FIXME: the cmb_phas and noise_phas should have more than one field for Pol or MV estimators
                G, C = self.filter.get_qlms_mf(
                    self.mf_key, q_geom, mchain, 
                    phas=self.mf_cmb_phas.get_sim(idx, idf=0), 
                    noise_phas=self.mf_noise_phas.get_sim(idx, idf=0), cls_filt=self.cls_filt)
                
                mf_cacher.cache(fn_qlm(idx), G)
            _Gmfs += mf_cacher.load(fn_qlm(idx))
        
        almxfl(_Gmfs, self.mf_lowpass_filter(np.arange(ffi.lmax_dlm+1)), ffi.mmax_dlm, inplace=True)   
        return _Gmfs / len(mcs)


class iterator_simf(qlm_iterator):
    """Monte-Carlo evaluation of mean-field


    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf_key:int, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_simf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        self.mf_key = mf_key


    @log_on_start(logging.DEBUG, "calc_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_graddet(it={itr}, key={key}) finished")
    def calc_graddet(self, itr, key):
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key in ['p'], key + '  not implemented'
        dlm = self.get_hlm(itr - 1, key)
        self.hlm2dlm(dlm, True)
        ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
        self.filter.set_ffi(ffi)
        mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter)
        t0 = time.time()
        q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
        G, C = self.filter.get_qlms_mf(self.mf_key, q_geom, mchain, cls_filt=self.cls_filt)
        almxfl(G if key.lower() == 'p' else C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
        log.info('get_qlm_mf calculation done; (%.0f secs)' % (time.time() - t0))
        if itr == 1:  # We need the gradient at 0 and the yk's to be able to rebuild all gradients
            fn_lik = '%slm_grad%sdet_it%03d' % (self.h, key.lower(), 0)
            self.cacher.cache(fn_lik, -G if key.lower() == 'p' else -C)
        # !FIXME: This sign is probably wrong, as it should return +g^MF
        return -G if key.lower() == 'p' else -C

class iterator_cstmf_bfgs0(iterator_cstmf):
    """Variant of the iterator where the initial curvature guess is itself a bfgs update from phi =0 to input plm

        df0 is gradient estimate for phi == 0, (- qlms calcualted with unlensed weight and filtering)

    """
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray, df0:str,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_cstmf_bfgs0, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, mf0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        #assert self.lmax_qlm == Alm.getlmax(df0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(df0.size, self.lmax_qlm))
        self.df0 = df0
        s0_fname = 'rlm_sn_%s_%s' % (0, 'p')
        if not self.hess_cacher.is_cached(s0_fname):  # Caching Hessian BFGS yk update :
            self.hess_cacher.cache(s0_fname, plm0)
            log.info("Cached " + s0_fname)

    def get_hessian(self, k, key):
        """
        We need the inverse hessian that will produce phi_iter.
        """
        # Zeroth order inverse hessian :
        apply_H0k = lambda rlm, q: almxfl(rlm, self.hh_h0, self.mmax_qlm, False)
        apply_B0k = lambda rlm, q: almxfl(rlm, cli(self.hh_h0), self.mmax_qlm, False)
        lp1 = 2 * np.arange(self.lmax_qlm + 1) + 1
        dot_op = lambda rlm1, rlm2: np.sum(lp1 * alm2cl(rlm1, rlm2, self.lmax_qlm, self.mmax_qlm, self.lmax_qlm))
        BFGS_H = bfgs.BFGS_Hessian(self.hess_cacher, apply_H0k, {}, {}, dot_op,
                                   L=self.NR_method, verbose=self.verbose, apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for k_ in range(np.max([0, k - BFGS_H.L]), k + 1):
            BFGS_H.add_ys('rlm_yn_%s_%s.npy' % (k_, key), 'rlm_sn_%s_%s.npy' % (k_, key), k_)
        return BFGS_H


    @log_on_start(logging.DEBUG, "build_incr(it={it}, key={key}) started")
    @log_on_end(logging.DEBUG, "build_incr(it={it}, key={key}) finished")
    def build_incr(self, it, key, gradn):
        assert it > 0, it
        k = it - 1
        yk_fname = 'rlm_yn_%s_%s' % (k, key)
        if k >= 0 and not self.hess_cacher.is_cached(yk_fname):  # Caching hessian BFGS yk update :
            if k > 0:
                yk = gradn - self.load_gradient(k - 1, key)
                self.hess_cacher.cache(yk_fname, yk)
            else:
                self.hess_cacher.cache(yk_fname, gradn - self.df0)
                self.df0 = None
        k = it - 1
        BFGS = self.get_hessian(k, key)  # Constructing L-BFGS hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = 'rlm_sn_%s_%s' % (k + 1, key)
        if not self.hess_cacher.is_cached(sk_fname):
            log.debug("calculating descent direction")
            t0 = time.time()
            incr = BFGS.get_mHkgk(gradn, k + 1)
            incr = self.ensure_invertibility(self.get_hlm(it - 1, key), self.stepper.build_incr(incr, it), self.mmax_qlm)
            self.hess_cacher.cache(sk_fname, incr)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert self.hess_cacher.is_cached(sk_fname), sk_fname