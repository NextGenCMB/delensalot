"""Module for curved-sky iterative lensing estimation


    In contrast to cs_iterator.py, this module attemps to reconstruct jointly several fields
    (for example the lensing gradient and curl potential jointly)


"""
from __future__ import annotations

import os
from os.path import join as opj
import shutil
import time
import sys
import numpy as np

from plancklens.qcinv import multigrid

from lenspyx.utils_hp import Alm, almxfl, alm2cl, alm_copy
from lenspyx.remapping.utils_geom import pbdGeometry, pbounds, Geom
from lenspyx import cachers

from delensalot.core.iterator import bfgs, loggers
from delensalot.core.opfilt import opfilt_base
from delensalot.utility import utils_qe
from delensalot.utils import cli


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from delensalot.utility.utils_steps import nrstep, gradient, gradient_dotop

@log_on_start(logging.INFO, " Start of prt_time()")
@log_on_end(logging.INFO, " Finished prt_time()")
def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    log.info("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label)
    return

typs = ['T', 'QU', 'TQU']


class logger_norms(loggers.logger_norms):
    def __init__(self, txt_file):
        super().__init__(txt_file)
        self.txt_file = txt_file
        self.ti = None


    def on_iterdone(self, itr:int, key:str, iterator:gclm_iterator):
        incr = iterator.hess_cacher.load('rlm_sn_%s_%s' % (itr-1, key))
        norm_inc = iterator.calc_norm(incr) / iterator.calc_norm(iterator.get_hlm(0))
        norms = [iterator.calc_norm(iterator.load_gradient(itr - 1))]
        norm_grad_0 = iterator.calc_norm(iterator.load_gradient(0))
        for i in [0]: norms[i] = norms[i] / norm_grad_0

        with open(opj(iterator.lib_dir, 'history_increment.txt'), 'a') as file:
            file.write('%03d %.1f %.6f %.6f \n'
                       % (itr, time.time() - self.ti, norm_inc, norms[0]))
            file.close()


class gclm_iterator(object):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:list[tuple],
                 dat_maps:list or np.ndarray, plm0s:list or np.ndarray, pp_h0s:list[np.ndarray],
                 cpp_priors:list[np.ndarray], labels:tuple[str], cls_filt:dict,
                 ninv_filt:opfilt_base.alm_filter_wl,
                 k_geom:Geom,
                 chain_descr, stepper:nrstep,
                 lm_maxee:tuple[int] or None=None,
                 logger=None,
                 NR_method=100, tidy=0, verbose=True, soltn_cond=True, wflm0=None, _usethisE=None):
        """Lensing map iterator

            The bfgs hessian updates are called 'hlm's and are either in plm, dlm or klm space

            Args:
                h: 'k', 'd', 'p' if bfgs updates act on klm's, dlm's or plm's respectively
                plm0s: starting point for each field to be reconstructed
                pp_h0s: the starting hessian estimate for each field. (cl array, ~ 1 / N0 of the lensing potential)
                cpp_priors: fiducial lensing potential spectrum used for the prior term for each component
                labels: components identification string (e.g. ('p', 'x') for joint lensing gradient and curl rec)
                cls_filt (dict): dictionary containing the filter cmb unlensed spectra (here, only 'ee' is required)
                k_geom: scarf geometry for once-per-iterations opertations (like cehcking for invertibility etc)
                stepper: custom calculation of NR-step
                wflm0(optional): callable with Wiener-filtered CMB map search starting point


        """
        assert h in ['k', 'p', 'd']
        lmax_filt, mmax_filt = ninv_filt.lmax_sol, ninv_filt.mmax_sol
        plm0s = plm0s if isinstance(plm0s, list) else [plm0s]
        pp_h0s = pp_h0s if isinstance(pp_h0s, list) else [pp_h0s]
        cpp_priors = cpp_priors if isinstance(cpp_priors, list) else [cpp_priors]

        assert len(lm_max_dlm) == len(plm0s)
        assert len(plm0s) == len(pp_h0s) and len(plm0s) == len(cpp_priors)
        assert len(labels) >= len(plm0s)


        for plm, (lmax_qlm, mmax_qlm) in zip(plm0s, lm_max_dlm):
            assert len(pp_h0s[0]) > lmax_qlm
            assert Alm.getlmax(plm0s[0].size, mmax_qlm) == lmax_qlm
            if mmax_qlm is None: mmax_qlm = lmax_qlm

        # lmax'es: here same for all, but easy to change
        self.lmaxs_qlm = [lmax_qlm for lmax_qlm, mmax_qlm in lm_max_dlm]
        self.mmaxs_qlm = [mmax_qlm for lmax_qlm, mmax_qlm in lm_max_dlm]

        self.h = h

        self.lib_dir = lib_dir
        self.cacher = cachers.cacher_npy(lib_dir)
        self.hess_cacher = cachers.cacher_npy(opj(self.lib_dir, 'hessian'))
        self.wf_cacher = cachers.cacher_npy(opj(self.lib_dir, 'wflms'))
        if logger is None:
            logger = logger_norms(opj(lib_dir, 'history_increment.txt'))
        self.logger = logger

        self.chain_descr = chain_descr
        self.opfilt = sys.modules[ninv_filt.__module__] # filter module containing the ch-relevant info
        self.stepper = stepper
        self.soltn_cond = soltn_cond

        self.dat_maps = dat_maps

        chhs = []
        hh_h0s = []
        for cpp_prior, pp_h0, lmax_qlm in zip(cpp_priors, pp_h0s, self.lmaxs_qlm):
            chh_p = cpp_prior[:lmax_qlm+1] * self._p2h(lmax_qlm) ** 2
            hh_h0_p = cli(pp_h0[:lmax_qlm + 1] * self._h2p(lmax_qlm) ** 2 + cli(chh_p))  #~ (1/Cpp + 1/N0)^-1
            hh_h0_p *= (cpp_prior> 0)
            chhs.append(chh_p)
            hh_h0s.append(hh_h0_p)

        self.chhs = chhs # (rescaled) isotropic approximation to the likelihood curvature for each component
        self.hh_h0s = hh_h0s

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
        gclm_fname = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}['p'], self.h, 0)
        if not self.cacher.is_cached(gclm_fname):
            glm0 = gradient(plm0s, self.mmaxs_qlm, labels=labels)
            self.cacher.cache(gclm_fname, glm0.almxfl([self._p2h(self.lmaxs_qlm[0]),self._p2h(self.lmaxs_qlm[0]) ], False).getarray())
        self.logger.startup(self)
        self.labels = labels[:len(plm0s)]
        self.gradlm_size = np.sum([Alm.getsize(lmax_q, mmax_q) for lmax_q, mmax_q in zip(self.lmaxs_qlm, self.mmaxs_qlm)])
        # Size of total gradient array

        self._usethisE = _usethisE

        self.lm_maxee = lm_maxee

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

    def hlm2dlm(self, hlm:gradient, inplace):
        """Rescaling of the gradient if desired """
        h2ds = []
        for lmax, mmax in zip(hlm.lmaxs, hlm.mmaxs): #Fix h values
            if self.h == 'd':
                h2d = np.ones(lmax + 1, dtype=float)
            elif self.h == 'p':
                h2d = np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float))
            elif self.h == 'k':
                h2d = cli(0.5 * np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)))
            else:
                assert 0, self.h + ' not implemented'
            h2ds.append(h2d)
        if inplace:
            hlm.almxfl(h2ds, True)
        else:
            return  hlm.almxfl(h2ds, False)


    def _sk2plm(self, itr):
        sk_fname = lambda k: 'rlm_sn_%s_%s' % (k, 'p')
        rlm = self.cacher.load('phi_%slm_it000'%self.h)
        for i in range(itr):
            rlm += self.hess_cacher.load(sk_fname(i))
        return rlm

    def _yk2grad(self, itr):
        yk_fname = lambda k: 'rlm_yn_%s_%s' % (k, 'p')
        rlm = self.load_gradient(0).getarray()
        for i in range(itr):
            rlm += self.hess_cacher.load(yk_fname(i))
        return rlm

    def is_iter_done(self, itr):
        """Returns True if the iteration 'itr' has been performed already and False if not

        """
        if itr <= 0:
            return self.cacher.is_cached('%s_%slm_it000' % ('phi', self.h))
        sk_fname = lambda k: 'rlm_sn_%s_%s' % (k, 'p')
        return self.hess_cacher.is_cached(sk_fname(itr - 1))

    def _is_qd_grad_done(self, itr, key):
        if itr <= 0:
            return self.cacher.is_cached('%slm_grad%slik_it%03d' % (self.h, key.lower(), 0))
        yk_fname = lambda k: 'rlm_yn_%s_%s' % (k, 'p')
        for i in range(itr):
            if not self.hess_cacher.is_cached(yk_fname(i)):
                return False
        return True


    @log_on_start(logging.INFO, "get_template_blm() started: it={it}, calc={calc}")
    @log_on_end(logging.INFO, "get_template_blm() finished: it={it}")
    def get_template_blm(self, it, it_e, lmaxb=1024, lmin_plm=1, elm_wf:None or np.ndarray=None, dlm_mod=None, calc=False, Nmf=None,
                         perturbative=False):
        """Builds a template B-mode map with the iterated phi and input elm_wf

            Args:
                it: iteration index of lensing tracer
                it_e: iteration index of E-tracer (for best results use it_e = it + 1)
                elm_wf: Wiener-filtered E-mode (healpy alm array), if not an iterated solution (it_e will ignored if set)
                lmin_plm: the lensing tracer is zeroed below lmin_plm
                lmaxb: the B-template is calculated up to lmaxb (defaults to lmax elm_wf)
                perturbative: use pertubative instead of full remapping if set (may be useful for QE)

            Returns:
                blm healpy array

            Note:
                It can be a real lot better to keep the same L range as the iterations

        """
        cache_cond = (lmin_plm == 1) and (elm_wf is None)
        # TODO this needs a cleaner implementation. Duplicate in map_delenser
        if dlm_mod is not None:
            dlm_mod_string = '_dlmmod'
        else:
            dlm_mod_string = ''
        if Nmf == None:
            pass
        else:
            dlm_mod_string += "{:03d}".format(Nmf)
        fn = 'btempl_p%03d_e%03d_lmax%s%s' % (it, it_e, lmaxb, dlm_mod_string)
        fn += 'perturbative' * perturbative
        if not calc:
            if self.wf_cacher.is_cached(fn):
                return self.wf_cacher.load(fn)
        if elm_wf is None:
            if it_e > 0:
                e_fname = 'wflm_%s_it%s' % ('p', it_e - 1)
                assert self.wf_cacher.is_cached(e_fname)
                elm_wf = self.wf_cacher.load(e_fname)
            elif it_e == 0:
                elm_wf = self.wflm0()
            else:
                assert 0,'dont know what to do with it_e = ' + str(it_e)
        assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt
        mmaxb = lmaxb
        dlm = self.get_hlm(it)

        # subtract field from phi
        if dlm_mod is not None:
            dlm -= dlm_mod
        self.hlm2dlm(dlm, inplace=True)
        assert self.lmaxs_qlm[0] == self.lmaxs_qlm[1]
        assert self.mmaxs_qlm[0] == self.mmaxs_qlm[1]
        dlm.almxfl([np.arange(self.lmaxs_qlm[0] + 1, dtype=int) >= lmin_plm] * 2, True)
        if perturbative: # Applies perturbative remapping
            assert dlm.labels in [('p', 'x'), ('p',)], 'not implemented'
            get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
            geom, sht_tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
            d1 = geom.alm2map_spin([dlm.get_comp('p'), dlm.get_comp('x')], 1, self.lmaxs_qlm[0], self.mmaxs_qlm[0], sht_tr, [-1., 1.])
            dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
            dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
            del dp, dm, d1
            elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
        else: # Applies full remapping
            assert dlm.labels in [('p', 'x'), ('p',)], 'not implemented'
            ffi = self.filter.ffi.change_dlm([dlm.get_comp('p'), dlm.get_comp('x')], self.mmaxs_qlm[0])
            elm, blm = ffi.lensgclm(elm_wf, self.mmax_filt, 2, lmaxb, mmaxb)
        if cache_cond:
            self.wf_cacher.cache(fn, blm)
        return blm

    def _get_ffi(self, itr):
        dlm = self.hlm2dlm(self.get_hlm(itr), False)
        if dlm.labels in [('p', 'x'), ('p',)]:
            glm, clm = np.copy(dlm.get_comp('p')),  np.copy(dlm.get_comp('x'))
            assert self.lmaxs_qlm[0] == self.lmaxs_qlm[1]
            assert self.mmaxs_qlm[0] == self.mmaxs_qlm[1]
            ffi = self.filter.ffi.change_dlm([glm, clm], self.mmaxs_qlm[0], cachers.cacher_mem())
            return ffi
        elif 'pee' in dlm.labels and 'p_eb' in dlm.labels: # EE and EB lensing components
            ffi_ee = self.filter.ffi.change_dlm([dlm.get_comp('pee'),  dlm.get_comp('xee')], self.mmaxs_qlm[0], cachers.cacher_mem())
            ffi_eb = self.filter.ffi.change_dlm([dlm.get_comp('p_eb'),  dlm.get_comp('x_eb')], self.mmaxs_qlm[1], cachers.cacher_mem())
            return [ffi_ee, ffi_eb]
        else:
            assert 0, ('dont know what to do with labels ', dlm.labels)

    def get_hlm(self, itr):
        """Loads current estimate of the anistropy sources. It is a complex array"""
        if itr < 0:
            return np.zeros(self.gradlm_size, dtype=complex)
        fn = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}['p'.lower()], self.h, itr)
        if self.cacher.is_cached(fn):
            return gradient.fromarray(self.cacher.load(fn), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)
        return gradient.fromarray(self._sk2plm(itr), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)


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


    def load_graddet(self, itr):
        fn= '%slm_grad%sdet_it%03d' % (self.h, 'p'.lower(), itr)
        return gradient.fromarray(self.cacher.load(fn), self.lmaxs_qlm, self.mmaxs_qlm)

    def load_gradpri(self, itr):
        assert self.is_iter_done(itr -1)
        ret = self.get_hlm(itr)
        ret.almxfl([cli(chh) for chh in self.chhs], True)
        return ret

    def load_gradquad(self, k):
        fn = '%slm_grad%slik_it%03d' % (self.h, 'p'.lower(), k)
        return gradient.fromarray(self.cacher.load(fn), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)

    def load_gradient(self, itr):
        """Loads the total gradient at iteration iter.

                All necessary alm's must have been calculated previously

        """
        if itr == 0:
            g  = self.load_gradpri(0)
            g += self.load_graddet(0)
            g += self.load_gradquad(0)
            return g
        return gradient.fromarray(self._yk2grad(itr), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)

    def dotop(self, glms1:np.ndarray or gradient, glms2:np.ndarray or gradient):
        if isinstance(glms1, gradient) and isinstance(glms2, gradient):
            return gradient_dotop(glms1, glms2)
        ret = 0.
        N = 0
        for lmax, mmax in zip(self.lmaxs_qlm, self.mmaxs_qlm):
            siz = Alm.getsize(lmax, mmax)
            cl = alm2cl(glms1[N:N+siz], glms2[N:N+siz], None, mmax, None)
            ret += np.sum(cl * (2 * np.arange(len(cl)) + 1))
            N += siz
        return ret

    def calc_norm(self, qlm:np.ndarray):
        return np.sqrt(self.dotop(qlm, qlm))


    def apply_H0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for lmax, mmax, h0 in zip(self.lmaxs_qlm, self.mmaxs_qlm, self.hh_h0s):
            siz = Alm.getsize(lmax, mmax)
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], h0, mmax, False)
            N += siz
        return ret

    def apply_B0k(self, grad_lm:np.ndarray, kr):
        ret = np.empty_like(grad_lm)
        N = 0
        for lmax, mmax, h0 in zip(self.lmaxs_qlm, self.mmaxs_qlm, self.hh_h0s):
            siz = Alm.getsize(lmax, mmax)
            ret[N:N+siz] = almxfl(grad_lm[N:N+siz], cli(h0), mmax, False) #TOD0 this assumes >= 0
            N += siz
        return ret

    @log_on_start(logging.INFO, "get_hessian() started: k={k}, key={key}")
    @log_on_end(logging.INFO, "get_hessian() finished: k={k}, key={key}")

    def get_hessian(self, k, key):
        """Inverse hessian that will produce phi_iter.


        """
        # Zeroth order inverse hessian :
        BFGS_H = bfgs.BFGS_Hessian(self.hess_cacher, self.apply_H0k, {}, {}, self.dotop,
                                   L=self.NR_method, verbose=self.verbose, apply_B0k=self.apply_B0k)
        # Adding the required y and s vectors :
        for k_ in range(np.max([0, k - BFGS_H.L]), k):
            BFGS_H.add_ys('rlm_yn_%s_%s' % (k_, key), 'rlm_sn_%s_%s' % (k_, key), k_)
        return BFGS_H


    @log_on_start(logging.INFO, "build_incr() started: it={it}, key={key}")
    @log_on_end(logging.INFO, "build_incr() finished: it={it}, key={key}")
    def build_incr(self, it, key, gradn:gradient):
        """Search direction

           BGFS method with 'self.NR method' BFGS updates to the hessian.
            Initial hessian are built from N0s.

            :param it: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
            :param key: 'p' or 'o'
            :param gradn: current estimate of the gradient
            :return: increment for next iteration (alm array)

            s_k = x_k+1 - x_k = - H_k g_k
            y_k = g_k+1 - g_k
        """
        assert it > 0, it
        k = it - 2
        yk_fname = 'rlm_yn_%s_%s' % (k, key)
        if k >= 0 and not self.hess_cacher.is_cached(yk_fname):  # Caching hessian BFGS yk update :
            yk = gradn - self.load_gradient(k)
            self.hess_cacher.cache(yk_fname, yk.getarray())
        k = it - 1
        BFGS = self.get_hessian(k, key)  # Constructing L-BFGS hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        sk_fname = 'rlm_sn_%s_%s' % (k, key)
        if not self.hess_cacher.is_cached(sk_fname):
            log.info("calculating descent direction" )
            t0 = time.time()
            incr = gradient.fromarray(BFGS.get_mHkgk(gradn.getarray(), k), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)
            # giving the invertibility check that up at least for the moment
            #incr = self.ensure_invertibility(self.get_hlm(it - 1), self.stepper.build_incr(incr, it))
            incr = self.stepper.build_incr(incr, it)
            self.hess_cacher.cache(sk_fname, incr.getarray())
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert self.hess_cacher.is_cached(sk_fname), sk_fname


    @log_on_start(logging.INFO, "iterate() started: it={itr}, key={key}")
    @log_on_end(logging.INFO, "iterate() finished: it={itr}, key={key}")
    def iterate(self, itr, key):
        """Performs iteration number 'itr'

            This is done by collecting the gradients at level iter, and the lower level potential

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self.is_iter_done(itr):
            assert self.is_iter_done(itr - 1), 'previous iteration not done'
            self.logger.on_iterstart(itr, key, self)
            # Calculation in // of lik and det term :
            glm  = self.calc_gradlik(itr, key)
            glm += self.calc_graddet(itr)
            glm += self.load_gradpri(itr - 1)
            glm.almxfl([chh > 0 for chh in self.chhs], True) # kills all modes where priors are set to zero
            self.build_incr(itr, key, glm)
            del glm
            self.logger.on_iterdone(itr, key, self)
            if self.tidy > 2:  # Erasing deflection databases
                if os.path.exists(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr))):
                    shutil.rmtree(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr)))


    @log_on_start(logging.INFO, "calc_gradlik() started: it={itr}, key={key}")
    @log_on_end(logging.INFO, "calc_gradlik() finished: it={itr}, key={key}")
    def calc_gradlik(self, itr, key, iwantit=False):
        """Computes the quadratic part of the gradient for plm iteration 'itr'

        """
        assert self.is_iter_done(itr - 1)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self._is_qd_grad_done(itr, key) or iwantit:
            assert key in ['p'], key + '  not implemented'
            self.filter.set_ffi(self._get_ffi(itr - 1))
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
                    assert soltn.ndim == 1, 'Fix following lines'
                    mchain.solve(soltn, self.dat_maps, dot_op=self.filter.dot_op())
                    fn_wf = 'wflm_%s_it%s' % (key.lower(), itr - 1)
                    log.info("caching "  + fn_wf)
                    self.wf_cacher.cache(fn_wf, soltn)
                else:
                    log.info("Using cached WF solution at iter %s "%itr)

            t0 = time.time()
            q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
            Gs, Cs = self.filter.get_qlms(self.dat_maps, soltn, q_geom)

            # GC can either G, C for a simple component gradient, or (G1, G2), (C1, C2) etc for a multicomponent
            # TODO match G and Cs to label components in general case
            if self.labels in [('p', 'x'), ('p',), ('x',), ('x', 'p')]:
                grad_lm = gradient( [-Gs* ('p' in self.labels), -Cs * ('x' in self.labels)],  self.mmaxs_qlm, labels=self.labels)
                grad_lm.almxfl([self._h2p(self.lmaxs_qlm[0]), self._h2p(self.lmaxs_qlm[1])], True)
            elif self.labels == ('pee', 'p_eb'):
                if self.lm_maxee is not None and self.lm_maxee[0] < self.lmaxs_qlm[0]:
                    print("seeing smaller lmax for ee, patching")
                    # ee only up to some lmax, then total gradient
                    felp = np.ones(self.lmaxs_qlm[0] + 1) * (np.arange(self.lmaxs_qlm[0] + 1) > self.lm_maxee[0])
                    felm = np.ones(self.lmaxs_qlm[0] + 1) * (np.arange(self.lmaxs_qlm[0] + 1) <= self.lm_maxee[0])
                    Gee = almxfl(Gs[0], felm, self.mmaxs_qlm[0], False)
                    G_p = Gs[1] + alm_copy(almxfl(Gs[0], felp, self.mmaxs_qlm[0], False), self.mmaxs_qlm[0], self.lmaxs_qlm[1], self.mmaxs_qlm[1])
                    G_ee =Gee +  almxfl(alm_copy(G_p, self.mmaxs_qlm[1], self.lmaxs_qlm[0], self.mmaxs_qlm[0]), felp, self.mmaxs_qlm[0], False)
                    grad_lm = gradient([-G_ee, -G_p],  self.mmaxs_qlm, labels=self.labels)

                else:
                    grad_lm = gradient([-Gs[0], -Gs[1]],  self.mmaxs_qlm, labels=self.labels)
                del Cs
            else:
                assert 0, ('dont know what to do with ',self.labels)
            log.info('get_qlms calculation done; (%.0f secs)'%(time.time() - t0))
            if itr == 1: #We need the gradient at 0 and the yk's to be able to rebuild all gradients
                fn_lik = '%slm_grad%slik_it%03d' % (self.h, key.lower(), 0)
                self.cacher.cache(fn_lik, grad_lm.getarray())
            return grad_lm

    @log_on_start(logging.INFO, "calc_graddet() started: it={itr}. subclassed")
    @log_on_end(logging.INFO, "calc_graddet() finished: it={itr}. subclassed")
    def calc_graddet(self, itr):
        assert 0, 'subclass this'

       
class iterator_cstmf(gclm_iterator):
    """Constant mean-field


    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:list[tuple],
                 dat_maps:list or np.ndarray, plm0s:list or np.ndarray, mf0s:list or np.ndarray, pp_h0s:list[np.ndarray],
                 cpp_priors:list[np.ndarray], labels:tuple[str], cls_filt:dict,
                 ninv_filt:opfilt_base.alm_filter_wl,
                 k_geom:Geom,
                 chain_descr, stepper:nrstep,**kwargs):
        super(iterator_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0s, pp_h0s, cpp_priors, labels, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)

        if not self.cacher.is_cached('mf'):
            if not isinstance(mf0s, list):
                mf0s = [mf0s]
            mf0 = gradient(mf0s, self.mmaxs_qlm)
            self.cacher.cache('mf', mf0.almxfl([self._h2p(self.lmaxs_qlm[0]), self._h2p(self.lmaxs_qlm[1])],False).getarray())


    @log_on_start(logging.INFO, "load_graddet() started: it={k}")
    @log_on_end(logging.INFO, "load_graddet() finished: it={k}")
    def load_graddet(self, k):
        return gradient.fromarray(self.cacher.load('mf'), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)

    @log_on_start(logging.INFO, "calc_graddet() started: it={k}")
    @log_on_end(logging.INFO, "calc_graddet() finished: it={k}")
    def calc_graddet(self, k):
        return gradient.fromarray(self.cacher.load('mf'), self.lmaxs_qlm, self.mmaxs_qlm, labels=self.labels)


# TODO add visitor pattern if desired