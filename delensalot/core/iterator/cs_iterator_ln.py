"""Module for curved-sky iterative lensing estimation

    Tests for recontruction using lognormal statistics for the deflection field


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
from lenspyx.remapping.deflection_028 import rtype

from delensalot.utils import cli, read_map
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utility import utils_qe

from delensalot.core import cachers
from delensalot.core.opfilt import opfilt_base
from delensalot.core.iterator import bfgs, steps, cs_iterator

alm2rlm = lambda alm : alm # get rid of this
rlm2alm = lambda rlm : rlm


typs = ['T', 'QU', 'TQU']


class iterator_cstmf(cs_iterator.qlm_iterator):
    """Constant mean-field
    """
    #def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
    #             dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
    #             cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
    #             chain_descr, stepper:steps.nrstep, **kwargs):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, kappa0, **kwargs):
        super(iterator_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        assert self.lmax_qlm == Alm.getlmax(mf0.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(mf0.size, self.lmax_qlm))
        self.cacher.cache('mf', almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))
        self.kappa_0 = kappa0

    def hlm2dlm(self, hlm, inplace):
        """In the lgnormal solver, the reconstructed field is the Gaussian log-field

            In the model A = ln(1 + kappa / kappa_0) we may write the gradient w.r.t. the Gaussian field as

            g^A_{L'M'} = kappa_0 \int dx e^A(x) g^\kappa(x) Y^*_{L'M'}(x)

            There is a non-zero contribution at L' = 0 here

            hlm is the Gaussian field A, with

        """
        assert self.h == 'p', 'other variants not implemented here'

        geom, tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
        k = np.exp(geom.synthesis(hlm, 0, self.lmax_qlm, self.mmax_qlm, nthreads=tr).squeeze()) - 1.
        kL = self.kappa_0 * geom.adjoint_synthesis(k, 0, self.lmax_qlm, self.mmax_qlm, nthreads=tr).squeeze()
        d2k = 0.5 * np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float))
        almxfl(kL, cli(d2k), self.mmax_qlm, inplace=True)
        ls = np.arange(self.lmax_qlm + 1)
        sd = np.sqrt(np.sum(alm2cl(kL, kL, self.lmax_qlm,  self.mmax_qlm, self.lmax_qlm)[ls] * (2 * ls + 1.)) / (4 * np.pi))
        print('deflection rms %.3f amin'%(sd / np.pi * 180 * 60))
        if inplace:
            hlm[:] = kL
        else:
            return kL

    def gradphi2grada(self, gp, Alm):
        """Turns gradient w.r.t. phi to gradient w.r.t. A = ln(1 + k/k0)

        """
        h2k = 0.5 * np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float)
        gk = almxfl(gp, cli(h2k), self.mmax_qlm, False)
        geom, tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
        expa = np.exp(geom.synthesis(Alm, 0, self.lmax_qlm, self.mmax_qlm, nthreads=tr))
        expa *= geom.synthesis(gk, 0, self.lmax_qlm, self.mmax_qlm, nthreads=tr)
        return geom.adjoint_synthesis(expa, 0, self.lmax_qlm, self.mmax_qlm, nthreads=tr).squeeze()

    @staticmethod
    def plm2Alm(plm, k0, lmax, mmax, geom:utils_geom.Geom, tr):
        """ A = ln(1 + k/k0)

                returns Alm from phi_lm

        """
        p2k = 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
        k = geom.synthesis(almxfl(plm, p2k, mmax, False), 0, lmax, mmax, nthreads=tr).squeeze()
        if (np.min(k) / k0 + 1.) < 0.:
            print('some k too negative, rescaling', np.min(k)/k0)
            k *= (-0.8 / np.min(k) * k0)
        Alm = geom.adjoint_synthesis(np.log(1. + k / k0), 0, lmax, mmax, nthreads=tr).squeeze()
        return Alm

    @staticmethod
    def Alm2plm(Alm, k0, lmax, mmax, geom:utils_geom.Geom, tr):
        """ A = ln(1 + k/k0)

                returns phi_lm from Alm

        """
        p2k = 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
        A = geom.synthesis(Alm, 0, lmax, mmax, nthreads=tr).squeeze()
        k = k0 * (np.exp(A) - 1.)
        k = geom.adjoint_synthesis(k, 0, lmax, mmax, nthreads=tr).squeeze()
        almxfl(k, cli(p2k), mmax, True)
        return k

    @log_on_start(logging.DEBUG, "calc_gradlik(it={itr}, key={key}) started")
    @log_on_end(logging.DEBUG, "calc_gradlik(it={itr}, key={key}) finished")
    def calc_gradlik(self, itr, key, iwantit=False):
        """Computes the quadratic part of the gradient for plm iteration 'itr'

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
                    log.info("caching " + fn_wf)
                    self.wf_cacher.cache(fn_wf, soltn)
                else:
                    log.info("Using cached WF solution at iter %s " % itr)

            t0 = time.time()
            if ffi.pbgeom.geom is self.k_geom and ffi.pbgeom.pbound == pbounds(0., 2 * np.pi):
                # This just avoids having to recalculate angles on a new geom etc
                q_geom = ffi.pbgeom
            else:
                q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
            G, C = self.filter.get_qlms(self.dat_maps, soltn, q_geom)
            almxfl(G if key.lower() == 'p' else C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
            # for ln case we want the gradient w.r.t. the logfield
            assert key.lower() == 'p' and self.h == 'p', 'fix this'
            G = self.gradphi2grada(G, self.get_hlm(itr - 1, key))
            log.info('get_qlms calculation done; (%.0f secs)' % (time.time() - t0))
            if itr == 1:  # We need the gradient at 0 and the yk's to be able to rebuild all gradients
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
