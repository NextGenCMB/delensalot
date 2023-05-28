"""Module for curved-sky iterative lensing estimation

    Version with approximate, isotropic filtering.

    This is meant to be used under idealized condition only, and you probably want to fix the large-scale lenses to the QE solution


"""

import os
from os.path import join as opj
import shutil
import time
import sys
import numpy as np

import lenspyx.remapping.utils_geom as utils_geom

from plancklens.qcinv import multigrid

from delensalot.utils import cli, read_map
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.core import cachers
from delensalot.core.iterator import steps
import delensalot.core.iterator.cs_iterator
from delensalot.core.opfilt import opfilt_base


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

alm2rlm = lambda alm : alm # get rid of this
rlm2alm = lambda rlm : rlm


class iterator_cstmf(delensalot.core.iterator.cs_iterator.qlm_iterator):
    """Constant mean-field


    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)

        assert self.lmax_qlm == Alm.getlmax(mf0.size, self.mmax_qlm), (self.lmax_qlm, mf0.size, Alm.getlmax(mf0.size, self.lmax_qlm))

        self.cacher.cache('mf', almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))


    @log_on_start(logging.INFO, "fastWF.load_graddet() started: it={k}, key={key}")
    @log_on_end(logging.INFO, "fastWF.load_graddet() finished: it={k}, key={key}")
    def load_graddet(self, k, key):
        return self.cacher.load('mf')

    @log_on_start(logging.INFO, "fastWF.calc_graddet() started: it={k}, key={key}")
    @log_on_end(logging.INFO, "fastWF.calc_graddet() finished: it={k}, key={key}")
    def calc_graddet(self, k, key):
        return self.cacher.load('mf')

    @log_on_start(logging.INFO, "fastWF.calc_gradlik() started: it={itr}, key={key}")
    @log_on_end(logging.INFO, "fastWF.calc_gradlik() finished: it={itr}, key={key}")
    def calc_gradlik(self, itr, key, iwantit=False):
        """Computes the quadratic part of the gradient for plm iteration 'itr'

        """
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self._is_qd_grad_done(itr, key) or iwantit:
            log.info("before loading dlm")
            assert key in ['p'], key + '  not implemented'
            dlm = self.get_hlm(itr - 1, key)
            self.hlm2dlm(dlm, True)
            ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm, cachers.cacher_mem(safe=False))
            # Here we delens the data and set then the defl to zero in the filters
            # and we assume dat is EB
            mmax = None  # FIXME: here should be data actual mmax. We assume same as lmax
            #: FIXME total unsafe hack to see if this is pol or TT rec:
            PorT = 'p' in self.opfilt.__name__.split('.')[-1] or 'e' in self.opfilt.__name__.split('.')[-1]
            log.info("before lensing")
            if PorT: # Pol rec.
                delEB = np.empty_like(self.dat_maps)
                delEB[0] = almxfl(self.dat_maps[0], cli(self.filter.transf_elm), mmax, False)
                delEB[1] = almxfl(self.dat_maps[1], cli(self.filter.transf_blm), mmax, False)
                delEB = ffi.lensgclm(delEB, self.filter.mmax_len, 2, self.filter.lmax_len, self.filter.mmax_len, backwards=True, nomagn=True)
                almxfl(delEB[0], self.filter.transf_elm, mmax, True)
                almxfl(delEB[1], self.filter.transf_blm, mmax, True)
            else: # TT-rec
                delT = almxfl(self.dat_maps, cli(self.filter.transf), mmax, False)
                delT = ffi.lensgclm(delT, self.filter.mmax_len, 0, self.filter.lmax_len, self.filter.mmax_len, backwards=True, nomagn=True)
                almxfl(delT, self.filter.transf, mmax, True)
            self.filter.set_ffi(self.filter.ffi.change_dlm([np.zeros_like(dlm), None], self.mmax_qlm, cachers.cacher_mem(safe=False)))
            log.info("multigrid.multigrid_chain")
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter)
            soltn, it_soltn = self.load_soltn(itr, key)

            if it_soltn < itr - 1:
                soltn *= self.soltn_cond
                assert soltn.ndim == 1, 'Fix following lines'
                if PorT:
                    log.info('before mchain.solve()')
                    mchain.solve(soltn, delEB, dot_op=self.filter.dot_op())
                    log.info('after mchain.solve()')
                else:
                    mchain.solve(soltn, delT, dot_op=self.filter.dot_op())
                fn_wf = 'wflm_%s_it%s' % (key.lower(), itr - 1)
                log.info("caching "  + fn_wf)
                self.wf_cacher.cache(fn_wf, soltn)
            else:
                log.info("Using cached WF solution at iter %s "%itr)

            t0 = time.time()
            if ffi.pbgeom.geom is self.k_geom and ffi.pbgeom.pbound == utils_geom.pbounds(0., 2 * np.pi):
                # This just avoids having to recalculate angles on a new geom etc
                q_geom = ffi.pbgeom
            else:
                q_geom = utils_geom.Geometry(self.k_geom, utils_geom.pbounds(0., 2 * np.pi))
            self.filter.set_ffi(ffi)
            G, C = self.filter.get_qlms(self.dat_maps, soltn, q_geom)
            almxfl(G if key.lower() == 'p' else C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
            assert self.h == 'p'
            log.info('get_qlms calculation done; (%.0f secs)'%(time.time() - t0))
            if itr == 1: #We need the gradient at 0 and the yk's to be able to rebuild all gradients
                fn_lik = '%slm_grad%slik_it%03d' % (self.h, key.lower(), 0)
                self.cacher.cache(fn_lik, -G if key.lower() == 'p' else -C)
            return -G if key.lower() == 'p' else -C