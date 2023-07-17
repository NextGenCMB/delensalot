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


class qlm_iterator(csit.qlm_iterator):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict,
                 ninv_filt:opfilt_base.alm_filter_wl,
                 k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep,
                 logger=None,
                 NR_method=100, tidy=0, verbose=True, soltn_cond=True, wflm0=None, _usethisE=None, kappa0: float = 1., ymu: float = 0., cly: np.ndarray = None):
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

        super().__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt, ninv_filt, k_geom,
                            chain_descr, stepper, logger, NR_method, tidy, verbose, soltn_cond, wflm0, _usethisE)
        
        self.kappa0 = kappa0
        self.cly = cly


    def phi_to_kappa(self, phi_lm):
        ells = np.arange(0, self.lmax_qlm+1, 1)
        factor = ells*(ells+1)/2
        return almxfl(phi_lm, factor, self.mmax_qlm, False)
    
    def alm2map(self, alm):
        return self.filter.geom.alm2map(alm, self.lmax_qlm, self.mmax_qlm, self.ffi.sht_tr, (-1., 1.))
    
    def map2alm(self, map):
        return self.filter.geom.map2alm(map, self.lmax_qlm, self.mmax_qlm, self.ffi.sht_tr, (-1., 1.))
    

    def kappa_shifted(self, kappa):
        return kappa+self.kappa0
    
    def kappa_to_y_real(self, kappa):
        y = np.log(self.kappa_shifted(kappa))-self.ymu
        return y

    def load_gradpri(self, itr, key):
        assert key in ['p'], key + ' not implemented'
        assert self.is_iter_done(itr -1 , key)
        ret = self.get_hlm(itr, key)
        
        #get the convergence map from the estimated Wiener filtered potential map
        kappa_lm = self.phi_to_kappa(ret)
        kappa = self.alm2map(kappa_lm)

        y = self.kappa_to_y_real(kappa)
        y_lm = self.map2alm(y)
        y_filtered_lm = almxfl(y_lm, cli(self.cly), self.mmax_qlm, False)
        y_filtered = self.alm2map(y_filtered_lm)

        kappa_shifted = self.kappa_shifted(kappa)

        ret = -(1/kappa_shifted*y_filtered+1/np.abs(kappa_shifted))
        return ret

    


