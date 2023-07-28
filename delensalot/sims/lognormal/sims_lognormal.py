"""
Generates a log-normal simulation with a given power spectrum and skewness. It uses methods described in e.g. https://arxiv.org/abs/1602.08503

NOTE:
    An improvement could be done by setting some work at the unlcmb library level. Will leave this for now as an exploration.
    The reason is that in the future users might want to set up their own generation of sims. There should be some simple way to ovverride
    getting maps, maybe just some class inheritance somewhere.
"""

import healpy as hp
import numpy as np
import lognormal_utils as lu
from delensalot.sims import sims_gaussian
import os


class sims_gaussian(sims_gaussian.sims_gaussian):
    """Simulations with lognormal phi

        Args:
            lib_dir: the phases of the CMB maps and the lensed CMBs will be stored there
            lmax_cmb: cmb maps are generated down to this max multipole
            cls_unl: dictionary of unlensed CMB spectra
            dlmax, nside_lens, facres, nbands: lenspyx lensing module parameters
            wcurl: include field rotation map in the lensing deflection (default to False for historical reasons)


        This uses the cl_fid phi from sims_postborn to generate new lensing potential fields.

    """
    def __init__(self, lib_dir, lmax_cmb, cls_unl:dict, wcurl=False,
                 dlmax=1024, nside_lens=4096, epsilon=1e-5, cache_plm=True, mu: float = 0.0, var: float = 1.0, skew: float = 0.0, input_cl: np.ndarray = None, lmax_gen: int = 8000):

        lmax_plm = lmax_cmb + dlmax
        mmax_plm = lmax_plm

        self.lmax_plm = lmax_plm
        self.mmax_plm = mmax_plm
        self.path = None

        self.cache_plm = cache_plm
        self.wcurl = wcurl
        self.epsilon = epsilon
        
        cmb_cls = {}
        for k in cls_unl.keys():
                cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])

        super(sims_gaussian, self).__init__(lib_dir,  lmax_cmb, cmb_cls,
            dlmax=dlmax, nside_lens=nside_lens, epsilon=self.epsilon)
        
        if input_cl is None:
            self.input_cl = cmb_cls['pp']

        self.mu = mu
        self.lamb = lu.get_lambda_from_skew(skew, var, mu)
        self.lmax_gen = lmax_gen
        

    @staticmethod
    def kappa_lm_to_phi_lm(klm: np.ndarray) -> np.ndarray:
        """
        Converts the input kappa_lm to phi_lm.
        """
        lmax = hp.Alm.getlmax(klm.size)
        ls = np.arange(0, lmax)
        factor = np.nan_to_num(1/((ls * (ls + 1.)) / 2.))
        return hp.almxfl(klm, factor)
    
    def get_sim_plm(self, idx):
        """
        Get a simulated lensing potential map
        """
        fn = os.path.join(self.lib_dir, 'plm_in_%04d_lmax%s.fits'%(idx, self.lmax_plm))
        if not os.path.exists(fn):
            klm = lu.create_lognormal_single_map(inputcl = self.input_cl, nside = self.nnside_lensside, lmax_gen = self.lmax_gen, mu = self.mu, lamb = self.lamb)
            plm = self.kappa_lm_to_phi_lm(klm)
            if self.cache_plm:
                hp.write_alm(fn, plm)
            return plm
        return hp.read_alm(fn)