"""Simulation module including GF non-linear kappa maps


"""
import os
from os.path import join as opj

import healpy as hp
import numpy as np

from lenscarf.sims import generic


class sims_gaussian(generic.sims_cmb_len):
    """Simulations of Gaussian phi

        Args:
            lib_dir: the phases of the CMB maps and the lensed CMBs will be stored there
            lmax_cmb: cmb maps are generated down to this max multipole
            cls_unl: dictionary of unlensed CMB spectra
            dlmax, nside_lens, facres, nbands: lenspyx lensing module parameters
            wcurl: include field rotation map in the lensing deflection (default to False for historical reasons)


        This uses the cl_fid phi from sims_postborn to generate new lensing potential fields.

    """
    def __init__(self, lib_dir, lmax_cmb, cls_unl:dict, wcurl=False,
                 dlmax=1024, nside_lens=4096, facres=0, nbands=8, cache_plm=True):

        lmax_plm = lmax_cmb + dlmax
        mmax_plm = lmax_plm

        self.lmax_plm = lmax_plm
        self.mmax_plm = mmax_plm
        self.path = None

        self.cache_plm = cache_plm
        self.wcurl = wcurl
        
        cmb_cls = {}
        for k in cls_unl.keys():
                cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])

        super(sims_gaussian, self).__init__(lib_dir,  lmax_cmb, cmb_cls,
            dlmax=dlmax, nside_lens=nside_lens, facres=facres, nbands=nbands)


    def get_sim_plm(self, idx):
        fn = os.path.join(self.lib_dir, 'plm_in_lmax%s.fits'%self.lmax_plm)
        if not os.path.exists(fn):
            plm = self.unlcmbs.get_sim_plm(self.offset_index(idx, self.offset_plm[0], self.offset_plm[1]))
            if self.cache_plm:
                hp.write_alm(fn, plm)
            return plm
        return hp.read_alm(fn)