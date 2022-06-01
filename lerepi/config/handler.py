#!/usr/bin/env python

"""handler.py: This module handles the configuration file and turns it into lerepi / dlensalot language
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj

import healpy as hp
import numpy as np

import plancklens
from plancklens import qest, qecl
from plancklens import utils, qest, qecl
from plancklens.filt import filt_util
from plancklens.qcinv import cd_solve
from plancklens.filt import filt_cinv, filt_util

from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt.bmodes_ninv import template_dense

"""
Change this and only differentiate between dlensalot and 'other' params
"""

class lensing_config():
    def __init__(self, config_file, TEMP):
        # TODO Here what we want is to transform human-input into lerepi-language
        self.nside = config_file.nside
        self.zbounds = config_file.zbounds
        self.zbounds_len = config_file.zbounds_len

        self.lmax_transf = config_file.lmax_transf
        self.lmax_filt = config_file.lmax_filt
        
        self.lmin_tlm = config_file.lmin_tlm
        self.lmin_elm = config_file.lmin_elm
        self.lmin_blm = config_file.lmin_blm
        self.lmax_qlm = config_file.lmax_qlm
        self.mmax_qlm = config_file.mmax_qlm
        
        self.lmax_ivf = config_file.lmax_ivf
        self.mmax_ivf = config_file.mmax_ivf
        self.lmin_ivf = config_file.lmin_ivf
        self.lmax_unl = config_file.lmax_unl
        self.mmax_unl = config_file.mmax_unl

        self.stepper = config_file.stepper

        cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
        self.cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
        self.cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))

        self.cpp = np.copy(self.cls_unl['pp'][:self.lmax_qlm + 1])
        self.cpp[:config_file.Lmin] *= 0.

        self.mc_sims_mf_it0 = np.arange(config_file.nsims_mf)
        self.lensres = config_file.LENSRES
        self.lenjob_pbgeometry = config_file.lenjob_pbgeometry
        self.ninvjob_geometry = config_file.ninvjob_geometry
        self.isOBD = config_file.isOBD
        self.tr = int(os.environ.get('OMP_NUM_THREADS', 8)) #TODO hardcoded.. what to do with it?

        self.iterator = config_file.ITERATOR

        if config_file.isOBD:
            self.tpl = template_dense(200, self.ninvjob_geometry, self.tr, _lib_dir=config_file.BMARG_LIBDIR)
        else:
            self.tpl = None

        if config_file.CHAIN_DESCRIPTOR == 'default':
            self.chain_descr = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, config_file.nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

        if config_file.STANDARD_TRANSFERFUNCTION == True:
            # Fiducial model of the transfer function
            self.transf_tlm   =  gauss_beam(config_file.BEAM/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_tlm)
            self.transf_elm   =  gauss_beam(config_file.BEAM/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_elm)
            self.transf_blm   =  gauss_beam(config_file.BEAM/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_blm)

            # Isotropic approximation to the filtering (used eg for response calculations)
            self.ftl =  cli(self.cls_len['tt'][:config_file.lmax_ivf + 1] + (config_file.nlev_t / 180 / 60 * np.pi) ** 2 * cli(self.transf_tlm ** 2)) * (self.transf_tlm > 0)
            self.fel =  cli(self.cls_len['ee'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_elm ** 2)) * (self.transf_elm > 0)
            self.fbl =  cli(self.cls_len['bb'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_blm ** 2)) * (self.transf_blm > 0)

            # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
            self.ftl_unl =  cli(self.cls_unl['tt'][:config_file.lmax_ivf + 1] + (config_file.nlev_t / 180 / 60 * np.pi) ** 2 * cli(self.transf_tlm ** 2)) * (self.transf_tlm > 0)
            self.fel_unl =  cli(self.cls_unl['ee'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_elm ** 2)) * (self.transf_elm > 0)
            self.fbl_unl =  cli(self.cls_unl['bb'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_blm ** 2)) * (self.transf_blm > 0)

        self.masks = config_file.masks
        if config_file.FILTER == 'cinv_sepTP':
            self.ninv_t = [np.array([hp.nside2pixarea(config_file.nside, degrees=True) * 60 ** 2 / config_file.nlev_t ** 2])] + config_file.masks
            self.ninv_p = [[np.array([hp.nside2pixarea(config_file.nside, degrees=True) * 60 ** 2 / config_file.nlev_p ** 2])] + config_file.masks]

            self.cinv_t = filt_cinv.cinv_t(opj(TEMP, 'cinv_t'), config_file.lmax_ivf,config_file.nside, self.cls_len, self.transf_tlm, self.ninv_t,
                            marge_monopole=True, marge_dipole=True, marge_maps=[])

            self.cinv_p = filt_cinv.cinv_p(opj(TEMP, 'cinv_p'), self.lmax_ivf, config_file.nside, self.cls_len, self.transf_elm, self.ninv_p,
                        chain_descr=self.chain_descr(config_file.lmax_ivf, config_file.CG_TOL), transf_blm=self.transf_blm, marge_qmaps=(), marge_umaps=())

            self.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(TEMP, 'ivfs'), config_file.sims, self.cinv_t, self.cinv_p, self.cls_len)
            self.ftl_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_tlm)
            self.fel_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_elm)
            self.fbl_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_blm)
            self.ivfs   = filt_util.library_ftl(self.ivfs_raw, config_file.lmax_ivf, self.ftl_rs, self.fel_rs, self.fbl_rs)

        if config_file.QE_LENSING_CL_ANALYSIS == True:
            self.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
            self.ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

            self.ivfs_d = filt_util.library_shuffle(self.ivfs, config_file.ds_dict)
            self.ivfs_s = filt_util.library_shuffle(self.ivfs, config_file.ss_dict)

            self.qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), config_file.ivfs, config_file.ivfs_d, self.cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)
            self.qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), config_file.ivfs, config_file.ivfs_s, self.cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)

            self.qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), self.qlms_ds, self.qlms_ds, np.array([]))  # for QE RDN0 calculations
            self.qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), self.qlms_ss, self.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations

        if config_file.FILTER_QE == 'sepTP':
            # ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
            self.mc_sims_bias = np.arange(60, dtype=int)
            self.mc_sims_var  = np.arange(60, 300, dtype=int)
            self.qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)
            self.qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), self.qlms_dd, self.qlms_dd, self.mc_sims_bias)


class survey_config():
    def __init__(self, config_file):
        # TODO Here what we want is to transform human-input into lerepi-language
        self.sims = config_file.sims
        self.THIS_CENTRALNLEV_UKAMIN = config_file.THIS_CENTRALNLEV_UKAMIN
        self.beam = config_file.BEAM

        self.DATA_libdir = config_file.DATA_LIBDIR
        self.BMARG_LIBDIR = config_file.BMARG_LIBDIR
        self.BMARG_LCUT = config_file.BMARG_LCUT

        self.fg = config_file.fg
        self.transf = config_file.transf


class run_config():
    def __init__(self, config_file):
        # TODO Here what we want is to transform human-input into lerepi-language
        self.k = config_file.K

        self.v = config_file.V      
        
        self.itmax = config_file.ITMAX
        self.imin = config_file.IMIN
        self.imax = config_file.IMAX
        
        self.tol = config_file.TOL
        self.tol_iter = lambda it : 10 ** (- self.tol)
        self.soltn_cond = config_file.soltn_cond # Uses (or not) previous E-mode solution as input to search for current iteration one
        self.cg_tol = config_file.CG_TOL

        self.mask_suffix = config_file.mask_suffix
        self.isOBD = config_file.isOBD
        
        if config_file.V == 'noMF':
            self.nsims_mf = 0
        else:
            self.nsims_mf = config_file.nsims_mf
        suffix = '08d_%s_r%s'%(config_file.fg,self.mask_suffix,)+'_isOBD'*self.isOBD
        if self.nsims_mf > 0:
            suffix += '_MF%s'%(self.nsims_mf)
        self.TEMP =  opj(os.environ['SCRATCH'], 'cmbs4', suffix)


        