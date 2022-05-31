import os
from os.path import join as opj

import healpy as hp
import numpy as np

import plancklens
from plancklens import qest, qecl
from plancklens.filt import filt_util
from plancklens import utils, qest, qecl
from plancklens.qcinv import cd_solve

from lenscarf.utils_hp import gauss_beam, almxfl, alm_copy
from lenscarf.utils import cli, read_map
from plancklens.filt import filt_cinv, filt_util


class lensing_config():
    def __init__(self, config_file, TEMP):
        self.lmax_transf = config_file.lmax_transf
        self.lmax_filt = config_file.lmax_filt
        
        self.lmin_tlm = config_file.lmin_tlm
        self.lmin_elm = config_file.lmin_elm
        self.lmin_blm = config_file.lmin_blm
        self.lmax_qlm = config_file.lmax_qlm
        self.mmax_qlm = config_file.mmax_qlm
        self.lmax_ivf = config_file.lmax_ivf
        self.lmin_ivf = config_file.lmin_ivf
        self.lmax_unl = config_file.lmax_unl
        self.mmax_unl = config_file.mmax_unl

        cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
        cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
        cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))

        if config_file.QE_LENSING_CL_ANALYSIS == True:
            self.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
            self.ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

            self.ivfs_d = filt_util.library_shuffle(ivfs, config_file.ds_dict)
            self.ivfs_s = filt_util.library_shuffle(ivfs, config_file.ss_dict)

            self.qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), config_file.ivfs, config_file.ivfs_d, cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)
            self.qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), config_file.ivfs, config_file.ivfs_s, cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)

            self.qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), self.qlms_ds, self.qlms_ds, np.array([]))  # for QE RDN0 calculations
            self.qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), self.qlms_ss, self.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations


        if config_file.STANDARD_TRANSFERFUNCTION == True:
            # Fiducial model of the transfer function
            self.transf_tlm   =  gauss_beam(config_file.beam/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_tlm)
            self.transf_elm   =  gauss_beam(config_file.beam/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_elm)
            self.transf_blm   =  gauss_beam(config_file.beam/180 / 60 * np.pi, lmax=config_file.lmax_ivf) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_blm)

            # Isotropic approximation to the filtering (used eg for response calculations)
            self.ftl =  cli(cls_len['tt'][:config_file.lmax_ivf + 1] + (config_file.nlev_t / 180 / 60 * np.pi) ** 2 * cli(self.transf_tlm ** 2)) * (self.transf_tlm > 0)
            self.fel =  cli(cls_len['ee'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_elm ** 2)) * (self.transf_elm > 0)
            self.fbl =  cli(cls_len['bb'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_blm ** 2)) * (self.transf_blm > 0)

            # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
            self.ftl_unl =  cli(cls_unl['tt'][:config_file.lmax_ivf + 1] + (config_file.nlev_t / 180 / 60 * np.pi) ** 2 * cli(self.transf_tlm ** 2)) * (self.transf_tlm > 0)
            self.fel_unl =  cli(cls_unl['ee'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_elm ** 2)) * (self.transf_elm > 0)
            self.fbl_unl =  cli(cls_unl['bb'][:config_file.lmax_ivf + 1] + (config_file.nlev_p / 180 / 60 * np.pi) ** 2 * cli(self.transf_blm ** 2)) * (self.transf_blm > 0)

        if config_file.CHAIN_DESCRIPTOR == 'default':
            self.chain_descr = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, config_file.nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]


        if config_file.FILTER == 'cinv_sepTP':
            self.ninv_t = [np.array([hp.nside2pixarea(config_file.nside, degrees=True) * 60 ** 2 / config_file.nlev_t ** 2])] + config_file.masks
            self.ninv_p = [[np.array([hp.nside2pixarea(config_file.nside, degrees=True) * 60 ** 2 / config_file.nlev_p ** 2])] + config_file.masks]

            self.cinv_t = filt_cinv.cinv_t(opj(TEMP, 'cinv_t'), config_file.lmax_ivf,config_file.nside, cls_len, self.transf_tlm, self.ninv_t,
                            marge_monopole=True, marge_dipole=True, marge_maps=[])

            self.cinv_p = filt_cinv.cinv_p(opj(TEMP, 'cinv_p'), self.lmax_ivf, config_file.nside, cls_len, self.transf_elm, self.ninv_p,
                        chain_descr=self.chain_descr(config_file.lmax_ivf, 1e-5), transf_blm=self.transf_blm, marge_qmaps=(), marge_umaps=())

            self.ivfs_raw    = filt_cinv.library_cinv_sepTP(opj(TEMP, 'ivfs'), config_file.sims, self.cinv_t, self.cinv_p, cls_len)
            self.ftl_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_tlm)
            self.fel_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_elm)
            self.fbl_rs = np.ones(config_file.lmax_ivf + 1, dtype=float) * (np.arange(config_file.lmax_ivf + 1) >= config_file.lmin_blm)
            self.ivfs   = filt_util.library_ftl(self.ivfs_raw, config_file.lmax_ivf, self.ftl_rs, self.fel_rs, self.fbl_rs)


        if config_file.FILTER_QE == 'sepTP':
            # ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
            self.mc_sims_bias = np.arange(60, dtype=int)
            self.mc_sims_var  = np.arange(60, 300, dtype=int)
            self.qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), self.ivfs, self.ivfs, cls_len['te'], config_file.nside, lmax_qlm=config_file.lmax_qlm)
            self.qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), self.qlms_dd, self.qlms_dd, self.mc_sims_bias)


class survey_config():
    def __init__(self, config_file):
        self.THIS_CENTRALNLEV_UKAMIN = config_file.THIS_CENTRALNLEV_UKAMIN
        self.beam_ILC = config_file.beam_ILC

        self.DATA_libdir = config_file.DATA_LIBDIR
        self.BMARG_LIBDIR = config_file.BMARG_LIBDIR
        self.BMARG_LCUT = config_file.BMARG_LCUT

        self.get_nlevp = config_file.get_nlevp
        self.fg = config_file.fg
        self.transf = config_file.transf


class run_config():
    def __init__(self, config_file):
        self.k = config_file.k

        self.v = config_file.v       
        
        self.itmax = config_file.itmax
        self.imin = config_file.imin
        self.imax = config_file.imax
        
        self.tol = config_file.tol
        self.tol_iter = lambda it : 10 ** (- self.tol)
        self.soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one
        self.cg_tol = config_file.cg_tol

        self.mask_suffix = config_file.mask_suffix
        self.isOBD = config_file.isOBD
        
        if config_file.v == 'noMF':
            self.nsims_mf = 0
        else:
            self.nsims_mf = config_file.nsims_mf
        suffix = '08d_%s_r%s'%(config_file.fg,self.mask_suffix,)+'_isOBD'*self.isOBD
        if self.nsims_mf > 0:
            suffix += '_MF%s'%(self.nsims_mf)
        self.TEMP =  opj(os.environ['SCRATCH'], 'cmbs4', suffix)


        