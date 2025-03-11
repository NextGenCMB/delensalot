import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np
import os

from plancklens import qest, qresp

from delensalot.core.cg import cd_solve
from delensalot.core.ivf import filt_util, filt_cinv, filt_simple
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD

from delensalot.config.etc import logger
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import alm_copy
from delensalot.utils import cli

class PlancklensInterface:
    # def __init__(self, filter_desc):
    def __init__(self, data_container, lm_max_ivf, lm_max_qlm, lmin_teb, cg_tol, sht_threads, cls_len, cls_unl, estimator_type, libdir, chain_descr=None, zbounds=(-1,1), inv_operator_desc=None):
        # This class is to interface with Plancklens
        
        self.data_container = data_container
        self.estimator_type = estimator_type
        self.libdir = libdir or opj(os.environ['SCRATCH'], 'QE')

        # nivjob_geominfo, niv_desc, nlev, ttebl, filtering_spatial_type, 
        self.nivjob_geominfo = inv_operator_desc['geominfo']
        self.niv_desc = inv_operator_desc['niv_desc']
        self.nlev = inv_operator_desc['nlev']
        self.filtering_spatial_type = inv_operator_desc['filtering_spatial_type']
        self.transferfunction = inv_operator_desc['transferfunction']
        self.sky_coverage = inv_operator_desc['sky_coverage']
        
        # OBD='trunc', obd_libdir='obd', obd_rescale=1.,
        self.OBD = inv_operator_desc['OBD']
        self.obd_libdir = inv_operator_desc['obd_libdir']
        self.obd_rescale = inv_operator_desc['obd_rescale']

        self.cls_len = cls_len
        self.cls_unl = cls_unl

        self.lm_max_ivf = lm_max_ivf
        self.lm_max_qlm = lm_max_qlm
        self.lmin_teb = lmin_teb

        self.zbounds = zbounds
        self.cg_tol = cg_tol

        self.sht_threads = sht_threads
        self.chain_descr = chain_descr or (lambda p2, p5 : [[0, ["diag_cl"], p2, self.nivjob_geominfo[1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]])

        # Isotropic approximation to the filtering (using 'len' for lensed spectra)
        self.ftebl_len = {key: self.__compute_transfer(cls_key, nlev_key, transf_key, 'len') 
            for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}
        # Same using unlensed spectra (using 'unl' for unlensed spectra)
        self.ftebl_unl = {key: self.__compute_transfer(cls_key, nlev_key, transf_key, 'unl') 
            for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}


    @log_on_start(logging.INFO, 'filterqest', logger=log)
    def _init_filterqest(self):
        if self.sky_coverage == 'full' and self.filtering_spatial_type == 'isotropic':
            self.ivf = filt_simple.library_fullsky_sepTP(
                opj(self.libdir, 'ivf'),
                self.data_container,
                self.nivjob_geominfo[1]['nside'],
                self.transferfunction,
                self.cls_len,
                self.ftebl_len['t'],
                self.ftebl_len['e'],
                self.ftebl_len['b'],
                cache=True)
            if self.estimator_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(
                    opj(self.libdir, 'qlms_dd'),
                    self.ivf,
                    self.ivf,
                    self.cls_len['te'],
                    self.nivjob_geominfo[1]['nside'],
                    lmax_qlm=self.lm_max_qlm[0])
        elif self.sky_coverage == 'masked' or self.filtering_spatial_type == 'anisotropic':
            ## Wait for finished run(), as plancklens triggers cinv_calc...
            self.cinv_t = filt_cinv.cinv_t(
                lib_dir = opj(self.libdir, 'cinv_t'),
                lmax = self.lm_max_ivf[0],
                nside = self.nivjob_geominfo[1]['nside'],
                cl = self.cls_len,
                transf = self.transferfunction['t'],
                ninv = [self.niv_desc['t']],
                marge_monopole=True,
                marge_dipole=True,
                marge_maps=[],
            )

            transf_elm_loc = self.transferfunction['e']
            if self.OBD == 'OBD':
                log.log(logging.DEBUG, 'Using OBD')
                self.cinv_p = cinv_p_OBD.cinv_p(
                    lib_dir = opj(self.libdir, 'cinv_p'),
                    lmax = self.lm_max_ivf[0],
                    nside = self.nivjob_geominfo[1]['nside'],
                    cl = self.cls_len,
                    transf = transf_elm_loc[:self.lm_max_ivf[0]+1],
                    ninv = [self.niv_desc['e']],
                    geom = self.nivjob_geomlib,
                    chain_descr = self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                    bmarg_lmax = self.lmin_teb[2],
                    zbounds = (-1,1),
                    _bmarg_lib_dir = self.obd_libdir,
                    _bmarg_rescal = self.obd_rescale,
                    sht_threads = self.sht_threads)
            else:
                log.log(logging.INFO, 'Using trunc')
                self.cinv_p = filt_cinv.cinv_p(
                    lib_dir = opj(self.libdir, 'cinv_p'),
                    lmax = self.lm_max_ivf[0],
                    nside = self.nivjob_geominfo[1]['nside'],
                    cl = self.cls_len,
                    transf = self.transferfunction['e'],
                    ninv = [self.niv_desc['e']],
                    chain_descr = self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                    transf_blm = self.transferfunction['b'],
                    marge_qmaps = (),
                    marge_umaps = ()
                )
                log.log(logging.DEBUG, 'filt_cinv.cinv_p initialized')

            _filter_raw = filt_cinv.library_cinv_sepTP(
                lib_dir = opj(self.libdir, 'ivf'),
                sim_lib = self.data_container,
                cinvt = self.cinv_t,
                cinvp = self.cinv_p,
                cl_weights = self.cls_len,
            )
            log.log(logging.DEBUG, 'filt_cinv.library_cinv_sepTP initialized')
            _ftebl_rs = lambda x: np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[x])
            self.ivf = filt_util.library_ftl(
                ivfs = _filter_raw,
                lmax = self.lm_max_qlm[0],
                lfilt_t = _ftebl_rs(0),
                lfilt_e = _ftebl_rs(1),
                lfilt_b = _ftebl_rs(2),
            )
            log.log(logging.DEBUG, 'filt_util.library_ftl initialized')
            self.qlms_dd = qest.library_sepTP(
                lib_dir = opj(self.libdir, 'qlms_dd'),
                ivfs1 = self.ivf,
                ivfs2 = self.ivf,
                clte = self.cls_len['te'],
                nside = self.nivjob_geominfo[1]['nside'],
                lmax_qlm=self.lm_max_qlm[0]
            )
            log.log(logging.DEBUG, 'qest.library_sepTP initialized')
        return self.qlms_dd


    def get_wflm(self, idx, key, lm_max=None):
        lm_max = lm_max or self.lm_max_ivf
        if key in ['ptt']:
            return alm_copy(self.ivf.get_sim_tmliklm(idx), None, *lm_max)
        elif key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'x_p', 'x_eb', 'xeb', 'x_be', 'xee']:
            return alm_copy(self.ivf.get_sim_emliklm(idx), None, *lm_max)
        elif key in ['p']:
            return np.array([alm_copy(self.ivf.get_sim_tmliklm(idx), None, *lm_max), alm_copy(self.ivf.get_sim_emliklm(idx), None, *lm_max)])
        elif key in ['a_p']:
            return alm_copy(self.ivf.get_sim_emliklm(idx), None, *lm_max)
        else:
            raise ValueError('Unknown estimator_key:', key)


    def get_ivflm(self, idx, key):
        if key in ['ptt']:
            return alm_copy(self.ivf.get_sim_tlm(idx), None, *self.lm_max_ivf)
        elif key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'x_p', 'x_eb', 'xeb', 'x_be', 'xee']:
            return alm_copy(self.ivf.get_sim_elm(idx), None, *self.lm_max_ivf), alm_copy(self.ivf.get_sim_blm(idx), None, *self.lm_max_ivf)
        elif key in ['p']:
            return np.array([alm_copy(self.ivf.get_sim_tlm(idx), None, *self.lm_max_ivf), alm_copy(self.ivf.get_sim_elm(idx), None, *self.lm_max_ivf)])
        elif key in ['a_p']:
            return alm_copy(self.ivf.get_sim_elm(idx), None, *self.lm_max_ivf), alm_copy(self.ivf.get_sim_blm(idx), None, *self.lm_max_ivf)
        else:
            raise ValueError('Unknown estimator_key:', key)
        

    def get_response_unl(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_unl, self.cls_unl, self.ftebl_unl, lmax_qlm=lmax_qlm)
    

    def get_response_len(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=lmax_qlm)
    

    def __compute_transfer(self, cls_key, nlev_key, component, spectrum_type):
        cls = self.cls_len if spectrum_type == 'len' else self.cls_unl
        return cli(cls[cls_key][:self.lm_max_ivf[0] + 1] + df.a2r(self.nlev[nlev_key])**2 * cli(self.transferfunction[component] ** 2)) * (self.transferfunction[component] > 0)