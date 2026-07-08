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
from delensalot.core import mpi

from delensalot.config.etc import logger
from delensalot.config.config_helper import data_functions as df
from delensalot.utility.utils_hp import alm_copy
from delensalot.utils import cli


# NOTE This class is to interface with Plancklens. TODO lenpsyx could get its own interface in here
class PlancklensInterface:
    def __init__(self, data_container, lm_max_ivf, lm_max_qlm, lmin_teb, cg_tol, sht_threads, cls_len, cls_unl, TP_strategy, libdir, chain_descr=None, zbounds=(-1,1), inv_operator_desc=None, sht_tr=None):
        self.data_container = data_container
        self.TP_strategy = TP_strategy
        self.libdir = libdir or opj(os.environ['SCRATCH'], 'QE')

        # nivjob_geominfo, niv_desc, nlev, ttebl, filtering_type, 
        self.nivjob_geominfo = inv_operator_desc['geominfo']
        self.niv_desc = inv_operator_desc['niv_desc']
        self.nlev = inv_operator_desc['nlev']
        self.filtering_type = inv_operator_desc['filtering_type']
        self.transferfunction = inv_operator_desc['transferfunction']
        
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


    def _qe_data_part(self, key):
        if "_" in key:
            return key.split("_", 1)[1]
        if key in ["p"]:
            return "tp"
        if key.endswith("tt"):
            return "tt"
        if key.endswith("eb"):
            return "eb"
        if key.endswith("be"):
            return "be"
        if key.endswith("ee"):
            return "ee"
        if key.endswith("p"):
            return "p"
        return key

    @log_on_start(logging.DEBUG, 'filterqest', logger=log)
    def _init_filterqest(self):
        if self.filtering_type == 'isotropic':
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
            if self.TP_strategy == 'separate':
                self.qlms_dd = qest.library_sepTP(
                    opj(self.libdir, 'qlms_dd'),
                    self.ivf,
                    self.ivf,
                    self.cls_len['te'],
                    self.nivjob_geominfo[1]['nside'],
                    lmax_qlm=self.lm_max_qlm[0])
            elif self.TP_strategy == 'joint':
                assert 0, 'Not implemented yet'
        elif self.filtering_type == 'anisotropic':
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
                chain_descr = self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
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
                if mpi.rank==0: log.log(logging.INFO, 'Using trunc')
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
                lmax_qlm = self.lm_max_qlm[0]
            ) 
            log.log(logging.DEBUG, 'qest.library_sepTP initialized')
        return self.qlms_dd


    def get_wflm(self, idx, key, lm_max=None):
        lm_max = lm_max or self.lm_max_ivf
        data_part = self._qe_data_part(key)

        if data_part == "tt":
            return alm_copy(self.ivf.get_sim_tmliklm(idx), None, *lm_max)

        elif data_part in ["p", "eb", "be", "ee"]:
            return alm_copy(self.ivf.get_sim_emliklm(idx), None, *lm_max)

        elif data_part == "tp":
            return np.array([
                alm_copy(self.ivf.get_sim_tmliklm(idx), None, *lm_max),
                alm_copy(self.ivf.get_sim_emliklm(idx), None, *lm_max),
            ])

        else:
            raise ValueError(f"Unknown estimator_key/data_part: {key} / {data_part}")


    def get_ivflm(self, idx, key):
        data_part = self._qe_data_part(key)

        if data_part == "tt":
            return alm_copy(self.ivf.get_sim_tlm(idx), None, *self.lm_max_ivf)

        elif data_part in ["p", "eb", "be", "ee"]:
            return (
                alm_copy(self.ivf.get_sim_elm(idx), None, *self.lm_max_ivf),
                alm_copy(self.ivf.get_sim_blm(idx), None, *self.lm_max_ivf),
            )

        elif data_part == "tp":
            return np.array([
                alm_copy(self.ivf.get_sim_tlm(idx), None, *self.lm_max_ivf),
                alm_copy(self.ivf.get_sim_elm(idx), None, *self.lm_max_ivf),
            ])

        else:
            raise ValueError(f"Unknown estimator_key/data_part: {key} / {data_part}")
        

    def get_response_unl(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_unl, self.cls_unl, self.ftebl_unl, lmax_qlm=lmax_qlm)
    

    def get_response_len(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=lmax_qlm)
    

    def __compute_transfer(self, cls_key, nlev_key, component, spectrum_type):
        cls = self.cls_len if spectrum_type == 'len' else self.cls_unl
        return cli(cls[cls_key][:self.lm_max_ivf[0] + 1] + df.a2r(self.nlev[nlev_key])**2 * cli(self.transferfunction[component] ** 2)) * (self.transferfunction[component] > 0)