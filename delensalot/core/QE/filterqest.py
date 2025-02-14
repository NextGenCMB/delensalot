from os.path import join as opj
import numpy as np

from plancklens import qest, qresp

from lenspyx.lensing import get_geom 

from delensalot.utility.utils_hp import gauss_beam
from delensalot.utility.utils_hp import alm_copy
from delensalot.core.ivf import filt_util, filt_cinv, filt_simple
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD

from delensalot.config.config_helper import data_functions as df
from delensalot.utils import cli

class base:
    def __init__(self, filter_desc):
        # This class tries to hide the uglyness to interfacing with plancklens (no offense)
        self.lm_max_ivf = filter_desc['lm_max_sky']
        self.lm_max_qlm = filter_desc['LM_max']
        self.lmin_teb = filter_desc['lmin_teb']

        self.cls_len = filter_desc['cls_len']
        self.cls_unl = filter_desc['cls_unl']

        self.nivjob_geominfo = filter_desc['nivjob_geominfo']
        self.niv_desc = filter_desc['niv_desc']
        self.nlev = filter_desc['nlev']
        self.ttebl = filter_desc['ttebl']
        # Isotropic approximation to the filtering (using 'len' for lensed spectra)
        self.ftebl_len = {key: self.__compute_transfer(cls_key, nlev_key, transf_key, 'len') 
            for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}
        # Same using unlensed spectra (using 'unl' for unlensed spectra)
        self.ftebl_unl = {key: self.__compute_transfer(cls_key, nlev_key, transf_key, 'unl') 
            for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}

        self.spatial_type = filter_desc['spatial_type']
        self.estimator_type = filter_desc['estimator_type']
        self.libdir = filter_desc['libdir']

        self.simulationdata = filter_desc['simulationdata']

        self.chain_descr = filter_desc['chain_descr']
        self.cg_tol = filter_desc['QE_cg_tol']
        self.sht_threads = filter_desc['sht_threads']

        self.beam = filter_desc['beam'] #FIXME i dont want this here, can possibly be removed

        self.OBD = filter_desc['OBD']
        self.obd_libdir = filter_desc['obd_libdir']
        self.obd_rescale = filter_desc['obd_rescale']


    def _init_filterqest(self):
        if self.spatial_type == 'isotropic':
            self.ivf = filt_simple.library_fullsky_sepTP(opj(self.libdir, 'ivf'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
            if self.estimator_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(opj(self.libdir, 'qlms_dd'), self.ivf, self.ivf, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        elif self.spatial_type == 'anisotropic':
            ## Wait for finished run(), as plancklens triggers cinv_calc...
            self.cinv_t = filt_cinv.cinv_t(opj(self.libdir, 'cinv_t'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    self.ttebl['t'], self.niv_desc['T'],
                    marge_monopole=True, marge_dipole=True, marge_maps=[])

            # FIXME is this right? what if analysis includes pixelwindow function?
            transf_elm_loc = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_ivf[0])
            if self.OBD == 'OBD':
                nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
                self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir, 'cinv_p'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    transf_elm_loc[:self.lm_max_ivf[0]+1], self.niv_desc['P'], geom=nivjob_geomlib_, #self.nivjob_geomlib,
                    chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                    zbounds=(-1,1), _bmarg_lib_dir=self.obd_libdir, _bmarg_rescal=self.obd_rescale,
                    sht_threads=self.sht_threads)
            else:
                self.cinv_p = filt_cinv.cinv_p(opj(self.libdir, 'cinv_p'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    self.ttebl['e'], self.niv_desc['P'], chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                    transf_blm=self.ttebl['b'], marge_qmaps=(), marge_umaps=())
            _filter_raw = filt_cinv.library_cinv_sepTP(opj(self.libdir, 'ivf'), self.simulationdata, self.cinv_t, self.cinv_p, self.cls_len)
            _ftebl_rs = lambda x: np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[x])
            self.ivf = filt_util.library_ftl(_filter_raw, self.lm_max_qlm[0], _ftebl_rs(0), _ftebl_rs(1), _ftebl_rs(2))
            self.qlms_dd = qest.library_sepTP(opj(self.libdir, 'qlms_dd'), self.ivf, self.ivf, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        return self.qlms_dd


    def get_wflm(self, simidx, key, lm_max):
        if key in ['ptt']:
            return alm_copy(self.ivf.get_sim_tmliklm(simidx), None, *lm_max)
        elif key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'x_p', 'x_eb', 'xeb', 'x_be', 'xee']:
            return alm_copy(self.ivf.get_sim_emliklm(simidx), None, *lm_max)
        elif key in ['p']:
            return np.array([alm_copy(self.ivf.get_sim_tmliklm(simidx), None, *lm_max), alm_copy(self.ivf.get_sim_emliklm(simidx), None, *lm_max)])
        elif key in ['a_p']:
            return alm_copy(self.ivf.get_sim_emliklm(simidx), None, *lm_max)
        else:
            raise ValueError('Unknown estimator_key:', key)


    def get_ivflm(self, simidx, key, lm_max):
        if key in ['ptt']:
            return alm_copy(self.ivf.get_sim_tlm(simidx), None, *lm_max)
        elif key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'x_p', 'x_eb', 'xeb', 'x_be', 'xee']:
            return alm_copy(self.ivf.get_sim_elm(simidx), None, *lm_max), alm_copy(self.ivf.get_sim_blm(simidx), None, *lm_max)
        elif key in ['p']:
            return np.array([alm_copy(self.ivf.get_sim_tlm(simidx), None, *lm_max), alm_copy(self.ivf.get_sim_elm(simidx), None, *lm_max)])
        elif key in ['a_p']:
            return alm_copy(self.ivf.get_sim_elm(simidx), None, *lm_max), alm_copy(self.ivf.get_sim_blm(simidx), None, *lm_max)
        else:
            raise ValueError('Unknown estimator_key:', key)
        

    def get_response_unl(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_unl, self.cls_unl, self.ftebl_unl, lmax_qlm=lmax_qlm)
    

    def get_response_len(self, key, key0, lmax_qlm):
        return qresp.get_response(key, self.lm_max_ivf[0], key0, self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=lmax_qlm)
    

    def __compute_transfer(self, cls_key, nlev_key, transf_key, spectrum_type):
        cls = self.cls_len if spectrum_type == 'len' else self.cls_unl
        return cli(cls[cls_key][:self.lm_max_ivf[0] + 1] + df.a2r(self.nlev[nlev_key])**2 * cli(self.ttebl[transf_key] ** 2)) * (self.ttebl[transf_key] > 0)