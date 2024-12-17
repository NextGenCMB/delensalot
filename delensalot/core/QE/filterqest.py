from os.path import join as opj
import numpy as np

from plancklens import qest

from lenspyx.lensing import get_geom 

from delensalot.utility.utils_hp import gauss_beam
from delensalot.utility.utils_hp import alm_copy
from delensalot.core.ivf import filt_util, filt_cinv, filt_simple
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD

class base:
    def __init__(self, filter_desc):
        # This class tries to hide the uglyness to interfacing with plancklens (no offense)
        self.lm_max_ivf = filter_desc['lm_max_ivf']
        self.lm_max_qlm = filter_desc['lm_max_qlm']
        self.lm_max_len = filter_desc['lm_max_len']
        self.lm_max_len = filter_desc['lm_max_unl']
        self.lmin_teb = filter_desc['lmin_teb']

        self.ftebl_len = filter_desc['ftebl_len']
        self.fteb_unl = filter_desc['fteb_unl']
        self.ttebl = filter_desc['ttebl']

        self.qe_filter_directional = filter_desc['qe_filter_directional']
        self.estimator_type = filter_desc['estimator_type']
        self.libdir_QE = filter_desc['libdir_QE']

        self.simulationdata = filter_desc['simulationdata']

        self.nivjob_geominfo = filter_desc['nivjob_geominfo']
        self.nivt_desc = filter_desc['nivt_desc']
        self.nivp_desc = filter_desc['nivp_desc']
        
        self.cls_len = filter_desc['cls_len']
        self.cls_unl = filter_desc['cls_unl']

        self.chain_descr = filter_desc['chain_descr']
        self.cg_tol = filter_desc['QE_cg_tol']
        self.sht_threads = filter_desc['sht_threads']

        self.beam = filter_desc['beam'] #FIXME i dont want this here, can possibly be removed

        self.OBD = filter_desc['OBD']
        self.obd_libdir = filter_desc['obd_libdir']
        self.obd_rescale = filter_desc['obd_rescale']


    def _init_filterqest(self):
        if self.qe_filter_directional == 'isotropic':
            self.ivf = filt_simple.library_fullsky_sepTP(opj(self.libdir_QE, 'ivf'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
            if self.estimator_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivf, self.ivf, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        elif self.qe_filter_directional == 'anisotropic':
            ## Wait for finished run(), as plancklens triggers cinv_calc...
            self.cinv_t = filt_cinv.cinv_t(opj(self.libdir_QE, 'cinv_t'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    self.ttebl['t'], self.nivt_desc,
                    marge_monopole=True, marge_dipole=True, marge_maps=[])

            # FIXME is this right? what if analysis includes pixelwindow function?
            transf_elm_loc = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_ivf[0])
            if self.OBD == 'OBD':
                nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
                self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    transf_elm_loc[:self.lm_max_ivf[0]+1], self.nivp_desc, geom=nivjob_geomlib_, #self.nivjob_geomlib,
                    chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                    zbounds=(-1,1), _bmarg_lib_dir=self.obd_libdir, _bmarg_rescal=self.obd_rescale,
                    sht_threads=self.sht_threads)
            else:
                self.cinv_p = filt_cinv.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                    self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                    self.ttebl['e'], self.nivp_desc, chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                    transf_blm=self.ttebl['b'], marge_qmaps=(), marge_umaps=())
            _filter_raw = filt_cinv.library_cinv_sepTP(opj(self.libdir_QE, 'ivf'), self.simulationdata, self.cinv_t, self.cinv_p, self.cls_len)
            _ftebl_rs = lambda x: np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[x])
            self.ivf = filt_util.library_ftl(_filter_raw, self.lm_max_qlm[0], _ftebl_rs(0), _ftebl_rs(1), _ftebl_rs(2))
            self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivf, self.ivf, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])


    def get_wflm(self, simidx):
        if self.estimator_key in ['ptt']:
            return lambda: alm_copy(self.ivf.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivf.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p']:
            return lambda: np.array([alm_copy(self.ivf.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivf.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])
        
    def get_ivf(self, simidx):
        if self.estimator_key in ['ptt']:
            return lambda: alm_copy(self.ivf.get_sim_tlm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivf.get_sim_elm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.estimator_key in ['p']:
            return lambda: np.array([alm_copy(self.ivf.get_sim_tlm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivf.get_sim_emlm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])
        