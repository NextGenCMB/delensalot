"""Integration test: Setting up the correct filter means that validator/, transformer/lerepi2delensalot, core/handler (and iteration_handler) did work together, so these 'modules' are integrated. Testing this for,
    - estimator keys
    - full sky / masked sky
    Tests are considered successfull if the correct filters are initialized

    COMMENT: For some reason, asserting fails if both classes are tested at the same time, i.e. `python -m unittest test_integration_filter` but this failing has nothing to do with delensalot itself.
    Recommend to use,
        `python -m unittest test_integration_filter.FS`,
        `python -m unittest test_integration_filter.MS`
    individually.
"""


import unittest
import shutil
import os
import delensalot
from delensalot.config.visitor import transform, transform3d
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer, l2T_Transformer
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Analysis

from delensalot.core.opfilt import MAP_opfilt_aniso_p, MAP_opfilt_aniso_t, MAP_opfilt_iso_p, MAP_opfilt_iso_t, MAP_opfilt_iso_e, MAP_opfilt_iso_tp, QE_opfilt_aniso_p, QE_opfilt_aniso_t, QE_opfilt_iso_p, QE_opfilt_iso_t

os.environ['SCRATCH'] += 'test'

class FS(unittest.TestCase):
    """Full sky - temperature

    Args:
        unittest (_type_): _description_
    """

    def __init__(self, args, **kwargs):
        super(FS, self).__init__(args, **kwargs)

        self.whitelist_FS_P = {
            'QE_lensrec': {
                'p_p': [QE_opfilt_iso_p.alm_filter_nlev],
                'pee': [QE_opfilt_iso_p.alm_filter_nlev],
                'p_eb': [QE_opfilt_iso_p.alm_filter_nlev], 
                'p_be': [QE_opfilt_iso_p.alm_filter_nlev],
                'peb': [QE_opfilt_iso_p.alm_filter_nlev],
            },'MAP_lensrec': {
                'p_p': [MAP_opfilt_iso_p.alm_filter_nlev_wl],
                'pee': [MAP_opfilt_iso_p.alm_filter_nlev_wl, MAP_opfilt_iso_e.alm_filter_nlev_wl],
                'p_eb': [MAP_opfilt_iso_p.alm_filter_nlev_wl], 
                'p_be': [MAP_opfilt_iso_p.alm_filter_nlev_wl],
                'peb': [MAP_opfilt_iso_p.alm_filter_nlev_wl]}
        }

        self.whitelist_FS_T = {
            'QE_lensrec': {
                'ptt': [QE_opfilt_iso_t.alm_filter_nlev],

            },'MAP_lensrec': {
                'ptt': [MAP_opfilt_iso_t.alm_filter_nlev_wl],
            }
        }

        self.whitelist_FS_TP = {
            # 'MAP_lensrec': {
            #     'p': [MAP_opfilt_iso_tp.alm_filter_nlev_wl],
            # }
        }

        
    def test_fullsky_T(self):
        for job_id, key_dict in self.whitelist_FS_T.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_FS_T[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_FS_T[job_id][key], key)
                del model, dlensalot_model

    def test_fullsky_P(self):
        for job_id, key_dict in self.whitelist_FS_P.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_FS_P[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_FS_P[job_id][key], key)
                del model, dlensalot_model

    def test_fullsky_TP(self):
        for job_id, key_dict in self.whitelist_FS_TP.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='TP_FS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_FS_TP[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_FS_TP[job_id][key], key)
                del model, dlensalot_model


class MS(unittest.TestCase):
    """Full sky - temperature

    Args:
        unittest (_type_): _description_
    """

    def __init__(self, args, **kwargs):
        super(MS, self).__init__(args, **kwargs)


        self.whitelist_MS_P = {
            'QE_lensrec': {
                'p_p': [QE_opfilt_aniso_p.alm_filter_ninv],
                # 'pee': [QE_opfilt_aniso_p.alm_filter_ninv],
                'p_eb': [QE_opfilt_aniso_p.alm_filter_ninv], 
                'p_be': [QE_opfilt_aniso_p.alm_filter_ninv],
                'peb': [QE_opfilt_aniso_p.alm_filter_ninv],
            },'MAP_lensrec': {
                'p_p': [MAP_opfilt_aniso_p.alm_filter_ninv_wl],
                # 'pee': [MAP_opfilt_aniso_p.alm_filter_ninv_wl],
                'p_eb': [MAP_opfilt_aniso_p.alm_filter_ninv_wl], 
                'p_be': [MAP_opfilt_aniso_p.alm_filter_ninv_wl],
                'peb': [MAP_opfilt_aniso_p.alm_filter_ninv_wl]}
        }

        self.whitelist_MS_T = {
            'QE_lensrec': {
                'ptt': [QE_opfilt_aniso_t.alm_filter_ninv],

            },'MAP_lensrec': {
                'ptt': [MAP_opfilt_aniso_t.alm_filter_ninv_wl],
            }
        }

        self.whitelist_MS_TP = {
            # 'MAP_lensrec': {
            #     'p': [MAP_opfilt_aniso_tp.alm_filter_ninv_wl],
            # }
        }


    def test_maskedsky_T(self):
        for job_id, key_dict in self.whitelist_MS_T.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_MS_T[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_MS_T[job_id][key], key)
                del model, dlensalot_model

    def test_maskedsky_P(self):
        for job_id, key_dict in self.whitelist_MS_P.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_MS_P[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_MS_P[job_id][key], key)
                del model, dlensalot_model

    def test_maskedsky_TP(self):
        for job_id, key_dict in self.whitelist_MS_TP.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='TP_MS_TEST', analysis = DLENSALOT_Analysis(key=key))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
                assert type(model.filter) in self.whitelist_MS_TP[job_id][key], "{} != {} for key {}".format(model.filter, self.whitelist_MS_TP[job_id][key], key)
                del model, dlensalot_model


if __name__ == '__main__':
    unittest.main()
    temppath = os.environ["SCRATCH"]+"/delensalot"
    if os.path.exists(temppath):
        shutil.rmtree(temppath)
