"""Integration test: Setting up the correct model means that validator/, transformer/lerepi2delensalot, and core/handler did work together, so these 'modules' are integrated. Testing this for,
    - Full sky - polarization
    - Full sky - temperature
    - Masked sky - polarization
    Tests are considered successfull if the correct filters are initialized
"""


import unittest

import os
from os.path import join as opj
import numpy as np
import healpy as hp

import delensalot
from delensalot.run import run
from delensalot.config.visitor import transform, transform3d
from delensalot.core.opfilt.opfilt_handler import QE_transformer, MAP_transformer
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer 
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model
from delensalot.core.iterator.iteration_handler import iterator_transformer

from delensalot.core.opfilt import MAP_opfilt_aniso_p, MAP_opfilt_aniso_t, MAP_opfilt_iso_p, MAP_opfilt_iso_t, MAP_opfilt_iso_e
from delensalot.core.opfilt import QE_opfilt_aniso_p, QE_opfilt_aniso_t, QE_opfilt_iso_p, QE_opfilt_iso_t

os.environ['SCRATCH'] += 'test'

class Modeltester_FS_P(unittest.TestCase):
    """Full sky - polarization

    Args:
        unittest (_type_): _description_
    """


    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_CMBS4')
        job_id = 'generate_sim'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_CMBS4')
        job_id = 'build_OBD'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_QElensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_CMBS4')
        job_id = 'QE_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_MAPlensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_CMBS4')
        job_id = 'MAP_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_delens(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_CMBS4')
        job_id = 'delens'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

class Modeltester_MS_P(unittest.TestCase):
    """Masked sky - polarization

    Args:
        unittest (_type_): _description_
    """

    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_CMBS4')
        job_id = 'generate_sim'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_CMBS4')
        job_id = 'build_OBD'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_QElensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_CMBS4')
        job_id = 'QE_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_MAPlensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_CMBS4')
        job_id = 'MAP_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_delens(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='P_MS_CMBS4')
        job_id = 'delens'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

class Modeltester_FS_T(unittest.TestCase):
    """Masked sky - temperature

    Args:
        unittest (_type_): _description_
    """

    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_CMBS4')
        job_id = 'generate_sim'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_generatesim(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_CMBS4')
        job_id = 'build_OBD'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_QElensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_CMBS4')
        job_id = 'QE_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_MAPlensrec(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_CMBS4')
        job_id = 'MAP_lensrec'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        

    def test_delens(self):
        dlensalot_model = DLENSALOT_Model(defaults_to='T_FS_CMBS4')
        job_id = 'delens'
        model = transform3d(dlensalot_model, job_id, l2delensalotjob_Transformer())
        


if __name__ == '__main__':
    unittest.main()
