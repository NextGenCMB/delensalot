"""Integration test: Setting up the correct filter means that validator/, transformer/lerepi2delensalot, core/handler (and iteration_handler) did work together, so these 'modules' are integrated. Testing this for,
    - estimator keys
    - full sky / masked sky
    Tests are considered successfull if the correct filters are initialized

"""


import unittest
import shutil
import os
from os.path import join as opj
from pathlib import Path

import numpy as np
import healpy as hp
import delensalot
from delensalot.run import run
from delensalot.config.visitor import transform, transform3d
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer, l2T_Transformer
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Analysis

from delensalot.core.opfilt import MAP_opfilt_aniso_p, MAP_opfilt_aniso_t, MAP_opfilt_iso_p, MAP_opfilt_iso_t, MAP_opfilt_iso_e, MAP_opfilt_iso_tp, QE_opfilt_aniso_p, QE_opfilt_aniso_t, QE_opfilt_iso_p, QE_opfilt_iso_t

os.environ['SCRATCH'] += 'test'

class Tutorial(unittest.TestCase):
    """Full sky - temperature

    Args:
        unittest (_type_): _description_
    """

    def __init__(self, args, **kwargs):
        super(Tutorial, self).__init__(args, **kwargs)

        self.job_ids = ['generate_sim', 'QE_lensrec', 'MAP_lensrec']

        
    def test_conf_mwe_applynoOBD(self):
        m = np.zeros(hp.nside2npix(1))
        m[[4]] = 1
        nside = 512
        rhits = np.abs(hp.smoothing(hp.ud_grade(m, nside_out=nside),0.1))
        mask = hp.ud_grade(m, nside_out=nside)
        mask_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits')
        rhits_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits')

        if not os.path.isdir(os.path.dirname(mask_fn)):
            os.makedirs(os.path.dirname(mask_fn))
            
        if not os.path.isfile(mask_fn):
            hp.write_map(mask_fn, mask)
            hp.write_map(rhits_fn, rhits)
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_applynoOBD.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()
        assert 1

    def test_conf_mwe_applyOBD(self):
        m = np.zeros(hp.nside2npix(1))
        m[[4]] = 1
        nside = 512
        rhits = np.abs(hp.smoothing(hp.ud_grade(m, nside_out=nside),0.1))
        mask = hp.ud_grade(m, nside_out=nside)
        mask_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits')
        rhits_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits')

        if not os.path.isdir(os.path.dirname(mask_fn)):
            os.makedirs(os.path.dirname(mask_fn))
            
        if not os.path.isfile(mask_fn):
            hp.write_map(mask_fn, mask)
            hp.write_map(rhits_fn, rhits)
        tniti_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'tniti.npy')
        np.save(tniti_fn, np.random.random(100))
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_applyOBD.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()
        assert 1

    def test_conf_mwe_buildOBD(self):
        m = np.zeros(hp.nside2npix(1))
        m[[4]] = 1
        nside = 512
        rhits = np.abs(hp.smoothing(hp.ud_grade(m, nside_out=nside),0.1))
        mask = hp.ud_grade(m, nside_out=nside)
        mask_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits')
        rhits_fn = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits')

        if not os.path.isdir(os.path.dirname(mask_fn)):
            os.makedirs(os.path.dirname(mask_fn))
            
        if not os.path.isfile(mask_fn):
            hp.write_map(mask_fn, mask)
            hp.write_map(rhits_fn, rhits)
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_buildOBD.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()
        assert 1

    def test_conf_mwe_fastWF(self):
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_fastWF.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()
        assert 1

    def test_conf_mwe_fullsky(self):
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_fullsky.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()
        assert 1

    def test_conf_mwe_maskedsky(self):
        m = np.zeros(hp.nside2npix(1))
        m[[7]] = 1
        mask = hp.ud_grade(m, nside_out=2048)
        print('fsky: {:.3f}'.format(np.mean(m)))
        mask_fn = opj(os.environ['SCRATCH'], 'analysis/mfda_maskedsky_lminB200/mask.fits')
        if not os.path.isdir(os.path.dirname(mask_fn)):
            os.makedirs(os.path.dirname(mask_fn))
        if not os.path.isfile(mask_fn):
            hp.write_map(mask_fn, mask)
        for job_id in self.job_ids:
            fn = opj(Path(delensalot.__file__).parent.parent, 'first_steps/notebooks/', 'conf_mwe_maskedsky.py')
            delensalot_runner = run(config_fn=fn, job_id=job_id, verbose=True)
            ana_mwe = delensalot_runner.init_job()


if __name__ == '__main__':
    unittest.main()
    temppath = os.environ["SCRATCH"]+"/delensalot"
    if os.path.exists(temppath):
        shutil.rmtree(temppath)
