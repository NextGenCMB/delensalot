"""Integration test: Full QE and MAP reconstruction. Testing this for,
    - estimator keys
    - full sky / masked sky
    Tests are considered successfull if residual lensing amplitude is within expectation

    COMMENT: For some reason, asserting fails if both classes are tested at the same time, i.e. `python3 -m unittest test_integration_reconstruction` but this failing has nothing to do with delensalot itself.
    
    E.g.,
        python3 -m unittest test_integration_reconstruction.FS.test_P_approx

"""


import unittest
import shutil
import os
from os.path import join as opj
import healpy as hp
import numpy as np

import delensalot
from delensalot.run import run
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam, alm_copy
from delensalot.config.visitor import transform, transform3d
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer, l2T_Transformer
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Analysis, DLENSALOT_Job, DLENSALOT_Itrec
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
                # 'pee': [QE_opfilt_iso_p.alm_filter_nlev],
                # 'p_eb': [QE_opfilt_iso_p.alm_filter_nlev], 
                # 'p_be': [QE_opfilt_iso_p.alm_filter_nlev],
                # 'peb': [QE_opfilt_iso_p.alm_filter_nlev],
            },'MAP_lensrec': {
                'p_p': [MAP_opfilt_iso_p.alm_filter_nlev_wl],
                # 'pee': [MAP_opfilt_iso_p.alm_filter_nlev_wl, MAP_opfilt_iso_e.alm_filter_nlev_wl],
                # 'p_eb': [MAP_opfilt_iso_p.alm_filter_nlev_wl], 
                # 'p_be': [MAP_opfilt_iso_p.alm_filter_nlev_wl],
                # 'peb': [MAP_opfilt_iso_p.alm_filter_nlev_wl]
            }
        }

        self.whitelist_FS_T = {
            # 'QE_lensrec': {
            #     'ptt': [QE_opfilt_iso_t.alm_filter_nlev],

            # },'MAP_lensrec': {
            #     'ptt': [MAP_opfilt_iso_t.alm_filter_nlev_wl],
            # }
        }

        self.whitelist_FS_TP = {
            # 'MAP_lensrec': {
            #     'p': [MAP_opfilt_iso_tp.alm_filter_nlev_wl],
            # }
        }

        self.Al_assert = {
            'QE_lensrec': {
                'p_p': 0.29, # 0.28669273301356524
                # 'pee': np.inf,
                # 'p_eb': np.inf,
                # 'p_be': np.inf,
                # 'peb': np.inf,
            },'MAP_lensrec': {
                'p_p': 0.16, # 0.10060576537910079
                # 'pee': np.inf,
                # 'p_eb': np.inf,
                # 'p_be': np.inf,
                # 'peb': np.inf,
            }
        }


    def test_P_approx(self):
        """ P_FS_TEST may be somewhat too low for proper reconstruction (cg solver wouldn't converge), but works for fastWF. Result probably quite inaccurate, but this test checks if reconstruction runs until the end, which is what we want to test here 
        """        
        use_approximateWF = True
        for job_id, key_dict in self.whitelist_FS_P.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_TEST', analysis = DLENSALOT_Analysis(key=key, TEMP_suffix='test'), itrec = DLENSALOT_Itrec(itmax=3))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                delensalot_runner = run(config_fn='', job_id='generate_sim', config_model=dlensalot_model, verbose=True)
                ana_mwe = delensalot_runner.init_job()
                bsky = ana_mwe.simulationdata.get_sim_sky(simidx=0, field='polarization', space='alm', spin=0)[1]
                
                obs = ana_mwe.simulationdata.get_sim_obs(simidx=0, field='polarization', space='alm', spin=0)

                if job_id == 'QE_lensrec':
                    dlensalot_model.itrec.itmax = 0
                blt = delensalot.map2tempblm(
                    hp.alm2map_spin(obs, nside=ana_mwe.simulationdata.geominfo[1]['nside'], spin=2, lmax=ana_mwe.simulationdata.lmax), 
                    lmax_cmb=dlensalot_model.analysis.lm_max_ivf[0], 
                    beam=dlensalot_model.analysis.beam, 
                    itmax=dlensalot_model.itrec.itmax, 
                    nlev=dlensalot_model.noisemodel.nlev, 
                    use_approximateWF=use_approximateWF, 
                    defaults_to='P_FS_TEST', 
                    verbose=True, )

                bsky = alm_copy(bsky, None, hp.Alm.getlmax(blt.size), hp.Alm.getlmax(blt.size))
                input = hp.alm2cl(bsky, lmax=200)
                output = hp.alm2cl(bsky-blt, lmax=200)
                Al = np.mean(output[30:200]/input[30:200])
                assert Al < self.Al_assert[job_id][key], "{}, {}, {}, {}".format(job_id, key, Al, self.Al_assert[job_id][key])
                print(Al, self.Al_assert[job_id][key])


    def test_P(self):
        assert 0, 'cg-solver will not converge, change defaults if this should be integrated into workflow.'
        use_approximateWF = False
        for job_id, key_dict in self.whitelist_FS_P.items():
            for key in key_dict:
                dlensalot_model = DLENSALOT_Model(defaults_to='P_FS_TEST', analysis = DLENSALOT_Analysis(key=key, TEMP_suffix='test'), itrec = DLENSALOT_Itrec(itmax=3))
                delensalot.del_TEMP(transform(dlensalot_model, l2T_Transformer()))
                delensalot_runner = run(config_fn='', job_id='generate_sim', config_model=dlensalot_model, verbose=True)
                ana_mwe = delensalot_runner.init_job()
                bsky = ana_mwe.simulationdata.get_sim_sky(simidx=0, field='polarization', space='alm', spin=0)[1]
                obs = ana_mwe.simulationdata.get_sim_obs(simidx=0, field='polarization', space='alm', spin=0)

                blt = delensalot.map2tempblm(obs, lmax_cmb=dlensalot_model.analysis.lm_max_ivf[0], beam=dlensalot_model.analysis.beam, itmax=dlensalot_model.itrec.itmax, nlev=dlensalot_model.noisemodel.nlev, use_approximateWF=use_approximateWF, verbose=True, )

                input = hp.alm2cl(bsky, lmax=200)
                output = hp.alm2cl(bsky-blt, lmax=200)
                Al = np.mean(output[30:200]/input[30:200])
                assert Al < self.Al_assert[job_id][key], "{}, {}, {}, {}".format(job_id, key, Al, self.Al_assert[job_id][key])
                print(Al, self.Al_assert[job_id][key])


if __name__ == '__main__':
    unittest.main(exit=False)
    temppath = os.environ["SCRATCH"]+"/delensalot"
    if os.path.exists(temppath):
        shutil.rmtree(temppath)
