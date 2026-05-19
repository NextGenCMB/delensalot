"""
Integration smoke test: full pipeline run (sim generation, QE, MAP).
Tuned for CI speed: high noise, low lmax, 2 iterations, loose cg_tol.
Passes if all steps complete without error and outputs are finite.

Run with:
    python -m unittest tests/test_integration_mwe.py
"""

import unittest
import shutil
import os
import tempfile

import numpy as np
import healpy as hp

import delensalot
from delensalot.run import run
from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.metamodel.dlensalot_mm import (
    DLENSALOT_Model, DLENSALOT_Job, DLENSALOT_Analysis,
    DLENSALOT_Simulation, DLENSALOT_Noisemodel,
    DLENSALOT_Qerec, DLENSALOT_Itrec,
)

# CI-friendly parameters: fast but still exercises the full pipeline
LMAX   = 2000
NSIDE  = 2048
NLEV_P = 10.0          # muK-arcmin — high noise → fast CG convergence
NLEV_T = 10.0 / np.sqrt(2)
BEAM   = 5.0          # arcmin
ITMAX  = 2
CG_TOL = 1e-3


def _make_model(temp_suffix='ci_mwe'):
    return DLENSALOT_Model(
        defaults_to='default_CMBS4_fullsky_polarization',
        job=DLENSALOT_Job(
            jobs=["generate_sim", "QE_lensrec", "MAP_lensrec"],
        ),
        analysis=DLENSALOT_Analysis(
            key='p_p',
            simidxs=np.arange(0, 1),
            TEMP_suffix=temp_suffix,
            beam=BEAM,
            lm_max_ivf=(LMAX, LMAX),
        ),
        simulationdata=DLENSALOT_Simulation(
            space='cl',
            flavour='unl',
            lmax=LMAX,
            phi_lmax=LMAX + 512,
            transfunction=gauss_beam(BEAM / 180 / 60 * np.pi, lmax=LMAX),
            nlev={'P': NLEV_P, 'T': NLEV_T},
            geominfo=('healpix', {'nside': NSIDE}),
            lenjob_geominfo=('thingauss', {'lmax': LMAX + 300, 'smax': 3}),
            CMB_fn=os.path.join(
                os.path.dirname(delensalot.__file__),
                'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat',
            ),
        ),
        noisemodel=DLENSALOT_Noisemodel(
            nlev={'P': NLEV_P, 'T': NLEV_T},
            geominfo=('healpix', {'nside': NSIDE}),
        ),
        qerec=DLENSALOT_Qerec(
            tasks=["calc_phi"],
            lm_max_qlm=(LMAX, LMAX),
            cg_tol=CG_TOL,
        ),
        itrec=DLENSALOT_Itrec(
            tasks=["calc_phi"],
            itmax=ITMAX,
            lm_max_unl=(LMAX + 200, LMAX + 200),
            lm_max_qlm=(LMAX, LMAX),
            lenjob_geominfo=('thingauss', {'lmax': LMAX + 300, 'smax': 3}),
            cg_tol=CG_TOL,
        ),
    )


class MWE(unittest.TestCase):
    """Minimal working example: sim → QE → MAP on full sky, low-res, high noise."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='delensalot_ci_')
        os.environ.setdefault('SCRATCH', cls.tmpdir)
        cls.model = _make_model()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run(self, job_id):
        runner = run(config_fn="", config_model=self.model, job_id=job_id, verbose=False)
        runner.run()
        return runner.collect_model()

    def test_01_generate_sim(self):
        ana = self._run('generate_sim')
        obs = ana.simulationdata.get_sim_obs(
            space='alm', field='polarization', spin=0, simidx=0)
        for field, alm in zip(['E', 'B'], obs):
            cl = hp.alm2cl(alm)
            self.assertTrue(np.all(np.isfinite(cl)),
                            f"Observed {field} Cl contains non-finite values")
            self.assertGreater(np.sum(cl), 0,
                               f"Observed {field} Cl is all zeros")

    def test_02_qe_reconstruction(self):
        ana = self._run('QE_lensrec')
        plm = ana.get_plm_it(0, [0])[0]
        cl = hp.alm2cl(plm)
        self.assertTrue(np.all(np.isfinite(cl)),
                        "QE phi Cl contains non-finite values")
        self.assertGreater(np.mean(cl[2:100]), 0,
                           "QE phi Cl is zero at low-L")

    def test_03_map_reconstruction(self):
        ana = self._run('MAP_lensrec')
        plm_QE  = ana.get_plm_it(0, [0])[0]
        plm_MAP = ana.get_plm_it(0, [ITMAX])[0]

        cl_QE  = hp.alm2cl(plm_QE)
        cl_MAP = hp.alm2cl(plm_MAP)

        self.assertTrue(np.all(np.isfinite(cl_MAP)),
                        "MAP phi Cl contains non-finite values")

        # MAP and QE should have similar power within a factor of 3
        ratio = np.mean(cl_MAP[2:100]) / np.mean(cl_QE[2:100])
        self.assertGreater(ratio, 0.3,
                           f"MAP/QE power ratio {ratio:.3f} suspiciously low")
        self.assertLess(ratio, 3.0,
                        f"MAP/QE power ratio {ratio:.3f} suspiciously high")


if __name__ == '__main__':
    unittest.main()