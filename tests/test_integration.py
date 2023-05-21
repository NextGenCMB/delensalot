import unittest

import os
from os.path import join as opj
if "SCRATCH" not in os.environ:
    os.environ["SCRATCH"] = "/mnt/c/Users/sebas/OneDrive/SCRATCH"
import numpy as np
import healpy as hp

from delensalot.run import run

class JobTester(unittest.TestCase):

    def test_generatesim(self):
        fn = opj(os.getcwd(), 'conf_defaults.py')
        ana_mwe = run(config=fn, job_id='generate_sim', verbose=False)
        ana_mwe.run()
        ana_mwe = ana_mwe.job

        ans = sum([1, 2, 3])
        assert ans == 6, "Should be 6, got {}".format(ans)

    def test_generatesim(self):
        fn = opj(os.getcwd(), 'conf_defaults.py')
        ana_mwe = run(config=fn, job_id='build_OBD', verbose=False)
        ana_mwe.run()
        ana_mwe = ana_mwe.job


    def test_QElensrec(self):
        fn = opj(os.getcwd(), 'conf_defaults.py')
        ana_mwe = run(config=fn, job_id='QE_lensrec', verbose=False)
        ana_mwe.run()
        ana_mwe = ana_mwe.job


    def test_MAPlensrec(self):
        fn = opj(os.getcwd(), 'conf_defaults.py')
        ana_mwe = run(config=fn, job_id='MAP_lensrec', verbose=False)
        ana_mwe.run()
        ana_mwe = ana_mwe.job


    def test_delens(self):
        fn = opj(os.getcwd(), 'conf_defaults.py')
        ana_mwe = run(config=fn, job_id='delens', verbose=False)
        ana_mwe.run()
        ana_mwe = ana_mwe.job


if __name__ == '__main__':
    unittest.main()