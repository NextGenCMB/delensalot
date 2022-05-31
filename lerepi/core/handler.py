"""Base handler for lensing reconstruction pipelines.
    Loads and checks the configuration.
    Executes D.lensalot

Returns:
    _type_: _description_
"""

import importlib.util as iu
import os
import sys

import lerepi
from lerepi.core.asserter import survey_asserter as sca, lensing_asserter as lca, run_asserter as rca
from lerepi.config.handler import survey_config
from lerepi.config.handler import run_config
from lerepi.config.handler import lensing_config
from lerepi.core.delensing_interface import Dlensalot


class handler():

    def __init__(self, parser):
        module_path = os.path.dirname(lerepi.__file__)
        paramfile_path = module_path+'/config/'+parser.config_file

        spec = iu.spec_from_file_location('paramfile', paramfile_path)
        paramfile = iu.module_from_spec(spec)
        sys.modules['paramfile'] = paramfile
        spec.loader.exec_module(paramfile)

        self.rc = handler.get_run_config(paramfile)
        self.sc = handler.get_survey_config(paramfile)
        self.lc = handler.get_lensing_config(paramfile, self.rc.TEMP)

    @rca
    def get_run_config(config):
        rc = run_config(config)
        return rc

    @sca
    def get_survey_config(config):
        sc = survey_config(config)
        return sc

    @lca
    def get_lensing_config(config, TEMP):
        lc = lensing_config(config, TEMP)
        return lc


    def run(self):
        dlensalot = Dlensalot(self.sc, self.lc, self.rc)
        dlensalot.run()
