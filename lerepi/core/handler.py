"""Base handler for lensing reconstruction pipelines.
    Loads and checks the configuration.
    Executes D.lensalot

Returns:
    _type_: _description_
"""

import importlib.util as iu
import sys

import lerepi
from lerepi.core.asserter import survey_asserter as sca, lensing_asserter as lca, run_asserter as rca
from lerepi.config.handler import survey_config
from lerepi.config.handler import run_config
from lerepi.config.handler import lensing_config
from lerepi.core.delensing_interface import Dlensalot


class handler():

    def __init__(self, parser):
        spec = iu.spec_from_file_location("module.name", lerepi.__file__+'/config/'+parser.config+'.py')
        foo = iu.module_from_spec(spec)
        sys.modules["module.name"] = foo
        spec.loader.exec_module(foo)
        config = foo.survey_config()

        self.rc = handler.get_run_config(config)
        self.sc = handler.get_survey_config(config)
        self.lc = handler.get_lensing_config(config, self.rc.TEMP)

    @sca
    def get_survey_config(config, TEMP):
        sc = survey_config(config, TEMP)
        return sc

    @lca
    def get_lensing_config(config):
        lc = lensing_config(config)
        return lc

    @rca
    def get_run_config(config):
        rc = run_config(config)
        return rc


    def run(self):
        dlensalot = Dlensalot(self.sc, self.lc, self.rc)
        dlensalot.run()
