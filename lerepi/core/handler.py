#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Loads and checks the configuration.
    Executes D.lensalot
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import importlib.util as iu
import os
import sys

from logdecorator import log_on_start, log_on_end

import lerepi
from lerepi.core.asserter import survey_asserter as sca, lensing_asserter as lca, run_asserter as rca
from lerepi.config.handler import survey_config
from lerepi.config.handler import run_config
from lerepi.config.handler import lensing_config
from lerepi.core.delensing_interface import Dlensalot

# TODO Only differentiate between lensing_config and lerepi_config
# TODO lerepi_config is checked here, lensing_config later maybe?
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
    @log_on_start
    @log_on_end
    def get_run_config(config):
        rc = run_config(config)
        return rc

    @sca
    @log_on_start
    @log_on_end
    def get_survey_config(config):
        sc = survey_config(config)
        return sc

    @lca
    @log_on_start
    @log_on_end
    def get_lensing_config(config, TEMP):
        lc = lensing_config(config, TEMP)
        return lc

    # TODO only lensing_config should be handed over to dlensalot
    def run(self):
        dlensalot = Dlensalot(self.lc)
        dlensalot.run()
