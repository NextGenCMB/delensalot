#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Loads and checks the configuration.
    Executes D.lensalot
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
import sys
import importlib.util as iu

import logging
from logdecorator import log_on_start, log_on_end

import lerepi
from lerepi.core.visitor import transform


from lerepi.transformer.param2dlensalot import p2d_Transformer, p2l_Transformer, transform
from lerepi.core.delensing_interface import Dlensalot


class handler():

    def __init__(self, parser):
        module_path = os.path.dirname(lerepi.__file__)
        paramfile_path = module_path+'/config/'+parser.config_file
        spec = iu.spec_from_file_location('paramfile', paramfile_path)
        paramfile = iu.module_from_spec(spec)
        sys.modules['paramfile'] = paramfile
        spec.loader.exec_module(paramfile)

        self.dlensalot_model = transform(paramfile, p2d_Transformer())
        # In case there are settings which are solely for lerepi
        self.lerepi_model = transform(paramfile, p2l_Transformer())


    # TODO implement if needed
    def do_something_with_params(lerepi_model):
        """If there is a lerepi model, then possibly settings might be set here. Pass for now

        Args:
            lerepi_model (_type_): _description_
        """
        pass


    def run(self):
        handler.do_something_with_params(self.lerepi_model)
        dlensalot = Dlensalot(self.dlensalot_model)
        dlensalot.run()
