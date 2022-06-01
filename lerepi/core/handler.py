#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Loads and checks the configuration.
    Executes D.lensalot
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end

from lerepi.core.visitor import transform
from lerepi.transformer.param2dlensalot import p2d_Transformer, p2l_Transformer, transform


class handler():

    def __init__(self, parser):
        spec = iu.spec_from_file_location('paramfile', parser.config_file)
        paramfile = iu.module_from_spec(spec)
        sys.modules['paramfile'] = paramfile
        spec.loader.exec_module(paramfile)
        self.dlensalot_model = transform(paramfile.dlensalot_model, p2d_Transformer())
        
        # Store the dlensalot_model as parameterfile in TEMP, to use for resume purposes
        if parser.resume != '':
            # This is only done if not resuming. Otherwise it would be there already
            if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
                # Check if it is really a file 
                shutil.copyfile(parser.config_file, self.dlensalot_model.TEMP+'/'+parser.config_file.split('/')[-1])

        # In case there are settings which are solely for lerepi
        self.lerepi_model = transform(paramfile.dlensalot_model, p2l_Transformer())


    # TODO implement if needed
    @log_on_start(logging.INFO, "Start of do_something_with_params()")
    @log_on_end(logging.INFO, "Finished do_something_with_params()")
    def do_something_with_params(lerepi_model):
        """If there is a lerepi model, then possibly settings might be set here. Pass for now

        Args:
            lerepi_model (_type_): _description_
        """
        pass

    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        handler.do_something_with_params(self.lerepi_model)
        self.dlensalot_model.run()
