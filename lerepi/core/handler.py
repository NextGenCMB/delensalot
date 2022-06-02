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
        paramfile = handler.load_paramfile(parser.config_file, 'paramfile')
        self.dlensalot_model = transform(paramfile.dlensalot_model, p2d_Transformer())
        self.store(parser, paramfile, self.dlensalot_model.TEMP)
        # In case there are settings which are solely for lerepi
        self.lerepi_model = transform(paramfile.dlensalot_model, p2l_Transformer())


    def store(self, parser, paramfile, TEMP):
        # Store the dlensalot_model as parameterfile in TEMP, to use if run resumed
        dostore = True
        if parser.resume == '':
            # This is only done if not resuming. Otherwise file would already exist
            if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
                # if the file already exists, check if something changed
                if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                    print('Param file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                    paramfile_old = handler.load_paramfile(TEMP+'/'+parser.config_file.split('/')[-1], 'paramfile_old')   
                    for key, val in paramfile_old.dlensalot_model.__dict__.items():
                        for k, v in val.__dict__.items():
                            if v != paramfile.dlensalot_model.__dict__[key].__dict__[k]:
                                if callable(v):
                                    # If it's a function, we can test if bytecode is the same as a simple check won't work due to pointing to memory location
                                    if v.__code__.co_code != paramfile.dlensalot_model.__dict__[key].__dict__[k].__code__.co_code:
                                        print("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, paramfile.dlensalot_model.__dict__[key].__dict__[k]))
                                        dostore=False
        if dostore:
            print('Parameterfile stored at ', TEMP+'/'+parser.config_file.split('/')[-1])
            shutil.copyfile(parser.config_file, TEMP+'/'+parser.config_file.split('/')[-1])
        else:
            print('New parameterfile not stored due to conflict with already existing parameterfile')


    @log_on_start(logging.INFO, "Start of load_param()")
    @log_on_end(logging.INFO, "Finished load_param()")
    def load_paramfile(directory, descriptor):
        spec = iu.spec_from_file_location('paramfile', directory)
        p = iu.module_from_spec(spec)
        sys.modules[descriptor] = p
        spec.loader.exec_module(p)
        return p


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
