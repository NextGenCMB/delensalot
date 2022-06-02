#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Collects lerepi jobs defined in DLENSALOT_jobs
    Extracts models needed for each job via x2y_Transformer
    runs all jobs
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end

from lerepi.core.visitor import transform
from lerepi.transformer.param2dlensalot import p2d_Transformer, p2l_Transformer, p2v_Transformer, p2b_Transformer, p2T_Transformer, transform

import lenscarf.core.handler as lenscarf_handler

class handler():

    def __init__(self, parser):
        self.paramfile = handler.load_paramfile(parser.config_file, 'paramfile')
        TEMP = transform(self.paramfile.dlensalot_model, p2T_Transformer())
        self.store(parser, self.paramfile, TEMP)

        self.dlensalot_model = transform(self.paramfile.dlensalot_model, p2d_Transformer())
        # In case there are settings which are solely for lerepi
        self.lerepi_model = transform(self.paramfile.dlensalot_model, p2l_Transformer())


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        def _process_jobparams(pf):
            jobs = []
            if pf.QE_delensing:
                jobs.append(((self.paramfile.dlensalot_model, p2d_Transformer()), lenscarf_handler.QE_delensing))
            if pf.MAP_delensing:
                jobs.append(((self.paramfile.dlensalot_model, p2d_Transformer()), lenscarf_handler.MAP_delensing))
            if pf.Btemplate_per_iteration:
                jobs.append(((self.paramfile.dlensalot_model, p2b_Transformer()), lenscarf_handler.B_template_construction))
            if pf.inspect_result:
                jobs.append(((self.paramfile.dlensalot_model, p2v_Transformer()), lenscarf_handler.inspect_result))
            return jobs
        self.jobs = _process_jobparams(self.paramfile.dlensalot_model.job)


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        for transf, job in self.jobs:
            model = transform(*transf)
            j = job(model)
            j.collect_jobs()
            j.run()


    @log_on_start(logging.INFO, "Start of store()")
    @log_on_end(logging.INFO, "Finished store()")
    def store(self, parser, paramfile, TEMP):
        """ Store the dlensalot_model as parameterfile in TEMP, to use if run resumed

        Args:
            parser (_type_): _description_
            paramfile (_type_): _description_
            TEMP (_type_): _description_
        """
        dostore = True
        if parser.resume == '':
            # This is only done if not resuming. Otherwise file would already exist
            if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
                dostore = False
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
                                        print('Exit. Check config file.')
                                        sys.exit()
        if dostore:
            print('Parameterfile stored at ', TEMP+'/'+parser.config_file.split('/')[-1])
            shutil.copyfile(parser.config_file, TEMP+'/'+parser.config_file.split('/')[-1])
        else:
            print('Matching parameterfile found. Resuming where you left off.')


    @log_on_start(logging.INFO, "Start of load_paramfile()")
    @log_on_end(logging.INFO, "Finished load_paramfile()")
    def load_paramfile(directory, descriptor):
        """Load parameterfile

        Args:
            directory (_type_): _description_
            descriptor (_type_): _description_

        Returns:
            _type_: _description_
        """
        spec = iu.spec_from_file_location('paramfile', directory)
        p = iu.module_from_spec(spec)
        sys.modules[descriptor] = p
        spec.loader.exec_module(p)
        return p
