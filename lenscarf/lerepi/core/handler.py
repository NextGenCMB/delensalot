#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Collects lerepi jobs defined in DLENSALOT_jobs
    Extracts models needed for each job via x2y_Transformer
    runs all jobs
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"
# TODO this could be the level for _process_Model

import os
import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from plancklens.helpers import mpi

from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2j_Transformer, l2T_Transformer#, transform
from lenscarf.lerepi.core.transformer.lerepi2status import l2j_Transformer as l2js_Transformer#, transform as transform_status


class handler():

    def __init__(self, parser):
        self.configfile = handler.load_configfile(parser.config_file, 'configfile')
        TEMP = transform(self.configfile.dlensalot_model, l2T_Transformer())
        if parser.status == '':
            if mpi.rank == 0:
                self.store(parser, self.configfile, TEMP)
        self.parser = parser


    @log_on_start(logging.INFO, "collect_jobs() Started")
    @log_on_end(logging.INFO, "collect_jobs() Finished")
    def collect_jobs(self):
        if self.parser.status == '':
            self.jobs = transform(self.configfile.dlensalot_model, l2j_Transformer())
        else:
            self.jobs = transform(self.configfile.dlensalot_model, l2js_Transformer())


    @log_on_start(logging.INFO, "get_jobs() Started")
    @log_on_end(logging.INFO, "get_jobs() Finished")
    def get_jobs(self):

        return self.jobs


    @log_on_start(logging.INFO, "collect_jobs() Started")
    @log_on_end(logging.INFO, "init_job() Finished")
    def init_job(self, job):
        log.info('transform started')
        model = transform(*job[0])
        log.info('transform done')
        log.info('model init started')
        j = job[1](model)
        log.info('model init done')
        log.info('collect_jobs started')
        j.collect_jobs()
        log.info('collect_jobs done')

        return j


    @log_on_start(logging.INFO, "run() Started")
    @log_on_end(logging.INFO, "run() Finished")
    def run(self):
        for transf, job in self.jobs:
            log.info("Starting job {}".format(job))
            model = transform(*transf)
            log.info("Model collected {}".format(model))
            j = job(model)
            j.collect_jobs()
            j.run()


    @log_on_start(logging.INFO, "store() Started")
    @log_on_end(logging.INFO, "store() Finished")
    def store(self, parser, configfile, TEMP):
        """ Store the dlensalot_model as config file in TEMP, to use if run resumed

        Args:
            parser (_type_): _description_
            configfile (_type_): _description_
            TEMP (_type_): _description_
        """
        dostore = False
        if parser.resume == '':
            dostore = True
            # This is only done if not resuming. Otherwise file would already exist
            if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
                # if the file already exists, check if something changed
                if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                    dostore = False
                    logging.warning('config file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                    configfile_old = handler.load_configfile(TEMP+'/'+parser.config_file.split('/')[-1], 'configfile_old')   
                    for key, val in configfile_old.dlensalot_model.__dict__.items():
                        for k, v in val.__dict__.items():
                            if v.__str__() != configfile.dlensalot_model.__dict__[key].__dict__[k].__str__():
                                if callable(v):
                                    # If it's a function, we can test if bytecode is the same as a simple check won't work due to pointing to memory location
                                    if v.__code__.co_code != configfile.dlensalot_model.__dict__[key].__dict__[k].__code__.co_code:
                                        logging.error("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                        logging.error('Exit. Check config file.')
                                        sys.exit()
                    logging.info('config file look the same. Resuming where I left off last time.')

        if dostore:
            if mpi.rank == 0:
                if not os.path.exists(TEMP):
                    os.makedirs(TEMP)
                shutil.copyfile(parser.config_file, TEMP +'/'+parser.config_file.split('/')[-1])
            logging.info('config file stored at '+ TEMP +'/'+parser.config_file.split('/')[-1])
        else:
            if parser.resume == '':
                # Only give this info when not resuming
                logging.info('Matching config file found. Resuming where I left off.')


    @log_on_start(logging.INFO, "load_configfile() Started")
    @log_on_end(logging.INFO, "load_configfile() Finished")
    def load_configfile(directory, descriptor):
        """Load config file

        Args:
            directory (_type_): _description_
            descriptor (_type_): _description_

        Returns:
            _type_: _description_
        """
        spec = iu.spec_from_file_location('configfile', directory)
        p = iu.module_from_spec(spec)
        sys.modules[descriptor] = p
        spec.loader.exec_module(p)

        # printing this for the slurm log file
        _str = '---------------------------------------------------\n'
        for key, val in p.__dict__.items():
            if key == 'dlensalot_model':
                _str += '{}:\t{}'.format(key, val)
                _str += '\n'
                _str += '---------------------------------------------------\n'
        log.info(_str)

        return p
