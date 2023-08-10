#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Handles configuration file and makes sure analysis doesn't overwrite TEMP directory with altered configuration.
    Collects delensalot jobs defined in DLENSALOT_jobs.
    Extracts models needed for each delensalot job via x2y_Transformer.
    Runs all delensalot jobs.
"""


import os
from os.path import join as opj

import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import numpy as np

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI

from delensalot.config.validator import safelist
from delensalot.config.visitor import transform, transform3d
from delensalot.config.transformer.lerepi2dlensalot import l2T_Transformer, l2delensalotjob_Transformer

class abc:
    def __init__(self):
        pass

class config_handler():
    """Load config file and handle command line arguments 
    """

    def __init__(self, parser, config_model=None):
        sorted_joblist = ['generate_sim', 'QE_lensrec', 'MAP_lensrec', 'OBD_builder']
        if config_model is None:
            self.configfile = config_handler.load_configfile(parser.config_file, 'configfile')
        else:
            self.configfile = abc()
            self.configfile.dlensalot_model = config_model
        if 'job_id' in parser.__dict__:
            if parser.job_id is None:
                pass
            else:
                self.configfile.dlensalot_model.job.jobs = []
                if parser.job_id != "":
                    for sortedjob in sorted_joblist:
                        if sortedjob == parser.job_id:
                            self.configfile.dlensalot_model.job.jobs.append(sortedjob)
                            break
                        else:
                            self.configfile.dlensalot_model.job.jobs.append(sortedjob)
        TEMP = transform(self.configfile.dlensalot_model, l2T_Transformer())
        self.parser = parser
        self.TEMP = TEMP


    @log_on_start(logging.DEBUG, "collect_model() Started")
    @log_on_end(logging.DEBUG, "collect_model() Finished")
    def collect_model(self, djob_id=''):
        """

        Args:
            job_id (str, optional): _description_. Defaults to ''.
        """      
        if self.parser.status == '':
            if mpi.rank == 0:
                self.store(self.parser, self.configfile, self.TEMP)
        ## Making sure that specific job request from run() is processed
        self.configfile.dlensalot_model.job.jobs = [djob_id]
        self.djob_id = djob_id
        self.djobmodels = [transform3d(self.configfile.dlensalot_model, djob_id, l2delensalotjob_Transformer())]

        return self.djobmodels[0]


    @log_on_start(logging.DEBUG, "collect_models() Started")
    @log_on_end(logging.DEBUG, "collect_models() Finished")
    def collect_models(self, djob_id=''):
        """collect all requested jobs and build a mapping between the job, and their respective transformer
        This is called from both, terminal and interacitve runs.

        Args:
            job_id (str, optional): A specific job which should be performed. This one is not necessarily defined in the configuration file. It is handed over via command line or in interactive mode. Defaults to ''.
        """
        if self.parser.status == '':
            if mpi.rank == 0:
                self.store(self.parser, self.configfile, self.TEMP)
        self.djobmodels = []
        for job_id in self.configfile.dlensalot_model.job.jobs:
            self.djobmodels.append(transform3d(self.configfile.dlensalot_model, job_id, l2delensalotjob_Transformer()))
        return self.djobmodels
        

    @check_MPI
    @log_on_start(logging.INFO, "run() Started")
    @log_on_end(logging.INFO, "run() Finished")
    def run(self):
        """pass-through for running the delensalot job This esentially calls the run function of the `core.handler.<job_class>`. Used from interactive mode.

        Args:
            job_choice (list, optional): A specific job which should be performed. This one is not necessarily defined in the configuration file. It is handed over via command line or in interactive mode. Defaults to [].
        """ 
        for jobi, job in enumerate(self.djobmodels):
            log.info('running job {}'.format(self.configfile.dlensalot_model.job.jobs[jobi]))
            job.collect_jobs()    
            job.run()


    @log_on_start(logging.INFO, "store() Started")
    @log_on_end(logging.INFO, "store() Finished")
    def store(self, parser, configfile, TEMP):
        """ Store the dlensalot_model as config file in TEMP, to use if run resumed

        Args:
            parser (object): command line parser object
            configfile (str): the name of the configuration file
            TEMP (str): The location at which the configuration file (and all intermediate and final products of the analysis) will be stored
        """
        dostore = False
        # This is only done if not resuming. Otherwise file would already exist
        log.info(parser.config_file)
        if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
            # if the file already exists, check if something changed
            if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                logging.warning('config file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                configfile_old = config_handler.load_configfile(TEMP+'/'+parser.config_file.split('/')[-1], 'configfile_old')   
                for key, val in configfile_old.dlensalot_model.__dict__.items():
                    if hasattr(val, '__dict__'):
                        for k, v in val.__dict__.items():
                            if callable(v):
                                # skipping functions
                                pass
                            # FIXME if float, only check first digits for now.. this is presumably unsafe..
                            elif v.__str__()[:4] != configfile.dlensalot_model.__dict__[key].__dict__[k].__str__()[:4]:
                                logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                if k.__str__() in safelist:
                                    dostore = True
                                else:
                                    dostore = False
                                    logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                    logging.warning('Not part of safelist. Changing this value will likely result in a wrong analysis. Exit. Check config file.')
                                    sys.exit()
                    else:
                        ## Catching the infamous defaultstodictkey. Pass for now
                        pass
                logging.info('config file comparison done. No conflicts found.')
            else:
                dostore = True
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
                logging.info(TEMP+'/'+parser.config_file.split('/')[-1])


    @log_on_start(logging.INFO, "load_configfile() Started: {directory}")
    @log_on_end(logging.INFO, "load_configfile() Finished")
    def load_configfile(directory, descriptor):
        """Helper method for loading the configuration file.

        Args:
            directory (_type_): The directory to read from.
            descriptor (_type_): Identifier with which the configuration file is stored in memory.

        Returns:
            object: the configuration file
        """        
        
        spec = iu.spec_from_file_location('configfile', directory)
        p = iu.module_from_spec(spec)
        sys.modules[descriptor] = p
        spec.loader.exec_module(p)

        return p


    def purge_TEMPdir(self):
        TEMP = transform(self.configfile.dlensalot_model, l2T_Transformer())
        shutil.rmtree(TEMP, ignore_errors=True)
        log.info('Purged {}'.format(TEMP))


    def purge_TEMPconf(self):
        TEMP = transform(self.configfile.dlensalot_model, l2T_Transformer())
        TEMPconf = TEMP +'/'+self.parser.config_file.split('/')[-1]
        os.remove(TEMPconf)
        log.info('Purged {}'.format(TEMPconf))