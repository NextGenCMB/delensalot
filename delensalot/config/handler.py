#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Collects lerepi jobs defined in DLENSALOT_jobs
    Extracts models needed for each job via x2y_Transformer
    runs all jobs
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

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI
from delensalot.config.validator import safelist
from delensalot.config.visitor import transform
from delensalot.config.transformer.lerepi2dlensalot import l2j_Transformer, l2T_Transformer, l2ji_Transformer
from delensalot.config.transformer.lerepi2status import l2j_Transformer as l2js_Transformer


class handler():
    """_summary_
    """
    def __init__(self, parser, madel_kwargs={}):
        """_summary_

        Args:
            parser (_type_): _description_
        """
        self.configfile = handler.load_configfile(parser.config_file, 'configfile')
        # TODO hack. remove when v1 is gone
        if 'madel' in self.configfile.dlensalot_model.__dict__:
            self.configfile.dlensalot_model.madel.__dict__.update(madel_kwargs)
        if 'job_id' in parser.__dict__:
            if parser.job_id is None:
                pass
            else:
                self.configfile.dlensalot_model.job.__dict__.update({'jobs': [parser.job_id]})
        # TODO catch here TEMP dir for build_OBD? or make TEMP builder output buildOBDspecific
        TEMP = transform(self.configfile.dlensalot_model, l2T_Transformer())
        if parser.status == '':
            if mpi.rank == 0:
                self.store(parser, self.configfile, TEMP)
        self.parser = parser
        self.TEMP = TEMP


    @log_on_start(logging.DEBUG, "collect_jobs() Started")
    @log_on_end(logging.DEBUG, "collect_jobs() Finished")
    def collect_job(self, job_id=''):
        """_summary_
        """
        ## Making sure that specific job request from run() is processed
        self.configfile.dlensalot_model.job.jobs = [job_id]
        self.job_id = job_id
        
        if self.parser.status == '':
            self.jobs = transform(self.configfile.dlensalot_model, l2j_Transformer())
        else:
            if mpi.rank == 0:
                self.jobs = transform(self.configfile.dlensalot_model, l2js_Transformer())
            else:
                self.jobs = []


    @log_on_start(logging.DEBUG, "collect_jobs() Started")
    @log_on_end(logging.DEBUG, "collect_jobs() Finished")
    def collect_jobs(self, job_id=''):
        """_summary_
        """
        ## Making sure that specific job request from run() is processed
        self.configfile.dlensalot_model.job.jobs.append(job_id)
        self.job_id = job_id
        
        if self.parser.status == '':
            self.jobs = transform(self.configfile.dlensalot_model, l2j_Transformer())
        else:
            if mpi.rank == 0:
                self.jobs = transform(self.configfile.dlensalot_model, l2js_Transformer())
            else:
                self.jobs = []


    @log_on_start(logging.INFO, "make_interactive_job() Started")
    @log_on_end(logging.INFO, "make_interactive_job() Finished")
    def make_interactive_job(self):
        self.jobs = transform(self.configfile.dlensalot_model, l2ji_Transformer())


    @log_on_start(logging.INFO, "build_model() Started")
    @log_on_end(logging.INFO, "build_model() Finished")
    def build_model(self, job):
        return transform(*job[0])


    @log_on_start(logging.INFO, "init_job() Started")
    @log_on_end(logging.INFO, "init_job() Finished")
    def init_job(self, job, model):
        j = job[1](model)
        # j.collect_jobs()

        return j


    @check_MPI
    @log_on_start(logging.INFO, "run() Started")
    @log_on_end(logging.INFO, "run() Finished")
    def run(self, job_choice=[]):
        
        if job_choice == []:
            for jobdict in self.jobs:
                for job_id, val in jobdict.items():
                    job_choice.append(job_id)
    
        for jobdict in self.jobs:
            for job_id, val in jobdict.items():
                if job_id in job_choice:
                    conf = val[0][0]
                    transformer = val[0][1]
                    job = val[1]
                    
                    model = transform(conf, transformer)

                    if mpi.rank == 0:
                        mpi.disable()
                        delensalot_job = job(model)
                        mpi.enable()
                        [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
                    else:
                        mpi.receive(None, source=mpi.ANY_SOURCE)
                    delensalot_job = job(model)
                    delensalot_job.collect_jobs()    
                    delensalot_job.run()
                

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
        # This is only done if not resuming. Otherwise file would already exist
        if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
            # if the file already exists, check if something changed
            if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                logging.warning('config file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                configfile_old = handler.load_configfile(TEMP+'/'+parser.config_file.split('/')[-1], 'configfile_old')   
                for key, val in configfile_old.dlensalot_model.__dict__.items():
                    if hasattr(val, '__dict__'):
                        for k, v in val.__dict__.items():
                            if v.__str__() != configfile.dlensalot_model.__dict__[key].__dict__[k].__str__():
                                logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                if k.__str__() in safelist:
                                    dostore = True
                                    # if callable(v):
                                    #     # If function, we can test if bytecode is the same as a simple check won't work due to pointing to memory location
                                    #     if v.__code__.co_code != configfile.dlensalot_model.__dict__[key].__dict__[k].__code__.co_code:
                                    #         logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                    #         logging.warning('Exit. Check config file.')
                                    #         sys.exit()
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


    @log_on_start(logging.INFO, "load_configfile() Started: {directory}")
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

        return p
