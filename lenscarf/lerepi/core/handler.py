#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Collects lerepi jobs defined in DLENSALOT_jobs
    Extracts models needed for each job via x2y_Transformer
    runs all jobs
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"
# TODO this could be the level for _process_Model

import os
from os.path import join as opj

import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from lenscarf.core import mpi

from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2j_Transformer, l2T_Transformer, l2ji_Transformer
from lenscarf.lerepi.core.transformer.lerepi2status import l2j_Transformer as l2js_Transformer


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


    @log_on_start(logging.INFO, "check_mpi() Started")
    @log_on_end(logging.INFO, "check_mpi() Finished")
    def check_mpi(self):
        """_summary_
        """        
        log.info("rank: {}, size: {}, name: {}".format(mpi.rank, mpi.size, mpi.name))


    @log_on_start(logging.INFO, "collect_jobs() Started")
    @log_on_end(logging.INFO, "collect_jobs() Finished")
    def collect_jobs(self, job_id=''):
        """_summary_
        """
        ## Making sure that specific job request from run() is processed
        print(self.configfile.dlensalot_model.job.jobs, job_id)
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


    @log_on_start(logging.INFO, "get_jobs() Started")
    @log_on_end(logging.INFO, "get_jobs() Finished")
    def get_jobs(self):
        """_summary_

        Returns:
            _type_: _description_
        """        

        return self.jobs


    @log_on_start(logging.INFO, "init_job() Started")
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
        for jobdict in self.jobs:
            for job_id, val in jobdict.items():
                conf = val[0][0]
                transformer = val[0][1]
                job = val[1]

                log.info("Starting job {}".format(job_id))

                model = transform(conf, transformer)
                # log.info("Model collected {}".format(model))
                self.check_mpi()
                j = job(model)
                self.check_mpi()
                j.collect_jobs()
                self.check_mpi()
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
        safelist = [
            'version',
            'jobs',
            'simidxs',
            'simidxs_mf',
            'tasks',
            'cl_analysis',
            'blt_pert',
            'itmax',
            'cg_tol',
            'mfvar',
            'dlm_mod',
            'spectrum_calculator',
            'binning',
            'outdir_plot_root',
            'outdir_plot_rel',
            'OMP_NUM_THREADS',
        ]
        dostore = False
        # This is only done if not resuming. Otherwise file would already exist
        if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
            # if the file already exists, check if something changed
            if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                logging.warning('config file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                configfile_old = handler.load_configfile(TEMP+'/'+parser.config_file.split('/')[-1], 'configfile_old')   
                for key, val in configfile_old.dlensalot_model.__dict__.items():
                    for k, v in val.__dict__.items():
                        if v.__str__() != configfile.dlensalot_model.__dict__[key].__dict__[k].__str__():
                            log.info('Different item found')
                            if k.__str__() in safelist:
                                logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                dostore = True
                            else:
                                # if callable(v):
                                #     # If function, we can test if bytecode is the same as a simple check won't work due to pointing to memory location
                                #     if v.__code__.co_code != configfile.dlensalot_model.__dict__[key].__dict__[k].__code__.co_code:
                                #         logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                #         logging.warning('Exit. Check config file.')
                                #         sys.exit()
                                # else:
                                    dostore = False
                                    logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, configfile.dlensalot_model.__dict__[key].__dict__[k]))
                                    logging.warning('Not part of safelist. Changing this value will likely result in a wrong analysis. Exit. Check config file.')
                                    sys.exit()
                logging.info('config file comparison done. No conflicts found.')

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
