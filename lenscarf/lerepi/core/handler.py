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
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2j_Transformer, l2T_Transformer
from lenscarf.lerepi.core.transformer.lerepi2status import l2j_Transformer as l2js_Transformer


class handler():
    """_summary_
    """
    def __init__(self, parser):
        """_summary_

        Args:
            parser (_type_): _description_
        """
        self.configfile = handler.load_configfile(parser.config_file, 'configfile')
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
        log.info("rank: {}, size: {}".format(mpi.rank, mpi.size))


    @log_on_start(logging.INFO, "collect_jobs() Started")
    @log_on_end(logging.INFO, "collect_jobs() Finished")
    def collect_jobs(self):
        """_summary_
        """        
        if self.parser.status == '':
            self.jobs = transform(self.configfile.dlensalot_model, l2j_Transformer())
        else:
            self.jobs = transform(self.configfile.dlensalot_model, l2js_Transformer())


    @log_on_start(logging.INFO, "get_jobs() Started")
    @log_on_end(logging.INFO, "get_jobs() Finished")
    def get_jobs(self):
        """_summary_

        Returns:
            _type_: _description_
        """        

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
        for jobdict in self.jobs:
            for job_id, val in jobdict.items():
                for transf, job in val:
                    log.info("Starting job {}".format(job_id))
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
        # _str = '---------------------------------------------------\n'
        # for key, val in p.__dict__.items():
        #     if key == 'dlensalot_model':
        #         _str += '{}:\t{}'.format(key, val)
        #         _str += '\n'
        #         _str += '---------------------------------------------------\n'
        # log.info(_str)

        return p


def purge(parser, TEMP):
    if parser.purgehashs:
        if mpi.rank == 0:
            def is_anadir(TEMP):
                if TEMP.startswith(os.environ['SCRATCH']):
                    return True
                else:
                    log.error('Not a $SCRATCH dir.')
                    sys.exit()

            def get_hashfiles(TEMP):
                counter = 0
                hashfiles = []
                for dirpath, dirnames, filenames in os.walk(TEMP):
                    _hshfile = [filename for filename in filenames if filename.endswith('hash.pk')]
                    counter += len(_hshfile)
                    if _hshfile != []:
                        hashfiles.append([dirpath, _hshfile])

                return hashfiles, counter

            if is_anadir(TEMP):
                log.info("====================================================")
                log.info("========        PURGING subroutine        ==========")
                log.info("====================================================")
                log.info("Will check {} for hash files: ".format(TEMP))
                hashfiles, counter = get_hashfiles(TEMP)
                if len(hashfiles)>0:
                    log.info("I find {} hash files,".format(counter))
                    log.info(hashfiles)
                    userinput = input('Please confirm purging with YES: ')
                    if userinput == "YES":
                        for pths in hashfiles:
                            for pth in pths[1]:
                                fn = opj(pths[0],pth)
                                os.remove(fn)
                                print("Deleted {}".format(fn))
                        print('All hashfiles have been deleted.')
                        hashfiles, counter = get_hashfiles(TEMP)
                        log.info("I find {} hash files".format(counter))  
                    else:
                        log.info("Not sure what that answer was.")
                else:
                    log.info("Cannot find any hash files.".format(counter))  

        log.info("====================================================")
        log.info("========        PURGING subroutine        ==========")
        log.info("====================================================")
        sys.exit() 
    
