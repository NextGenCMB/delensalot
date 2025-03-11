#!/usr/bin/env python

"""config_handler.py: handler for lensing reconstruction pipelines.
    Handles configuration files and TEMP directory.
    Collects delensalot jobs defined in DLENSALOT_jobs.
    Extracts models needed for each delensalot job via x2y_Transformer.
    Runs all delensalot jobs.
"""
import os, sys
import importlib.util as iu
import shutil

import logging
log = logging.getLogger(__name__)

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI

from delensalot.config import safelist
from delensalot.config.visitor import transform, transform3d
from delensalot.config.transformer.lerepi2dlensalot import l2delensalotjob_Transformer, get_TEMP_dir


def load_config(directory, descriptor):
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

    return p.delensalot_model

class ConfigHandler():
    """Load config file and handle command line arguments 
    """

    def __init__(self, parser, config=None, key=None):
        sorted_joblist = ['generate_sim', 'QE_lensrec', 'MAP_lensrec', 'analyse_phi', 'delens']
        self.config = config if config is not None else load_config(parser.config_file, 'configfile')
        if key is not None:
            self.config.analysis.key = key
        if 'job_id' in parser.__dict__ and parser.job_id is not None:
            self.config.job.jobs = []
            for job in sorted_joblist:
                if job == parser.job_id:
                    if job == 'analyse_phi' and self.config.simulationdata.flavour != 'unl':
                        log.error(f"Job {parser.job_id} cannot be run because input phi doesn't exist. Exiting.")
                        sys.exit()
                    self.config.job.jobs.append(job)
                    break
                elif job == 'build_OBD' and self.config.noisemodel.OBD == 'OBD':
                    self.config.job.jobs.append(job)
                else:
                    self.config.job.jobs.append(job)
        TEMP = get_TEMP_dir(self.config)
        self.parser = parser
        self.TEMP = TEMP

    # @log_on_start(logging.DEBUG, "collect_model() Started")
    # @log_on_end(logging.DEBUG, "collect_model() Finished")
    def collect_model(self, djob_id=''):
        """
        Args:
            job_id (str, optional): _description_. Defaults to ''.
        """      
        if mpi.rank == 0:
            self.store(self.parser, self.config, self.TEMP)
        
        self.config.job.jobs = [djob_id] ## NOTE killing all other jobs on purpose here.
        self.djob_id = djob_id
        self.djobmodels = [transform3d(self.config, djob_id, l2delensalotjob_Transformer())]

        return self.djobmodels[0]


    # @log_on_start(logging.DEBUG, "collect_models() Started")
    # @log_on_end(logging.DEBUG, "collect_models() Finished")
    def collect_models(self, djob_id=''):
        """collect all requested jobs and build a mapping between the job, and their respective transformer
        This is called from both, terminal and interacitve runs.

        Args:
            job_id (str, optional): A specific job which should be performed. This one is not necessarily defined in the configuration file. It is handed over via command line or in interactive mode. Defaults to ''.
        """
        if mpi.rank == 0:
            self.store(self.parser, self.config, self.TEMP)
        self.djobmodels = []
        for job_id in self.config.job.jobs:
            self.djobmodels.append(transform3d(self.config, job_id, l2delensalotjob_Transformer()))
        return self.djobmodels
        

    @check_MPI
    def run(self):
        """pass-through for running the delensalot job This esentially calls the run function of the `core.handler.<job_class>`. Used from interactive mode.

        Args:
            job_choice (list, optional): A specific job which should be performed. This one is not necessarily defined in the configuration file. It is handed over via command line or in interactive mode. Defaults to [].
        """ 
        if mpi.rank == 0:
            self.store(self.parser, self.config, self.TEMP)
        self.djobmodels = []
        for jobi, job_id in enumerate(self.config.job.jobs):
            djob = transform3d(self.config, job_id, l2delensalotjob_Transformer())
            log.info('running job {}'.format(self.config.job.jobs[jobi]))
            djob.collect_jobs()
            djob.run()
        return djob


    def store(self, parser, config, TEMP):
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
            if config.__dict__['validate_model'] == True:
                # If validation skipped, simply overwrite existing config file
                if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                    # if the file already exists, check if something changed
                    logging.warning('config file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                    config_prev = load_config(TEMP+'/'+parser.config_file.split('/')[-1], 'config_prev')   
                    for key, val in config_prev.__dict__.items():
                        if hasattr(val, '__dict__'):
                            for k, v in val.__dict__.items():
                                if callable(v):
                                    # skipping functions
                                    pass
                                # FIXME if float, only check first digits for now.. this is presumably unsafe..
                                elif v.__str__()[:4] != config.__dict__[key].__dict__[k].__str__()[:4]:
                                    logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, config.__dict__[key].__dict__[k]))
                                    if k.__str__() in safelist:
                                        dostore = True
                                    else:
                                        dostore = False
                                        logging.warning("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, config.__dict__[key].__dict__[k]))
                                        logging.warning('Not part of safelist. Changing this value will likely result in a wrong analysis. Exit. Check config file.')
                                        sys.exit()
                        else:
                            ## Catching the infamous defaultstodictkey. Pass for now
                            pass
                    logging.info('config file comparison done. No conflicts found.')
                else:
                    dostore = True
            else:
                dostore = True
        if dostore:
            if not os.path.exists(TEMP):
                os.makedirs(TEMP)
            try:
                shutil.copyfile(parser.config_file, TEMP +'/'+parser.config_file.split('/')[-1])
                logging.info('config file stored at '+ TEMP +'/'+parser.config_file.split('/')[-1])
            except shutil.SameFileError:
                log.debug("Did not copy config file as it appears to be the same.")
        else:
            if parser.resume == '':
                # Only give this info when not resuming
                logging.info('Matching config file found. Resuming where I left off.')
                logging.info(TEMP+'/'+parser.config_file.split('/')[-1])