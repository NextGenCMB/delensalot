#!/usr/bin/env python

"""parser.py: Read and validate user input from terminal
"""



import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import argparse
import os, sys
from os import walk

import delensalot.config as config


class lerepi_parser():
    def __init__(self):
        __argparser = argparse.ArgumentParser(description='delensalot entry point.')
        __argparser.add_argument('-r', dest='resume', type=str, default='', help='Absolute path to config file to resume.')
        __argparser.add_argument('-s', dest='status', type=str, default='', help='Absolute path for the analysis to write a report.')
        __argparser.add_argument('-job_id', dest='job_id', type=str, default=None, help='Execute job, overwrites config file')
        self.parser = __argparser.parse_args()

    @log_on_start(logging.INFO, "Start of validate()")
    @log_on_end(logging.INFO, "Finished validate()")
    def validate(self):

        def _validate_s(status_fn):
            if status_fn == '':
                pass
            elif status_fn.startswith('/'):
                if os.path.exists(status_fn):
                    log.info("Status report requested for {}".format(status_fn))
                    self.parser.config_file = status_fn
                    return True
                else:
                    log.error('Cannot find config file to resume at {}'.format(status_fn))
                    assert 0
                return True
            else:
                log.error("ERROR: status reports must use absolute path to config file. Your input was {}".format(status_fn))
                assert 0

        def _validate_r(resume_fn):
            if resume_fn != '':
                # if resume is asked, check path 
                if os.path.exists(resume_fn):
                    log.info("resuming previous run with {}".format(resume_fn))
                    self.parser.config_file = resume_fn
                    return True
                else:
                    log.error('Cannot find config file to resume at {}'.format(resume_fn))
                    assert 0, "I see the following options: {}".format(f)
                    
        def _validate_job(job_id):
            if job_id in ['QE_lensrec', 'MAP_lensrec', 'build_OBD', 'generate_sim', 'delens', 'analyse_phi', None]:
                return True
            else:
                assert 0, log.error("Job_id must be in {} but is {}".format(['QE_lensrec', 'MAP_lensrec', 'build_OBD', 'generate_sim', 'delens', 'analyse_phi', ], job_id))

        if self.parser.resume == '' and self.parser.status == '':
            print('Choose one of the available options to get going.')
            sys.exit()
        if _validate_s(self.parser.status):
            pass
        if _validate_r(self.parser.resume):
            pass
        if _validate_job(self.parser.job_id):
            pass

        return True


    @log_on_start(logging.INFO, "Start of get_parser()")
    @log_on_end(logging.INFO, "Finished get_parser()")
    def get_parser(self):

        return self.parser


