#!/usr/bin/env python

"""parser.py: Read and validate user input from terminal
"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import argparse
import os, sys

class LerepiParser():
    def __init__(self):
        __argparser = argparse.ArgumentParser(description='delensalot entry point.')
        __argparser.add_argument('-r', dest='resume', type=str, default='', help='Absolute path to config file to resume.')
        __argparser.add_argument('-job_id', dest='job_id', type=str, default=None, help='Execute job, overwrites config file')
        self.parser = __argparser.parse_args()


    @log_on_start(logging.INFO, "Start of validate()", logger=log)
    @log_on_end(logging.INFO, "Finished validate()", logger=log)
    def validate(self):
        def _validate_r(resume_fn):
            if resume_fn != '':
                # if resume is asked, check path 
                if os.path.exists(resume_fn):
                    log.info("resuming previous run with {}".format(resume_fn))
                    self.parser.config_file = resume_fn
                    return True
                else:
                    log.error('Cannot find config file to resume at {}'.format(resume_fn))
                    assert 0, "Cannot find config file to resume at {}".format(resume_fn)
                    
        def _validate_job(job_id):
            if job_id in ['QE_lensrec', 'MAP_lensrec', 'build_OBD', 'generate_sim', 'delens', 'analyse_phi', None]:
                return True
            else:
                assert 0, log.error("Job_id must be in {} but is {}".format(['QE_lensrec', 'MAP_lensrec', 'build_OBD', 'generate_sim', 'delens', 'analyse_phi', ], job_id))

        if self.parser.resume == '':
            assert 0, 'Choose one of the available options to get going.'
        if _validate_r(self.parser.resume):
            pass
        if _validate_job(self.parser.job_id):
            pass
        return True

    def get_parser(self):
        return self.parser