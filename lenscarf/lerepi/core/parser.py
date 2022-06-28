#!/usr/bin/env python

"""parser.py: Read and validate terminal user input and store param file to TEMP directory
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import argparse
import os
from os import walk

import lenscarf.lerepi as lerepi

# TODO Add DLENSALOT_Job configs
class lerepi_parser():

    def __init__(self):
        __argparser = argparse.ArgumentParser(description='D.lensalot entry point.')
        __argparser.add_argument('-p', dest='new', type=str, default='', help='Relative path to config file to run analysis.')
        __argparser.add_argument('-r', dest='resume', type=str, default='', help='Absolute path to config file to resume.')
        __argparser.add_argument('-s', dest='status', type=str, default='', help='Absolute path for the analysis to write a report.')
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

        def _validate_p(new_fn):
            if new_fn != '':
                _f = []
                module_path = os.path.dirname(lerepi.__file__)
                log.info(module_path)
                for (dirpath, dirnames, filenames) in walk(module_path+'/config/'):
                    _f.extend(filenames)
                    break
                f = [s for s in _f if s.startswith('c_')]
                paramfile_path = module_path+'/config/'+new_fn
                log.info("User config file: {}".format(paramfile_path))
                if os.path.exists(paramfile_path):
                    # if new run is asked, check path
                    log.info("New run requested with with {}".format(paramfile_path))
                    self.parser.config_file = paramfile_path
                    return True
                else:
                    log.error("ERROR: Cannot find file {}".format(paramfile_path))
                    assert 0, "I see the following options: {}".format(f)

        if self.parser.new == '' and self.parser.resume == '' and self.parser.status == '':
            assert 0, 'Choose one of the available options to get going.'
        if _validate_s(self.parser.status):
            pass
        if _validate_r(self.parser.resume):
            pass
        if _validate_p(self.parser.new):
            pass

        return True


    @log_on_start(logging.INFO, "Start of get_parser()")
    @log_on_end(logging.INFO, "Finished get_parser()")
    def get_parser(self):

        return self.parser


