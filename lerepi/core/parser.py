#!/usr/bin/env python

"""parser.py: Read and validate terminal user input and store param file to TEMP directory
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import logging
from logdecorator import log_on_start, log_on_end

import argparse
import os
from os import walk

import lerepi

# TODO currently only supports config file. Could think of adding DLENSALOT_Job parameters
class lerepi_parser():

    def __init__(self):
        __argparser = argparse.ArgumentParser(description='Lerepi main entry point')
        __argparser.add_argument('-p', dest='config_file', type=str, default='cmbs4/c08d.py', help='Parameterfile which defines all variables needed for delensing')
        __argparser.add_argument('-r', dest='resume', type=str, default='', help='Abolsute path to parameter file to resume')
        self.parser = __argparser.parse_args()

    @log_on_start(logging.INFO, "Start of validate()")
    @log_on_end(logging.INFO, "Finished validate()")
    def validate(self):
        _f = []
        module_path = os.path.dirname(lerepi.__file__)
        for (dirpath, dirnames, filenames) in walk(module_path+'/config/'):
            _f.extend(filenames)
            break
        f = [s for s in _f if s.startswith('c_')]
        paramfile_path = module_path+'/config/'+self.parser.config_file

        if self.parser.config_file == '' and self.parser.resume == '':
            # if -p  and -r empty, abort
            assert 0, 'ERROR: Must choose config file. I see the following options: {}'.format(f)
        elif self.parser.config_file == '' and self.parser.resume != '':
            # if resume is asked, check path
            paramfile_path = self.parser.resume
            if os.path.exists(paramfile_path):
                print("resuming previous run with {}".format(paramfile_path) )
                self.parser.config_file = paramfile_path
            else:
                print('Cannot find parameter file to resume at {}'.format(paramfile_path))
        elif os.path.exists(paramfile_path):
            # if new run is asked, check path
            print("New run requested with with {}".format(paramfile_path))
            self.parser.config_file = paramfile_path
        else:
            print("ERROR: Cannot find file {}".format(paramfile_path))
            assert 0, "I see the following options: {}".format(f)


    @log_on_start(logging.INFO, "Start of get_parser()")
    @log_on_end(logging.INFO, "Finished get_parser()")
    def get_parser(self):

        return self.parser


