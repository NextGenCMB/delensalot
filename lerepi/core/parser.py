import argparse
import os
from os import walk
import sys

import lerepi


class lerepi_parser():

    def __init__(self):
        __argparser = argparse.ArgumentParser(description='Lerepi main entry point')
        __argparser.add_argument('-p', dest='config_file', type=str, default='', help='Parameterfile which defines all variables needed for delensing')
        __argparser.add_argument('-sc', dest='survey_config', type=str, default='', help='Survey configuration')
        __argparser.add_argument('-lc', dest='lensing_config', type=str, default='', help='Lensing configuration')
        __argparser.add_argument('-rc', dest='run_config', type=str, default='', help='Run configuration')

        self.parser = __argparser.parse_args()


    def validate(self):
        _f = []
        module_path = os.path.dirname(lerepi.__file__)
        for (dirpath, dirnames, filenames) in walk(module_path+'/config/'):
            _f.extend(filenames)
            break
        f = [s for s in _f if s.startswith('c_')]
        paramfile_path = module_path+'/config/'+self.parser.config_file
        if self.parser.config_file == '':
            assert 0, 'ERROR: Must choose config file. I see the following options: {}'.format(f)
        elif os.path.exists(paramfile_path):
            print("Using {} file".format(self.parser.config_file))
        else:
            print("ERROR: Cannot find file {}".format(self.parser.config_file))
            assert 0, "I see the following options: {}".format(f)

        # TODO probably don't need this, a single config file is better than three
        self.parser.survey_config = self.parser.survey_config
        self.parser.lensing_config = self.parser.lensing_config
        self.parser.survey_config = self.parser.run_config

    def get_parser(self):
        return self.parser


