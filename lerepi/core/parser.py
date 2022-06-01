import argparse
import os
from os import walk

import lerepi


class lerepi_parser():

    def __init__(self):
        __argparser = argparse.ArgumentParser(description='Lerepi main entry point')
        __argparser.add_argument('-p', dest='config_file', type=str, default='', help='Parameterfile which defines all variables needed for delensing')
        __argparser.add_argument('-r', dest='resume', type=str, default='', help='Abolsute path to parameter file to resume')
        self.parser = __argparser.parse_args()


    def validate(self):
        _f = []
        module_path = os.path.dirname(lerepi.__file__)
        for (dirpath, dirnames, filenames) in walk(module_path+'/config/'):
            _f.extend(filenames)
            break
        f = [s for s in _f if s.startswith('c_')]
        paramfile_path = module_path+'/config/'+self.parser.config_file
        if self.parser.config_file == '' and self.parser.resume == '':
            assert 0, 'ERROR: Must choose config file. I see the following options: {}'.format(f)
        elif os.path.exists(paramfile_path):
            print("Using {} file".format(self.parser.config_file))
            self.parser.config_file = paramfile_path
        elif self.parser.resume != '':
            paramfile_path = self.parser.resume
            if os.path.exists(paramfile_path):
                print("Using {} file".format(paramfile_path))
                self.parser.config_file = paramfile_path
            else:
                print('Cannot find parameter file to resume at {}'.format(paramfile_path))
        else:
            print("ERROR: Cannot find file {}".format(self.parser.config_file))
            assert 0, "I see the following options: {}".format(f)


    def get_parser(self):

        return self.parser


