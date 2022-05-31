import argparse

class lerepi_parser():

    def validate(self):
        if self.parser.config == '':
            assert 0, 'Must choose config file'
        self.parser.survey_config = self.parser.config
        self.parser.lensing_config = self.parser.lensing_config
        self.parser.survey_config = self.parser.run_config


    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Lerepi main entry point')
        self.parser.add_argument('-dset', dest='dset', type=str, default='', help='Dataset to delense')
        self.parser.add_argument('-conf', dest='config', type=str, default='', help='Configuration')
        self.parser.add_argument('-sc', dest='survey_config', type=str, default='', help='Survey configuration')
        self.parser.add_argument('-lc', dest='lensing_config', type=str, default='', help='Lensing configuration')
        self.parser.add_argument('-rc', dest='run_config', type=str, default='', help='Run configuration')

        return self.parser.parse_args()