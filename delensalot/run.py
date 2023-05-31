#!/usr/bin/env python

"""run.py: Entry point for running delensalot
"""


import os, sys
import logging
import traceback

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI

from delensalot.config.handler import config_handler
import delensalot.config.etc.dev_helper as dh
from delensalot.config.etc.abstract import parserclass
from delensalot.config.parser import lerepi_parser


datefmt = "%m-%d %H:%M:%S"
FORMAT = '%(levelname)s:: %(asctime)s:: %(name)s.%(funcName)s - %(message)s'
formatter = logging.Formatter(FORMAT, datefmt=datefmt)
ConsoleOutputHandler = logging.StreamHandler(sys.stdout)
ConsoleOutputHandler.setFormatter(formatter)
ConsoleOutputHandler.setLevel(logging.INFO)

sys_logger = logging.getLogger(__name__)
sys_logger.addHandler(ConsoleOutputHandler)
sys_logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[ConsoleOutputHandler])
logging.getLogger("healpy").disabled = True
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class run():
    """Entry point for the interactive mode
    """
    def __init__(self, config_fn, job_id='generate_sim', config_model=None, verbose=True):
        """Entry point for the interactive mode. This initializes a 'runner'-objects which provides all functionalities to run delensalot analyses

        Args:
            config_fn (str): The config file for the analysis.
            job_id (str, optional): Identifier to choose the delensalot job. Valid values are: ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'delens']. Defaults to 'generate_sim'.
            config_model (DLENSALOT_Model): A delensalot model instance. If not None, this overwrites `config_fn`
            verbose (bool, optional): If true, sets logging information to DEBUG, otherwise INFO. Defaults to True.
        """        
        os.environ['USE_PLANCKLENS_MPI'] = "False"
        if not verbose:
            ConsoleOutputHandler.setLevel(logging.WARNING)
            sys_logger.setLevel(logging.WARNING)
            logging.basicConfig(level=logging.WARNING, handlers=[ConsoleOutputHandler])
        else:
            ConsoleOutputHandler.setLevel(logging.INFO)
            sys_logger.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, handlers=[ConsoleOutputHandler])
        self.parser = parserclass()
        self.parser.resume =  ""
        self.parser.config_file = config_fn
        self.parser.status = ''

        self.delensalotjob = job_id
        self.config_handler = config_handler(self.parser, config_model)


    def collect_model(self):
        if mpi.size > 1:
            if mpi.rank == 0:
                mpi.disable()
                self.config_handler.collect_model(self.delensalotjob)
                mpi.enable()
                [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)

        return self.config_handler.collect_model(self.delensalotjob)
    

    def collect_models(self):
        if mpi.size > 1:
            if mpi.rank == 0:
                mpi.disable()
                self.config_handler.collect_models(self.delensalotjob)
                mpi.enable()
                [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)

        return self.config_handler.collect_models(self.delensalotjob)


    def run(self):

        return self.collect_model()


    def init_job(self):


        self.config_handler.run(self.delensalotjob)

        return self.config_handler.delensalotjobs[0]



if __name__ == '__main__':
    """Entry point for the command line
    """
    lparser = lerepi_parser()
    if lparser.validate():
        parser = lparser.get_parser()

    config_handler = config_handler(parser)
    if dh.dev_subr in parser.__dict__:
        dh.dev(parser, config_handler.TEMP)
        sys.exit()
    config_handler.collect_jobs()

    try:
        config_handler.run()
    except Exception as err:
        # expection formatter. Don't want all these logdecorator functions in the trace.
        _msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        msg = ''
        skip = 0
        for line in _msg.splitlines():
            if skip > 0:
                skip -=1
            else:
                # Each decorator call comes with three lines of trace, and there are about 4 decorators for each exception..
                if 'logdecorator' in line:
                    skip = 3
                else:
                    msg += line + '\n'
        print(msg)
