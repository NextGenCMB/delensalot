#!/usr/bin/env python

"""run.py: Entry point for running delensalot
"""

import os, sys
import logging
import traceback

import delensalot.core.mpi as mpi

from delensalot.config.config_handler import ConfigHandler
from delensalot.config.etc.abstract import parserclass
from delensalot.config.parser import LerepiParser
from delensalot.config.etc import logger

from delensalot.core.MAP.context import get_computation_context

datefmt = "%m-%d %H:%M:%S"
FORMAT = '%(levelname)s:: %(asctime)s:: %(name)s.%(funcName)s - %(message)s'
formatter = logging.Formatter(FORMAT, datefmt=datefmt)

ConsoleOutputHandler = logging.StreamHandler(sys.stdout)
ConsoleOutputHandler.setFormatter(formatter)

# Use root logger instead of a fixed "global_logger"
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    root_logger.addHandler(ConsoleOutputHandler)
    root_logger.setLevel(logging.INFO)  # Default level

def set_logging_level(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO

    # Ensure both the logger and the handler get updated
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)  # Ensure all handlers allow the logs

np_logger = logging.getLogger("numpy")
np_logger.setLevel(logging.WARNING)
np_logger = logging.getLogger("matplotlib")
np_logger.setLevel(logging.WARNING)
logging.getLogger("healpy").setLevel(logging.WARNING)
np_logger.setLevel(logging.WARNING)

class run():
    """Entry point for the interactive mode
    """
    def __init__(self, config_fn=None, job_id='MAP_lensrec', config=None, verbose=True, key=None):
        """Entry point for the interactive mode. This initializes a 'runner'-objects which provides all functionalities to run delensalot analyses

        Args:
            config_fn (str): The config file for the analysis.
            job_id (str, optional): Identifier to choose the delensalot job. Valid values are: ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'delens']. Defaults to 'generate_sim'.
            config_model (DLENSALOT_Model): A delensalot model instance. If not None, this overwrites `config_fn`
            verbose (bool, optional): If true, sets logging information to DEBUG, otherwise INFO. Defaults to True.
        """    
        assert config_fn is not None or config is not None, "Either config_fn or config must be provided"    
        assert job_id in ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'delens'], "Invalid job_id: {}".format(job_id)
        if key is not None and config_fn is None:
            config.analysis.key = key
        os.environ['USE_PLANCKLENS_MPI'] = "False"
        set_logging_level(verbose=verbose)  # Set True for debug mode
        
        self.parser = parserclass()
        self.parser.resume =  ""
        self.parser.config_file = config_fn if config_fn is not None else 'config.py'
        self.parser.status = ''
        self.parser.job_id = job_id

        self.delensalotjob = job_id
        self.config_handler = ConfigHandler(self.parser, config, key)

        ctx, isnew = get_computation_context()
        if not isnew:
            ctx.reset()


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
        # FIXME need to fix this
        assert 0, "Please use the run() function instead. You should not be using collect_models if data not yet generated. QE Model will be already built although DataContainer info still changes, leading to wrong QE"
        if mpi.size > 1:
            if mpi.rank == 0:
                mpi.disable()
                self.config_handler.collect_models()
                mpi.enable()
                [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)

        return self.config_handler.collect_models()


    def run(self):
        self.config_handler.run()

        return self.config_handler.djobmodels


    def init_job(self):
        return self.collect_model()
    

if __name__ == '__main__':
    """Entry point from the command line
    """
    os.environ['USE_PLANCKLENS_MPI'] = "False"
    lparser = LerepiParser()
    if lparser.validate():
        parser = lparser.get_parser()

    config_handler = ConfigHandler(parser)
    if mpi.rank == 0:
        mpi.disable()
        config_handler.collect_models()
        mpi.enable()
        [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
    else:
        mpi.receive(None, source=mpi.ANY_SOURCE)
    if mpi.size > 1:
        config_handler.collect_models()

    ctx, isnew = get_computation_context()
    if not isnew:
        ctx.reset()

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
