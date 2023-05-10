#!/usr/bin/env python

"""run.py: Entry point for running D.lensalot
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
import logging
import traceback

from delensalot.lerepi.core import handler
import delensalot.lerepi.etc.dev_helper as dh
from delensalot.lerepi.etc.abstract import parserclass
from delensalot.lerepi.core.parser import lerepi_parser


datefmt = "%m-%d %H:%M"
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


class run():
    def __init__(self, config, job_id='interactive', verbose=True, madel_kwargs={}):
        if not verbose:
            ConsoleOutputHandler.setLevel(logging.WARNING)
            sys_logger.setLevel(logging.WARNING)
            logging.basicConfig(level=logging.WARNING, handlers=[ConsoleOutputHandler])
        else:
            ConsoleOutputHandler.setLevel(logging.INFO)
            sys_logger.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, handlers=[ConsoleOutputHandler])
        parser = parserclass()
        parser.resume =  ""
        parser.config_file = config
        parser.status = ''

        lerepi_handler = handler.handler(parser, madel_kwargs)
        if job_id == 'interactive':
            ## This is notebook interactive job. (Doesn't appear in config file job list)
            lerepi_handler.make_interactive_job()
            job = lerepi_handler.get_jobs()[0]
            self.job = lerepi_handler.init_job(job[job_id])
        else:
            lerepi_handler.collect_jobs(job_id)
            jobs = lerepi_handler.get_jobs()
            for jobdict in jobs:
                if job_id in jobdict:
                    self.job = lerepi_handler.init_job(jobdict[job_id])


if __name__ == '__main__':
    lparser = lerepi_parser()
    if lparser.validate():
        parser = lparser.get_parser()

    lerepi_handler = handler.handler(parser)
    if dh.dev_subr in parser.__dict__:
        dh.dev(parser, lerepi_handler.TEMP)
        sys.exit()
    lerepi_handler.collect_jobs()

    try:
        lerepi_handler.run()
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
