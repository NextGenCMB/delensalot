#!/usr/bin/env python

"""run.py: Entry point for running D.lensalot
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
import logging
import traceback

from lenscarf.lerepi.core.parser import lerepi_parser
from lenscarf.lerepi.core import handler


if __name__ == '__main__':
    datefmt = "%m-%d %H:%M"
    FORMAT = '%(levelname)s:: %(asctime)s:: %(name)s.%(funcName)s - %(message)s'
    formatter = logging.Formatter(FORMAT, datefmt=datefmt)
    ConsoleOutputHandler = logging.StreamHandler()
    ConsoleOutputHandler.setFormatter(formatter)
    ConsoleOutputHandler.setLevel(logging.INFO)

    sys_logger = logging.getLogger(__name__)
    sys_logger.addHandler(ConsoleOutputHandler)
    sys_logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, handlers=[ConsoleOutputHandler])
    logging.getLogger("healpy").disabled = True

    lparser = lerepi_parser()
    if lparser.validate():
        parser = lparser.get_parser()

    lerepi_handler = handler.handler(parser)
    if "purgehashs" in parser.__dict__:
        handler.purge(parser, lerepi_handler.TEMP)
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