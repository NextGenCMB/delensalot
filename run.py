#!/usr/bin/env python

"""run.py: Entry point for running lerepi
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import logging

from lerepi.core.parser import lerepi_parser
from lerepi.core import handler


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("healpy").disabled = True

    log = logging.getLogger(__name__)
    log.setLevel(logging.WARNING)

    ConsoleOutputHandler = logging.StreamHandler()
    log.addHandler(ConsoleOutputHandler)
    formatter = logging.Formatter('%(asctime)s:: %(name)s:: %(levelname)s - %(message)s')
    ConsoleOutputHandler.setFormatter(formatter)

    
    lparser = lerepi_parser()
    lparser.validate()
    parser = lparser.get_parser()

    lerepi_handler = handler.handler(parser)
    lerepi_handler.run()
