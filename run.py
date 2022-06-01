#!/usr/bin/env python

"""run.py: Entry point for running lerepi
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import logging

from lerepi.core.parser import lerepi_parser
from lerepi.core import handler


if __name__ == '__main__':
    # futher log
    logging.basicConfig(level=logging.INFO)

    lparser = lerepi_parser()
    lparser.validate()
    parser = lparser.get_parser()

    lerepi_handler = handler.handler(parser)
    lerepi_handler.run()
