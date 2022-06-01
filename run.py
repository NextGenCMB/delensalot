#!/usr/bin/env python

"""run.py: Entry point for running lerepi
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


from lerepi.core.parser import lerepi_parser
from lerepi.core import handler


if __name__ == '__main__':
    lparser = lerepi_parser()
    lparser.validate()
    parser = lparser.get_parser()

    lerepi_handler = handler.handler(parser)
    lerepi_handler.run()
