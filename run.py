"""Entry point for running lerepi 
"""

from lerepi.core.parser import lerepi_parser
from lerepi.core import handler


if __name__ == '__main__':
    parser = lerepi_parser()
    parser.validate()
    
    lerepi_handler = handler(parser)
    handler.run()
