#!/usr/bin/env python

"""delensing_interface.py: This module is a simple passthrough to dlensalot.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import logging
from logdecorator import log_on_start, log_on_end

from lenscarf.core import handler


class Dlensalot():
    def __init__(self, lc):
        self.QE = handler.QE_delensing(lc)
        self.MAP = handler.MAP_delensing(self.QE, lc)

    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        # TODO it feels unclean that self.QE has no jobs and no run. Implement at some point?
        self.MAP.collect_jobs()
        self.MAP.run()
