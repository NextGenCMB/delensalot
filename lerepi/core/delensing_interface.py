#!/usr/bin/env python

"""delensing_interface.py: This module is a simple passthrough to dlensalot.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import logging
from logdecorator import log_on_start, log_on_end

from lenscarf.core import handler


class Dlensalot(object):

    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        self.QE = handler.QE_delensing(self)

        self.MAP = handler.MAP_delensing(self.QE, self)
        # TODO it feels unclean that self.QE has no jobs and no run. Implement at some point?
        self.MAP.collect_jobs()
        self.MAP.run()
