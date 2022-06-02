#!/usr/bin/env python

"""delensing_interface.py: This module is a simple passthrough to dlensalot.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import logging
from logdecorator import log_on_start, log_on_end

from lenscarf.core import handler


class Dlensalot(object):
# TODO it feels unclean that self.QE has no jobs and no run. Implement at some point?
    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(dlensalot_model):
        if dlensalot_model.get_btemplate_per_iteration:
            QE = handler.QE_delensing(dlensalot_model)
            MAP = handler.B_template_construction(QE, dlensalot_model)
        else:
            QE = handler.QE_delensing(dlensalot_model)
            MAP = handler.MAP_delensing(QE, dlensalot_model)
        MAP.collect_jobs()
        MAP.run()

    def __repr__(self):
        print(self.__dict__)
