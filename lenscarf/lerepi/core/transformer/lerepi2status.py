#!/usr/bin/env python

"""lerepi2status.py: transformer module to build status report model from lerepi configuration file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os, sys
from os.path import join as opj


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end


import lenscarf.lerepi.core.sr as sr
from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2T_Transformer
from lenscarf.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model, DLENSALOT_Concept


class l2s_Transformer:

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build_v2(self, cf):
        def _process_Status(dl):
            dl.__dict__.update(cf.__dict__)
            dl.analysispath = l2T_Transformer().build_v2(cf)
            dl.itmax = cf.analysis.ITMAX
            dl.version = cf.analysis.V
            dl.imax = cf.data.IMAX

        dl = DLENSALOT_Concept()
        _process_Status(dl)

        return dl


class l2j_Transformer:
    """Extracts parameters needed for the specific D.Lensalot jobs
    """
    def build_v2(self, cf):
        def _process_Jobs(jobs):
            jobs.append({"report":((cf, l2s_Transformer()), sr.analysisreport)})

        jobs = []
        _process_Jobs(jobs)

        return jobs


@transform.case(DLENSALOT_Model, l2j_Transformer)
def f0(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model, l2s_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)
