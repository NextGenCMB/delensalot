#!/usr/bin/env python

"""lerepi2status.py: transformer module to build status report model from lerepi configuration file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os, sys
from os.path import join as opj
import importlib
import traceback

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp

import hashlib

import lenscarf.lerepi.core.sr as sr
from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2T_Transformer
from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Model as DLENSALOT_Model, DLENSALOT_Concept
from lenscarf.lerepi.core.metamodel.dlensalot_v2 import DLENSALOT_Model as DLENSALOT_Model_v2


class l2s_Transformer:
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        def _process_Status(dl):
            dl.__dict__.update(cf.__dict__)
            dl.analysispath = l2T_Transformer().build(cf)
            dl.itmax = cf.iteration.ITMAX
            dl.version = cf.iteration.V

        # TODO build list of files to be checked
        # build list of modifiers to test the files against, as in version, itmax, mf, ..

        dl = DLENSALOT_Concept()
        _process_Status(dl)

        
        return dl

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build_v2(self, cf):
        def _process_Status(dl):
            dl.__dict__.update(cf.__dict__)
            dl.analysispath = l2T_Transformer().build(cf)
            dl.itmax = cf.analysis.itmax
            dl.version = cf.analysis.V

        dl = DLENSALOT_Concept()
        _process_Status(dl)

        
        return dl


class l2j_Transformer:
    """Extracts parameters needed for the specific D.Lensalot jobs
    """
    def build(self, cf):
        def _process_Jobs(jobs):
            jobs.append(((cf, l2s_Transformer()), sr.analysisreport))

        jobs = []
        _process_Jobs(jobs)

        return jobs


@transform.case(DLENSALOT_Model, l2j_Transformer)
def f0(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2s_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_v2, l2j_Transformer)
def f2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_v2, l2s_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)
