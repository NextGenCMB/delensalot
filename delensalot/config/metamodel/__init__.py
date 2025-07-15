#!/usr/bin/env python

"""metamodel.__init__.py:
    The metamodel defines the available attributes and their valid values a delensalot configuration file can have.
    The init file is mainly a collection of default values for delensalot model.
    These dictionaries are, depending on the configuration, loaded at instantiation of a delensalot model.
"""

import os
from os.path import join as opj
import numpy as np
import psutil

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc

DEFAULT_NotAValue = -123456789
DEFAULT_NotASTR = '.-.,.-.,'
DEFAULT_NotValid = 9876543210123456789