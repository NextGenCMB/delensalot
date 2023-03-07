"""
Simulates CMB polarization maps generated on the fly, inclusive of isotropic white noise.
"""

import numpy as np
import os
import plancklens
from plancklens import utils
from os.path import join as opj

from delensalot.lerepi.core.metamodel import dlensalot_mm
from delensalot.lerepi.core.metamodel.dlensalot_mm import *


dlensalot_model = DLENSALOT_Model(defaults_to='T')