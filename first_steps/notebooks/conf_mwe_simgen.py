"""
Simulates CMB polarization maps generated on the fly, inclusive of isotropic white noise.
"""

import numpy as np
import os
import plancklens
from plancklens import utils
from os.path import join as opj

from delensalot.config.metamodel.dlensalot_mm import *


dlensalot_model = DLENSALOT_Model(
    defaults_to = 'P_FS_CMBS4',

    job = DLENSALOT_Job(
        jobs = ["generate_sim"]
    ),                             
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'simgen',
    ),
    data = DLENSALOT_Data(
        package_ = 'delensalot',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 5976,
            'cls_unl': utils.camb_clfile(opj(os.path.dirname(plancklens.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax5976', 'nlevp_sqrt(2)')
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 5976,
        nside = 2048,
    )
)