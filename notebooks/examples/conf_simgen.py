"""
Simulates CMB polarization maps generated on the fly, inclusive of isotropic white noise.
"""

import numpy as np
import os
import plancklens
from plancklens import utils
from os.path import join as opj

from delensalot.lerepi.core.metamodel.dlensalot_mm import *


dlensalot_model = DLENSALOT_Model(
    defaults_to = 'T',

    job = DLENSALOT_Job(
        jobs = ["generate_sim"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 1
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF',
        simidxs = np.arange(0,20),
        TEMP_suffix = 'simgen',
        Lmin = 2, 
        lm_max_ivf = (3000, 3000),
        lmin_teb = (2, 2, 200)
    ),
    data = DLENSALOT_Data(
        package_ = 'delensalot',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4096', 'nlevp_sqrt(2)')
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 3000,
        nside = 2048,
        transferfunction = 'gauss_no_pixwin'
    )
)