"""
Full sky iterative delensing on simulated CMB polarization maps generated on the fly, inclusive of isotropic white noise.
QE and iterative reconstruction use isotropic filters, deproject B-modes l<200, and reconstruct unlensed CMB up to 4200. iterative rec uses fastWF
Parameters not listed here default to 'P_FS_CMBS4'
"""

import numpy as np
import os
import delensalot
from delensalot import utils
from os.path import join as opj

from delensalot.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'P_FS_CMBS4',
    job = DLENSALOT_Job(
        jobs = ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    ),                          
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_fastWF',
        Lmin = 10
    ),
    data = DLENSALOT_Data(
        class_parameters = {
            'lmax': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(delensalot.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4096', 'nlevp_sqrt(2)')
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 4000,
        nside = 2048,
    ),
    noisemodel = DLENSALOT_Noisemodel(
        nlev_t = 1.00,
        nlev_p = np.sqrt(2)
    ),
    qerec = DLENSALOT_Qerec(
        lm_max_qlm = (4000, 4000)
    ),
    itrec = DLENSALOT_Itrec(
        iterator_typ = 'fastWF',
        itmax = 5,
        lm_max_unl = (4200, 4200),
        lm_max_qlm = (4000, 4000)
    )
)