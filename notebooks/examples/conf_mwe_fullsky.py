"""
Full sky iterative delensing on simulated CMB polarization maps generated on the fly, inclusive of isotropic white noise.
Delensing is done on one simulation.
The noise model is isotropic and white.
QE and iterative reconstruction uses isotropic filters, and we apply a fast Wiener filtering to the iterative reconstruction 
"""

import numpy as np
import os
import plancklens
from plancklens import utils
from os.path import join as opj

from delensalot.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec", "MAP_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 1
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'mfda',
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
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-6,
        lm_max_qlm = (3000, 3000)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        itmax = 5,
        cg_tol = 1e-5,
        lensres = 1.0,
        iterator_typ = 'constmf',
        lm_max_unl = (3200, 3200),
        lm_max_qlm = (3000, 3000)
    )
)