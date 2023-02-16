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

from lenscarf.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec", "MAP_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_fastWF_lownoise',
        Lmin = 2, 
        lm_max_ivf = (3000, 3000),
        lmin_teb = (2, 2, 200)
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 4000,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['CSCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4000', 'nlevp_sqrt05')
        },
        nlev_t = 1.00/np.sqrt(2),
        nlev_p = 0.5,
        beam = 1.00,
        lmax_transf = 4096,
        nside = 2048,
        transferfunction = 'gauss_no_pixwin'
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        nlev_t = 1.00/np.sqrt(2),
        nlev_p = 0.5
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-6,
        lm_max_qlm = (4000, 4000)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        itmax = 10,
        cg_tol = 1e-6,
        lensres = 1.7,
        iterator_typ = 'fastWF',
        lm_max_unl = (4000, 4000),
        lm_max_qlm = (4000, 4000)
    )
)