"""
Full sky iterative delensing on ffp10 polarization data generated on the fly, inclusive of isotropic white noise.
Here, delensing is done on two simulation sets.
Simulated maps are used up to lmax 4000.
The noise model is isotropic and white, and truncates B modes lmin<200
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
        OMP_NUM_THREADS = 2
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF',
        simidxs = np.arange(0,2),
        TEMP_suffix = 'my_first_dlensalot_analysis_fastWF',
        Lmin = 2, 
        lm_max_len = (4000, 4000),
        lm_ivf = ((2, 3000),(2, 3000)),
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 3000,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['CSCRATCH'], 'generic','nlevp1.00')
        },
        nlev_t = 1.00/np.sqrt(2),
        nlev_p = 1.00,
        beam = 1.00,
        lmax_transf = 4096,
        nside = 2048,
        transferfunction = 'gauss_no_pixwin'
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 200),
        nlev_t = 1.00/np.sqrt(2),
        nlev_p = 1.00
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield"],
        filter_directional = 'isotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-3,
        lm_max_qlm = (3000, 3000)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_meanfield"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
        filter_directional = 'isotropic',
        itmax = 10,
        cg_tol = 1e-3,
        lensres = 1.7,
        iterator_typ = 'fastWF',
        lm_max_unl = (3000, 3000),
        lm_max_qlm = (3000, 3000)
    )
)