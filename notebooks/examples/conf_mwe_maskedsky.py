"""
Masked sky iterative delensing on simulated CMB polarization data generated on the fly, inclusive of isotropic white noise.
Here, delensing is done on two simulation sets.
The noise model is isotropic and white, and truncates T,E, and B modes at low multipoles.
QE and iterative reconstruction uses anisotropic filters. 
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
        simidxs = np.arange(0,1),
        simidxs_mf = np.arange(0,60),
        TEMP_suffix = 'my_first_dlensalot_analysis_maskedsky',
        Lmin = 2, 
        lm_max_ivf = (2048, 2048),
        lmin_teb = (10, 10, 200),
        mask = opj(os.environ['SCRATCH'], 'dlensalot/lenscarf/generic/sims_cmb_len_lminB200_my_first_dlensalot_analysis_maskedsky/mask.fits')
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 2048,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['CSCRATCH'], 'generic', 'nside_1024', 'lmax_2048', 'nlevp_sqrt(2)')
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 2048,
        nside = 1024,
        transferfunction = 'gauss_with_pixwin'
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'dlensalot/lenscarf/generic/sims_cmb_len_lminB200_my_first_dlensalot_analysis_maskedsky/rhits.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi","calc_meanfield", "calc_blt"],
        filter_directional = 'anisotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-4,
        lm_max_qlm = (2048, 2048)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi"],
        filter_directional = 'anisotropic',
        itmax = 12,
        cg_tol = 1e-4,
        lensres = 1.7,
        iterator_typ = 'constmf',
        lm_max_unl = (2048, 2048),
        lm_max_qlm = (2048, 2048)
    )
)