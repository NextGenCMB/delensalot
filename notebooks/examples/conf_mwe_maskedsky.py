"""
Masked sky iterative delensing on simulated CMB polarization data generated on the fly, inclusive of isotropic white noise.
Here, delensing is done on two simulation sets.
The noise model is isotropic and white, and truncates T,E, and B modes at low multipoles.
QE and iterative reconstruction uses anisotropic filters. 
"""

import numpy as np
import os
import delensalot
from delensalot import utils
from os.path import join as opj

from delensalot.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec", "MAP_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = '',
        simidxs = np.arange(0,1),
        simidxs_mf = np.arange(0,5),
        TEMP_suffix = 'mfda_maskedsky',
        Lmin = 1, 
        lm_max_ivf = (3000, 3000),
        lmin_teb = (10, 10, 200),
        mask = opj(os.environ['SCRATCH'], 'delensalot/generic/sims_cmb_len_lminB200_mfda_maskedsky/mask.fits')
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
        lmax_transf = 4096,
        nside = 2048,
        transferfunction = 'gauss_no_pixwin'
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'delensalot/generic/sims_cmb_len_lminB200_mfda_maskedsky/rhits.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi","calc_meanfield", "calc_blt"],
        filter_directional = 'anisotropic',
        lm_max_qlm = (3000, 3000),
        cg_tol = 1e-3
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi"],
        filter_directional = 'anisotropic',
        itmax = 5,
        lm_max_unl = (3200, 3200),
        lm_max_qlm = (3000, 3000)
    )
)