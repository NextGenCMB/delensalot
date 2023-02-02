"""
Full sky iterative delensing on simulation polarization data generated on the fly, inclusive of noise and without foregrounds.
Here, delensing is done on two simulations.
Simulated maps are used up to lmax 3000.
The noise model is isotropic and white, and truncates B modes lmin<200
QE and iterative reconstruction uses isotropic filters, and we apply a fast Wiener filtering to the iterative reconstruction 
"""

import numpy as np
import plancklens
from plancklens import utils
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
        simidxs = np.arange(0,1),
        TEMP_suffix = 'postborn_GF_Louissettings',
        Lmin = 2,
        lm_max_ivf = (3000, 3000),
    ),
    data = DLENSALOT_Data(
        package_ = 'n32',
        module_ = 'sims.sims_postborn',
        class_ = 'sims_postborn',
        class_parameters = {
            'lmax_cmb': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['CSCRATCH'], 'sims_postborn', 'nlevp_sqrt2')
        },
        nside = 2048,
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        lmax_transf = 4096,
        data_type = 'map',
        data_field = 'qu',
        beam = 1
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        lmin_teb = (30, 30, 30),
        nlev_t = 1.00,
        nlev_p = np.sqrt(2)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-3,
        lm_max_qlm = (4000, 4000)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'isotropic',
        itmax = 20,
        cg_tol = 1e-5,
        lensres = 1.7,
        iterator_typ = 'constmf',
        lm_max_unl = (4000, 4000),
        lm_max_qlm = (4000, 4000)
    )
)