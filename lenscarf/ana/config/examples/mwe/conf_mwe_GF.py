"""
Full sky iterative delensing on simulation polarization data generated on the fly, without noise or foregrounds.
Here, delensing is done on two simulation sets.
Simulated maps are used up to lmax 4000.
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
        OMP_NUM_THREADS = 16
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF',
        simidxs = np.arange(0,2),
        TEMP_suffix = 'postborn_GF',
        Lmin = 2, 
        lm_max_len = (4000, 4000),
        lm_ivf = ((2, 4000),(2, 4000)),
    ),
    data = DLENSALOT_Data(
        package_ = 'n32',
        module_ = 'sims.sims_postborn',
        class_ = 'sims_postborn',
        class_parameters = {
            'lmax_cmb': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['HOME'], 'pixphas_nside_GF')
        },
        nside = 2048,
        nlev_t = 0.25/np.sqrt(2),
        nlev_p = 0.25,
        lmax_transf = 4096,
        data_type = 'map',
        data_field = 'qu',
        beam = 1
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 200),
        nlev_t = 0.25/np.sqrt(2),
        nlev_p = 0.25
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield", "calc_blt"],
        filter_directional = 'isotropic',
        qlm_type = 'sepTP',
        cg_tol = 1e-3,
        lm_max_qlm = (4000, 4000)
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_meanfield"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
        filter_directional = 'isotropic',
        itmax = 10,
        cg_tol = 1e-3,
        lensres = 0.8,
        iterator_typ = 'constmf',
        lm_max_unl = (4000, 4000),
        lm_max_qlm = (4000, 4000)
    )
)