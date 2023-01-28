"""
Masked sky iterative delensing on FFP10 simulation polarization data generated on the fly, inclusive of isotropic white noise at 0.25 mu k arcmin.
Here, delensing is done on two simulation sets.
Simulated maps are used up to lmax 4000.
The noise model is isotropic and white, and truncates B modes lmin<200
QE and iterative reconstruction uses isotropic filters, and we apply a fast Wiener filtering to the iterative reconstruction 
"""

import numpy as np
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
        simidxs = np.arange(0,2),
        TEMP_suffix = 'my_first_dlensalot_analysis',
        Lmin = 2, 
        lm_max_len = (4000, 4000),
        lm_ivf = ((2, 4000),(2, 4000))
        zbounds =  ('nmr_relative', 100),
        zbounds_len = ('extend', 5.)
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'ana.config.examples.mwe.data_mwe.sims_mwe_fullsky',
        class_ = 'mwe',
        class_parameters = {
            'nlev_p': 0.25
        }
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 200),
        nlev_t = 0.25/np.sqrt(2),
        nlev_p = 0.25,
        mask = ('nlev', np.inf),
        rhits_normalised = (opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield"],
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
        iterator_typ = 'fastWF',
        lm_max_unl = (4000, 4000),
        lm_max_qlm = (4000, 4000)
    )
)