import numpy as np
from dlensalot.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_noisemodel", "QE_lensrec", "MAP_lensrec"],
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 16
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        TEMP_suffix = 'my_first_dlensalot_analysis',
        ITMAX = 10,
        Lmin = 2, 
        lm_max_len = (4000, 4000),
        lm_max_unl = (4000, 4000),
        lm_ivf = ((2, 4000),(2, 4000)),
    ),
    data = DLENSALOT_Data(
        simidxs = np.arange(0,200),
        package_ = 'dlensalot',
        module_ = 'ana.config.examples.mwe.data_mwe.sims_mwe',
        class_ = 'mwe',
        class_parameters = {
            'nlev': '0.25'
        }
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
        ivfs = 'sepTP',
        qlms = 'sepTP',
        cg_tol = 1e-3,
        lm_max_qlm = (4000, 4000),
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_meanfield", "calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
        filter = 'opfilt_ee_wl.alm_filter_ninv_wl',
        cg_tol = 1e-3,
        lensres = 0.8,
        lm_max_qlm = (4000, 4000)
    )
)