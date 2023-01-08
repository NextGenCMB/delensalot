import numpy as np

from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *
import os
from os.path import join as opj

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        QE_lensrec = True,
        MAP_lensrec = True,
        OMP_NUM_THREADS = 16
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = '',
        K = 'p_p',
        ITMAX = 5,
        simidxs_mf = np.arange(0,5),
        LENSRES = 1.7,
        Lmin = 2, 
        lmax_filt = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 4000,
        mmax_ivf = 4000,
        lmin_ivf = 2,
        mmin_ivf = 2,
    ),
    data = DLENSALOT_Data(
        IMIN = 0,
        IMAX = 5,
        package_ = 'lenscarf',
        module_ = 'ana.config.examples.mwe.data_mwe.sims_mwe',
        class_ = 'mwe',
        class_parameters = {
            'nlev': '0.25'
        },
        data_type = 'alm',
        data_field = "eb",
        beam = 1,
        lmax_transf = 4000,
        nside = 2048
    ),
    noisemodel = DLENSALOT_Noisemodel(
        typ = 'trunc',
        lmin_tlm = 10,
        lmin_elm = 10,
        lmin_blm = 200,
        nlev_t = 0.25/np.sqrt(2),
        nlev_p = 0.25
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield", "calc_blt"],
        ivfs = 'sepTP',
        qlms = 'sepTP',
        cg_tol = 1e-3,
        lmax_qlm = 4000,
        mmax_qlm = 4000
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_meanfield", "calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
        filter = 'opfilt_ee_wl.alm_filter_ninv_wl',
        cg_tol = 1e-3,
        iterator_typ = 'constmf', # Either pertmf or const_mf
        mfvar = '',
    )
)
