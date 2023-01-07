import numpy as np

from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *
import os
from os.path import join as opj

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        build_OBD = False,
        QE_lensrec = False,
        MAP_lensrec = True,
        map_delensing = False,
        inspect_result = False,
        OMP_NUM_THREADS = 16
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = '',
        K = 'p_p',
        V = '',
        ITMAX = 12,
        simidxs_mf = np.arange(0,100),
        zbounds =  ('nmr_relative', np.inf),
        zbounds_len = ('extend', 5.),   
        pbounds = [1.97, 5.71],
        LENSRES = 1.7,
        Lmin = 2, 
        lmax_filt = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        STANDARD_TRANSFERFUNCTION = True
    ),
    data = DLENSALOT_Data(
        IMIN = 0,
        IMAX = 1,
        package_ = 'dlensalot',
        module_ = 'lerepi.config.examples.mwe.data_mwe.sims_mwe',
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
        typ = 'OBD',
        BMARG_LIBDIR = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/'),
        BMARG_LCUT = 200,
        BMARG_RESCALE = (0.42/0.350500)**2,
        ninvjob_geometry = 'healpix_geometry',
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 30,
        CENTRALNLEV_UKAMIN = 0.42,
        nlev_t = 0.42/np.sqrt(2),
        nlev_p = 0.42,
        nlev_dep = 10000.,
        inf = 1e4,
        mask = ('nlev', np.inf),
        rhits_normalised = (opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'), np.inf),
        tpl = 'template_dense'
    ),
    qerec = DLENSALOT_Qerec(
        ivfs = 'sepTP', # Change only if other than sepTP for QE is desired
        qlms = 'sepTP',
        cg_tol = 1e-3,
        ninvjob_qe_geometry = 'healpix_geometry_qe',
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        QE_LENSING_CL_ANALYSIS = False, # Change only if a full, Planck-like QE lensing power spectrum analysis is desired
        chain = DLENSALOT_Chaindescriptor(
            p0 = 0,
            p1 = ["diag_cl"],
            p2 = None,
            p3 = 2048,
            p4 = np.inf,
            p5 = None,
            p6 = 'tr_cg',
            p7 = 'cache_mem'
        )
    ),
    itrec = DLENSALOT_Itrec(
        filter = 'opfilt_ee_wl.alm_filter_ninv_wl',
        cg_tol = 1e-4,
        tasks = ["calc_phi", "calc_meanfield", "calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        iterator_typ = 'constmf', # Either pertmf or const_mf
        mfvar = '',
        soltn_cond = lambda it: True,
        stepper = DLENSALOT_Stepper(
            typ = 'harmonicbump',
            xa = 400,
            xb = 1500
        )
    ),
)
