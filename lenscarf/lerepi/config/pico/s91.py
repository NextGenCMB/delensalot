import numpy as np

from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *
from MSC import pospace

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        build_OBD = False,
        QE_lensrec = False,
        MAP_lensrec = False,
        map_delensing = True,
        inspect_result = False,
        OMP_NUM_THREADS = 16
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = '',
        K = 'p_p',
        V = '',
        ITMAX = 10,
        LENSRES = 1.7,
        Lmin = 2, 
        lmax_filt = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        STANDARD_TRANSFERFUNCTION = 'with_pixwin'
    ),
    data = DLENSALOT_Data(
        IMIN = 0,
        IMAX = 19,
        simidxs = np.arange(1,21,2),
        package_ = 'lenscarf',
        module_ = 'lerepi.config.pico.data.sims_90',
        class_ = 'ILC_Matthieu_Dec21',
        class_parameters = {
            'fg': '91'
        },
        beam = 8.0,
        lmax_transf = 4000,
        nside = 2048
    ),
    noisemodel = DLENSALOT_Noisemodel(
        typ = 'trunc',
        ninvjob_geometry = 'healpix_geometry',
        lmin_tlm = 30,
        lmin_elm = 2,
        lmin_blm = 200,
        CENTRALNLEV_UKAMIN = 2.0,
        nlev_t = 2.0/np.sqrt(2),
        nlev_p = 2.0,
        inf = 1e4
    ),
    qerec = DLENSALOT_Qerec(
        FILTER_QE = 'sepTP', # Change only if other than sepTP for QE is desired
        CG_TOL = 1e-3,
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
        FILTER = 'opfilt_ee_wl.alm_filter_ninv_wl',
        TOL = 1e-4,
        tasks = ["calc_phi", "calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
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
    madel = DLENSALOT_Mapdelensing(
        iterations = [8,10],
        nlevels = [1.2, 2, 10, 50],
        masks = ("masks", ["<path/to/mask>", "<path/to/mask>"]), #("nlevels", [1.2, 2, 10, 50])
        lmax_cl = 200,
        Cl_fid = 'ffp10',
        spectrum_type = 'unbinned',
        spectrum_calculator = pospace
    )
)
