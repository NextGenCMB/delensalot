import numpy as np

from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *
from MSC import pospace

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
        TEMP_suffix = 'r100_tol5e5_perfreport2',
        K = 'p_p',
        V = '',
        ITMAX = 4,
        simidxs_mf = np.arange(0,4),
        zbounds =  ('nmr_relative', 100),
        zbounds_len = ('extend', 10.),   
        pbounds = [0, 2*np.pi],
        LENSRES = 1.7,
        Lmin = 5, 
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
        IMAX = 3,
        simidxs = np.arange(0,4),
        package_ = 'lenscarf',
        module_ = 'lerepi.config.cmbs4.data.data_08d',
        class_ = 'ILC_May2022',
        class_parameters = {
            'fg': '00'
        },
        data_type = 'map',
        data_field = "qu",
        beam = 2.3,
        lmax_transf = 4000,
        nside = 2048
    ),
    noisemodel = DLENSALOT_Noisemodel(
        typ = 'OBD',
        BMARG_LIBDIR = '/global/cscratch1/sd/sebibel/cmbs4/OBD_matrices/08d/r100/',
        BMARG_LCUT = 200,
        BMARG_RESCALE = (0.65/0.59)**2,
        ninvjob_geometry = 'healpix_geometry',
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 30,
        CENTRALNLEV_UKAMIN = 0.65,
        nlev_t = 0.65/np.sqrt(2),
        nlev_p = 0.65,
        nlev_dep = 10000.,
        inf = 1e4,
        mask = ('nlev', 100),
        rhits_normalised = ('/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08d/masks/08d_rhits_positive_nonan.fits', 100),
        tpl = 'template_dense'
    ),
    qerec = DLENSALOT_Qerec(
        ivfs = 'sepTP', # Change only if other than sepTP for QE is desired
        qlms = 'sepTP',
        cg_tol = 3e-4,
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
        cg_tol = 5e-5,
        tasks = ["calc_phi"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
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
        iterations = [4],
        edges = ['cmbs4', 'ioreco'], # overwritten when binning=unbinned
        masks = ("nlevels", [1.2, 2, 10, 50]),
        lmax = 2048, # automatically set to 200 when binning=unbinned
        Cl_fid = 'ffp10',
        binning = 'binned',
        spectrum_calculator = pospace
    )
)
