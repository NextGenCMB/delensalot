import numpy as np
import healpy as hp

from lerepi.core.metamodel.dlensalot import *


dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        QE_lensrec = False,
        MAP_lensrec = True,
        Btemplate_per_iteration = True,
        map_delensing = True,
        inspect_result = False
    ),
    data = DLENSALOT_Data(
        DATA_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/',
        rhits = '/global/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits',
        TEMP_suffix = 'OBD200', 
        fg = '00',
        mask_suffix = 2,
        mask_norm = 1.0,
        sims = 'cmbs4/08d/ILC_May2022',
        mask = 'cmbs4/08d/ILC_May2022',
        masks = ['cmbs4/08d/ILC_May2022'],
        nside = 2048,
        BEAM = 2.3,
        lmax_transf = 4000,
        transf = hp.gauss_beam,
        zbounds = ('cmbs4/08d/ILC_May2022', np.inf),
        zbounds_len = ('cmbs4/08d/ILC_May2022', 5.),
        pbounds = (0., 2*np.pi),
        isOBD = False,
        BMARG_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08d/',
        BMARG_LCUT = 200,
        tpl = 'template_dense',
        CENTRALNLEV_UKAMIN = 0.59,
        nlev_t = 0.59/np.sqrt(2),
        nlev_p = 0.59
    ),
    iteration = DLENSALOT_Iteration(
        K = 'p_p',
        V = '', 
        ITMAX = 10,
        IMIN = 0,
        IMAX = 99,
        nsims_mf = 100,
        OMP_NUM_THREADS = 8,
        Lmin = 4, 
        CG_TOL = 1e-3,
        TOL = 3,
        soltn_cond = lambda it: True,
        lmax_filt = 4096,
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 200, #Supress all modes below this value, hacky version of OBD
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        LENSRES = 1.7, # Deflection operations will be performed at this resolution
        # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
        # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
        # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
        QE_LENSING_CL_ANALYSIS = False,
        STANDARD_TRANSFERFUNCTION = True, # Change the following block only if exotic transferfunctions are desired
        FILTER = 'cinv_sepTP', # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
        CHAIN_DESCRIPTOR = 'default', # Change the following block only if exotic chain descriptor are desired
        FILTER_QE = 'sepTP', # Change the following block only if other than sepTP for QE is desired
        ITERATOR = 'pertmf' # Choose your iterator. Either pertmf or const_mf
    ),
    geometry = DLENSALOT_Geometry(
        lmax_unl = 4000,
        zbounds = ('cmbs4/08d/ILC_May2022', np.inf),
        zbounds_len = ('cmbs4/08d/ILC_May2022', 5.),
        pbounds = (0., 2*np.pi),
        nside = 2048,
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        ninvjob_geometry = 'healpix_geometry',
        ninvjob_qe_geometry = 'healpix_geometry_qe'
    ),
    chain_descriptor = DLENSALOT_Chaindescriptor(
        p0 = 0,
        p1 = ["diag_cl"],
        p2 = None,
        p3 = 2048,
        p4 = np.inf,
        p5 = None,
        p6 = 'tr_cg',
        p7 = 'cache_mem'
    ),
    stepper = DLENSALOT_Stepper(
        typ = 'harmonicbump',
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        xa = 400,
        xb = 1500
    ),
    map_delensing = DLENSALOT_Mapdelensing(
        edges = 'cmbs4',
        IMIN = 0,
        IMAX = 99,
        ITMAX = 10,
        fg = '00',
        base_mask = 'cmbs4/08d/ILC_May2022',
        nlevels = [2., 5.],
        nside = 2048,
        lmax_cl = 2048,
        beam = 2.3,
        lmax_transf = 4000,
        transf = 'gauss',
        Cl_fid = 'ffp10'
    )
)
