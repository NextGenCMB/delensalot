
import numpy as np
import healpy as hp

from lerepi.metamodel.dlensalot import *

dlensalot_model = DLENSALOT_Model(
    data = DLENSALOT_Data(
        DATA_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/',
        rhits = '/global/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits',
        fg = '00',
        mask_suffix = 50,
        sims = '08d/ILC_May2022',
        mask = '08d/ILC_May2022',
        masks = ['08d/ILC_May2022'], # TODO lenscarf supports multiple masks. But lerepi currently doesn't
        nside = 2048,
        BEAM = 2.3,
        lmax_transf = 4000, # can be distinct from lmax_filt for iterations
        transf = hp.gauss_beam,
        zbounds = ('08d/ILC_May2022', np.inf),
        zbounds_len = ('08d/ILC_May2022', 5.), # Outside of these bounds the reconstructed maps are assumed to be zero
        pbounds = (0., 2*np.pi), # Longitude cuts, if any, in the form (center of patch, patch extent)
        isOBD = False,
        BMARG_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08d/',
        BMARG_LCUT = 200,
        tpl = 'template_dense',
        CENTRALNLEV_UKAMIN = 0.59,
        nlev_t = 0.59/np.sqrt(2),
        nlev_p = 0.59
    ),
    iteration = DLENSALOT_Iteration(
        K = 'p_p',# Lensing key, either p_p, ptt, p_eb
        # version, can be 'noMF
        V = '',
        ITMAX = 15,
        IMIN = 0,
        IMAX = 0,
        # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
        # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
        # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
        QE_LENSING_CL_ANALYSIS = False,
        # Change the following block only if exotic transferfunctions are desired
        STANDARD_TRANSFERFUNCTION = True,
        # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
        FILTER = 'cinv_sepTP',
        # Change the following block only if exotic chain descriptor are desired
        CHAIN_DESCRIPTOR = 'default',
        # Change the following block only if other than sepTP for QE is desired
        FILTER_QE = 'sepTP',
        # Choose your iterator. Either pertmf or const_mf
        ITERATOR = 'pertmf',
        # The following block defines various multipole limits. Change as desired
        lmax_filt = 4096, # unlensed CMB iteration lmax
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 10,
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        LENSRES = 1.7, # Deflection operations will be performed at this resolution
        Lmin = 2, # The reconstruction of all lensing multipoles below that will not be attempted
        # Meanfield, OBD, and tol settings
        CG_TOL = 1e-3,
        TOL = 4,
        soltn_cond = lambda it: True,
        OMP_NUM_THREADS = 8,
        nsims_mf = 10
    ),
    geometry = DLENSALOT_Geometry(
        lmax_unl = 4000,
        zbounds = ('08d/ILC_May2022', np.inf),
        zbounds_len = ('08d/ILC_May2022', 5.),
        pbounds = (0., 2*np.pi),
        nside = 2048,
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        ninvjob_geometry = 'healpix_geometry'
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
    )
)

