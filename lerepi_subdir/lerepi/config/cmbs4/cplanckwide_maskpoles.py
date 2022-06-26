import numpy as np
import healpy as hp

from lerepi.core.metamodel.dlensalot import *
# TODO how to add the maskpoles thingy?

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        QE_lensrec = False,
        MAP_lensrec = True,
        Btemplate_per_iteration = True,
        map_delensing = True,
        inspect_result = False
    ),
    data = DLENSALOT_Data(
        DATA_LIBDIR = None,
        rhits = None, # TODO implement
        TEMP_suffix = '',
        fg = None,
        mask_suffix = None,
        mask_norm = None, # TODO implement
        sims = 'plancklens/sims/maps/cmb_maps_nlev', # TODO turn this into DLENSALOT_sims or DLENSALOT_data
        mask = None, # TODO implement
        masks = None, # TODO implement
        zbounds = (-1.,1.), # TODO implement
        zbounds_len = (-1.,1.), # TODO implement
        nside = 2048,
        BEAM = 1.0,
        lmax_transf = 4000,
        transf = hp.gauss_beam,
        pbounds = (0, 2*np.pi),
        isOBD = False,  
        BMARG_LIBDIR = None,
        BMARG_LCUT = None,
        tpl = 'template_dense',
        BMARG_RESCALE = 1.,
        CENTRALNLEV_UKAMIN = 0.5,
        nlev_t = 0.5/np.sqrt(2),
        nlev_p = 0.5
    ),
    iteration = DLENSALOT_Iteration(
        K = 'p_p',
        V = '', 
        ITMAX = 12,
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
        lmin_blm = 200, #Supress all modes below this value, hacky version of OBD, overwrites isOBD
        lmax_qlm = 4096,
        mmax_qlm = 4096,
        lmax_unl = 4000,
        mmax_unl = 4000,
        lmax_ivf = 3000,
        mmax_ivf = 3000,
        lmin_ivf = 10,
        mmin_ivf = 10,
        LENSRES = 1.7,
        QE_LENSING_CL_ANALYSIS = False, # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
        STANDARD_TRANSFERFUNCTION = True, # Change the following block only if exotic transferfunctions are desired
        FILTER = 'cinv_sepTP', # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
        FILTER_QE = 'sepTP', # Change the following block only if other than sepTP for QE is desired
        ITERATOR = 'constmf' # Choose your iterator. Either pertmf or constmf
    ),
    geometry = DLENSALOT_Geometry(
        lmax_unl = 4000,
        zbounds = (-1,1), # TODO implement
        zbounds_len = (-1,1), # TODO implement
        pbounds = (0, 2*np.pi),
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
        # cl_type = 'binned', # TODO implement
        edges = 'fs',
        IMIN = 0,
        IMAX = 99,
        ITMAX = 10,
        fg = '00',
        base_mask = None,
        nlevels = None,
        nside = 2048,
        lmax_cl = 2048,
        beam = 1.0,
        lmax_transf = 4000,
        transf = 'gauss',
        Cl_fid = 'ffp10'
    )
)
