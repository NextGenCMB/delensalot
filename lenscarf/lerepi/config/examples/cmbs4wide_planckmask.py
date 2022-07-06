import os
from os.path import join as opj
import numpy as np
import healpy as hp

from MSC import pospace
from plancklens.sims import phas, planck2018_sims

from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        build_OBD = False,
        QE_lensrec = False,
        MAP_lensrec = True,
        map_delensing = True,
        inspect_result = False,
        OMP_NUM_THREADS = 16
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = 'planckmask_wmf',
        K = 'p_p',
        V = '',
        ITMAX = 12,
        nsims_mf = 10,
        zbounds =  (-1,1),
        zbounds_len = (-1,1),   
        pbounds = [0, 2*np.pi],
        LENSRES = 1.7,
        Lmin = 4, 
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
        IMAX = 10,
        package_ = 'plancklens',
        module_ = 'sims.maps',
        class_ = 'cmb_maps_nlev',
        class_parameters = {
            'sims_cmb_len': planck2018_sims.cmb_len_ffp10(),
            'cl_transf': hp.gauss_beam(1.0 / 180 / 60 * np.pi, lmax=4096),
            'nlev_t': 0.5/np.sqrt(2),
            'nlev_p': 0.5,
            'nside': 2048,
            'pix_lib_phas': phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside2048'), 3, (hp.nside2npix(2048),))
        },
        data_type = 'map',
        data_field = "qu",
        beam = 1.0,
        lmax_transf = 4000,
        nside = 2048
    ),
    noisemodel = DLENSALOT_Noisemodel(
        typ = 'trunc',
        ninvjob_geometry = 'healpix_geometry',
        lmin_tlm = 30,
        lmin_elm = 30,
        lmin_blm = 200,
        CENTRALNLEV_UKAMIN = 0.5,
        nlev_t = 0.5/np.sqrt(2),
        nlev_p = 0.5,
        mask = opj(os.environ['CFS'], "cmb/data/planck2018/pr3/Planck_L08_inputs/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz")
    ),
    qerec = DLENSALOT_Qerec(
        ivfs = 'sepTP',
        qlms = 'sepTP',
        cg_tol = 1e-3,
        ninvjob_qe_geometry = 'healpix_geometry_qe',
        lmax_qlm = 4000,
        mmax_qlm = 4000,
        QE_LENSING_CL_ANALYSIS = True,
        chain = DLENSALOT_Chaindescriptor(
            p0 = 0,
            p1 = ["diag_cl"],
            p2 = None,
            p3 = 2048,
            p4 = np.inf,
            p5 = None,
            p6 = 'tr_cg',
            p7 = 'cache_mem'
    )),
    itrec = DLENSALOT_Itrec(
        filter = 'opfilt_ee_wl.alm_filter_ninv_wl',
        tasks = ["calc_phi", "calc_meanfield", "calc_btemplate"],
        cg_tol = 3,
        lenjob_geometry = 'thin_gauss',
        lenjob_pbgeometry = 'pbdGeometry',
        iterator_typ = 'pertmf',
        mfvar = '',
        soltn_cond = lambda it: True,
        stepper = DLENSALOT_Stepper(
            typ = 'harmonicbump',
            xa = 400,
            xb = 1500
    )),
    madel = DLENSALOT_Mapdelensing(
        iterations = [8,10],
        edges = ['ioreco'], # overwritten when binning=unbinned
        masks = ("masks", [opj(os.environ['CFS'], "cmb/data/planck2018/pr3/Planck_L08_inputs/PR3vJan18_temp_lensingmask_gPR2_70_psPR2_143_COT2_smicadx12_smicapoldx12_psPR2_217_sz.fits.gz")]),
        lmax = 2048, # automatically set to 200 when binning=unbinned
        Cl_fid = 'ffp10',
        binning = 'unbinned',
        spectrum_calculator = pospace
    ))