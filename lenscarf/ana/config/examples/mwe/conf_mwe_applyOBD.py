import numpy as np
from lenscarf.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = 'noMF'
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_applyOBD',
        Lmin = 2, 
        lm_max_ivf = (300, 300),
        mask = '/global/cscratch1/sd/sebibel/dlensalot/lenscarf/generic/sims_cmb_len_lminB30_my_first_dlensalot_analysis_maskedsky/mask.fits'
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 300,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['CSCRATCH'], 'generic_lmax300','nlevp_sqrt(2)')
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 300,
        nside = 128,
        transferfunction = 'gauss_no_pixwin'
    ), 
    noisemodel = DLENSALOT_Noisemodel(
        OBD = True,
        sky_coverage = 'masked',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 200),
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = ('/global/cscratch1/sd/sebibel/dlensalot/lenscarf/generic/sims_cmb_len_lminB30_my_first_dlensalot_analysis_maskedsky/mask.fits', np.inf),
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix')
    )
)