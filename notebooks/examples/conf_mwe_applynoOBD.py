import numpy as np
import plancklens
from plancklens import utils
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
        version = 'noMF',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_applynoOBD',
        Lmin = 2, 
        lm_max_ivf = (1024, 1024),
        lmin_teb = (10, 10, 100),
        mask = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits'),
    ),
    data = DLENSALOT_Data(
        package_ = 'lenscarf',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 1024,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(plancklens.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside512', 'lmax1024', 'nlevp_sqrt(2)'),
            'nside_lens': 512
        },
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        beam = 1.00,
        lmax_transf = 1024,
        nside = 512,
        transferfunction = 'gauss_no_pixwin'
    ), 
    noisemodel = DLENSALOT_Noisemodel(
        OBD = False,
        sky_coverage = 'isotropic',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi"],
        filter_directional = 'anisotropic',
        qlm_type = 'sepTP',
        lm_max_qlm = (1024, 1024),
        cg_tol = 1e-3
    )
)