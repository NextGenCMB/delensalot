import numpy as np

from MSC import pospace
import delensalot
from delensalot import utils

from delensalot.lerepi.config.config_helper import LEREPI_Constants as lc
from delensalot.lerepi.core.metamodel.dlensalot_mm import *

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
        TEMP_suffix = 'workbench',
        Lmin = 2, 
        lm_max_ivf = (1024, 1024),
        lmin_teb = (10, 10, 100),
        mask = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits'),
    ),
    data = DLENSALOT_Data(
        package_ = 'delensalot',
        module_ = 'sims.generic',
        class_ = 'sims_cmb_len',
        class_parameters = {
            'lmax': 1024,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(delensalot.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
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
        OBD = True,
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'anisotropic',
        qlm_type = 'sepTP',
        lm_max_qlm = (1024, 1024),
        cg_tol = 1e-5
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100'),
    ),
    madel = DLENSALOT_Mapdelensing(
        data_from_CFS = False,
        edges = lc.cmbs4_edges,
        iterations = [0],
        masks_fn = [opj(os.environ['SCRATCH'], 'delensalot/generic/sims_cmb_len_lminB200_mfda_maskedsky/mask.fits')],
        lmax = 1000,
        Cl_fid = 'ffp10',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    ),
)