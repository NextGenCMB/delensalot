import numpy as np
import plancklens
from plancklens import utils
from delensalot.config.metamodel.dlensalot_mm import *

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
        TEMP_suffix = 'my_first_dlensalot_analysis_applyOBD',
        Lmin = 1, 
        lm_max_ivf = (1024, 1024),
        lmin_teb = (10, 10, 100),
        beam = 1.0,
        mask = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits'),
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'cl', 
        flavour = 'unl',
        lmax = 1024,
        phi_lmax = 1536,
        transfunction = gauss_beam(1.0/180/60 * np.pi, lmax=1024),
        nlev = {'P': np.sqrt(2)},
        geometry = ('healpix', {'nside': 512}),
        CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        OBD = True,
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        geometry = ('healpix', {'nside': 512}),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi"],
        filter_directional = 'anisotropic',
        qlm_type = 'sepTP',
        lm_max_qlm = (1024, 1024),
        cg_tol = 1e-5
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100'),
    )
)