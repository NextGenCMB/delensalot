"""
Masked sky iterative delensing on simulated CMB polarization data generated on the fly, inclusive of isotropic white noise.
Here, delensing is done on one simulation set, the meanfield is calculated from 5 simulations.
The noise model is isotropic and white, and truncates T,E, and B modes at low multipoles.
QE and iterative reconstruction uses anisotropic filters. 
"""

import numpy as np
import os
from os.path import join as opj

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'default_CMBS4_maskedsky_polarization',
    job = DLENSALOT_Job(
        jobs = ["QE_lensrec", "MAP_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),                              
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = '',
        simidxs = np.arange(0,1),
        simidxs_mf = np.arange(0,5),
        TEMP_suffix = 'mfda_maskedsky',
        Lmin = 1, 
        lm_max_ivf = (3000, 3000),
        lmin_teb = (10, 10, 200),
        zbounds = ('mr_relative', 10.),
        zbounds_len = ('extend', 5.),
        beam = 1.0,
        mask = opj(os.environ['SCRATCH'], 'analysis/mfda_maskedsky_lminB200/mask.fits')
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'cl', 
        flavour = 'unl',
        lmax = 4096,
        phi_lmax = 5120,
        transfunction = gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 2048}),
        lenjob_geominfo = ('thingauss', {'lmax': 5120, 'smax': 3}),
        CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        OBD = 'trunc',
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        rhits_normalised = (opj(os.environ['SCRATCH'], 'analysis/mfda_maskedsky_lminB200/mask.fits'), np.inf)
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield", "calc_blt"],
        filter_directional = 'anisotropic',
        lm_max_qlm = (4000, 4000),
        cg_tol = 1e-3
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        filter_directional = 'anisotropic',
        itmax = 1,
        cg_tol = 1e-3,
        lm_max_unl = (3200, 3200),
        lm_max_qlm = (4000, 4000),
        stepper = DLENSALOT_Stepper(
            typ = 'harmonicbump',
            lmax_qlm = 4000,
            mmax_qlm = 4000,
            a = 0.5,
            b = 0.499,
            xa = 400,
            xb = 1500
        ),
    ),
    madel = DLENSALOT_Mapdelensing(
        data_from_CFS = False,
        edges = lc.cmbs4_edges,
        iterations = [1],
        masks_fn = [opj(os.environ['SCRATCH'], 'analysis/mfda_maskedsky_lminB200/mask.fits')],
        lmax = 1024,
        Cl_fid = 'ffp10',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    )
)