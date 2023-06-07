"""
Full sky iterative delensing on simulated CMB polarization maps generated on the fly, inclusive of isotropic white noise.
QE and iterative reconstruction use isotropic filters, deproject B-modes l<200, and reconstruct unlensed CMB up to 3200.
Parameters not listed here default to 'P_FS_CMBS4'
"""

import numpy as np
import os
import delensalot
from os.path import join as opj

import delensalot.core.power.pospace as pospace
from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'P_FS_CMBS4',
    job = DLENSALOT_Job(
        jobs = ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    ),                          
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_fullsky',
        lm_max_ivf = (3000, 3000),
        beam = 1.0,
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'cl', 
        flavour = 'unl',
        lmax = 4096,
        phi_lmax = 5120,
        transfunction = gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        nlev = {'P': np.sqrt(10)},
        geometry = ('healpix', {'nside': 2048}),
        CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        nlev_t = np.sqrt(5),
        nlev_p = np.sqrt(10),
        geometry = ('healpix', {'nside': 2048}),
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        lm_max_qlm = (3000, 3000),
        cg_tol = 1e-5
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        itmax = 3,
        lm_max_unl = (3200, 3200),
        lm_max_qlm = (3000, 3000),
        cg_tol = 1e-5
    ),
    madel = DLENSALOT_Mapdelensing(
        data_from_CFS = False,
        edges = lc.cmbs4_edges,
        iterations = [2],
        masks_fn = [opj(os.environ['SCRATCH'], 'delensalot/generic/sims_cmb_len_lminB200_my_first_dlensalot_analysis_fullsky/mask.fits')],
        lmax = 1024,
        Cl_fid = 'ffp10',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    )
)