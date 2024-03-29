"""
Full sky iterative delensing on simulated CMB polarization maps generated on the fly, inclusive of isotropic white noise.
QE and iterative reconstruction use isotropic filters, deproject B-modes l<200, and reconstruct unlensed CMB up to 4200. iterative rec uses fastWF.
Parameters not listed here default to 'P_FS_CMBS4'
"""

import numpy as np
import os
from os.path import join as opj

import delensalot
import delensalot.core.power.pospace as pospace
from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'default_CMBS4_fullsky_polarization',
    job = DLENSALOT_Job(
        jobs = ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    ),                          
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        simidxs = np.arange(0,1),
        TEMP_suffix = 'my_first_dlensalot_analysis_fastWF',
        Lmin = 10,
        lm_max_ivf = (4000, 4000),
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'cl', 
        flavour = 'unl',
        lmax = 4096,
        phi_lmax = 5120,
        transfunction = gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 2048}),
        lenjob_geominfo = ('thingauss', {'lmax': 4200 + 300, 'smax': 3}),
        CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
    ),
    noisemodel = DLENSALOT_Noisemodel(
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 2048}),
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_blt"],
        lm_max_qlm = (4000, 4000),
        cg_tol = 1e-7
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", "calc_blt"],
        iterator_typ = 'fastWF',
        itmax = 5,
        lenjob_geominfo = ('thingauss', {'lmax': 4200 + 300, 'smax': 3}),
        lm_max_unl = (4200, 4200),
        lm_max_qlm = (4000, 4000),
        cg_tol = 1e-7
    ),
)