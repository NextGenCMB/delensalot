"""
This config file resulted in Alens^MAP = 0.059 on average across bin 1 to 7 
"""

import numpy as np
import os
from os.path import join as opj

import healpy as hp

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import *

fg = '00'

def func(data):
    return data * 1e6 * np.nan_to_num(utils.cli(hp.read_map(opj(os.environ['SCRATCH'], 'data/cmbs4/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'))))

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'default_CMBS4_maskedsky_polarization',
    job = DLENSALOT_Job(
        jobs = ["build_OBD", "QE_lensrec", "MAP_lensrec"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 16
    ),
    obd = DLENSALOT_OBD(
        libdir = '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b',
        nlev_dep = 1e4,
        rescale = (0.42/0.3505)**2
    ),
    analysis = DLENSALOT_Analysis(
        key = 'p_p',
        version = '',
        simidxs = np.arange(0,32),
        simidxs_mf = np.arange(0,96),
        TEMP_suffix = 'check_dc08b_oldtniti_nozbound',
        Lmin = 3, 
        lm_max_ivf = (4000, 4000),
        lmin_teb = (30, 30, 200),
        zbounds = (-1,1),
        zbounds_len = (-1,1),
        beam = 2.3,
        mask = opj(os.environ['SCRATCH'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_mask_from_rhits.fits'),
    ),
    simulationdata = DLENSALOT_Simulation(
        space = 'map', 
        flavour = 'obs',
        lmax = 4050,
        nlev = {'P': 0.42, 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 2048}),
        libdir = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.{fg}_umilta_210511/'.format(fg=fg)),
        fns = {'Q':'cmbs4_08b{fg}_cmb_b02_ellmin30_ellmax4050_map_2048_{{:04d}}.fits'.format(fg=fg), 'U':'cmbs4_08b{fg}_cmb_b02_ellmin30_ellmax4050_map_2048_{{:04d}}.fits'.format(fg=fg)},
        spin = 2,
        modifier = func,
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        OBD = 'OBD',
        nlev = {'P': 0.42, 'T': np.sqrt(1)},
        rhits_normalised = (opj(os.environ['SCRATCH'], 'data/cmbs4/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'), np.inf),
    ),
    qerec = DLENSALOT_Qerec(
        tasks = ["calc_phi", "calc_meanfield", "calc_blt"],
        filter_directional = 'anisotropic',
        lm_max_qlm = (4000, 4000),
        cg_tol = 1e-3
    ),
    itrec = DLENSALOT_Itrec(
        tasks = ["calc_phi", 'calc_blt'],
        filter_directional = 'anisotropic',
        itmax = 12,
        cg_tol = 1e-4,
        lm_max_unl = (4200, 4200),
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
        iterations = [12],
        nlevels = [2],
        masks_fn = [],
        lmax = 1024,
        basemap = 'lens_ffp10',
        Cl_fid = 'ffp10',
        libdir_it = None,
        binning = 'binned',
        spectrum_calculator = pospace,
    ),
)