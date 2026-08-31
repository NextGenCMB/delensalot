import os
from os.path import join as opj
import numpy as np
import psutil

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV, DEFAULT_NotASTR

# ---- band structure  ----
# LMAX_SKY   : the band the observed data is delivered at; must match analysis.lm_max_sky
# LMAX_DRAW  : unlensed CMB draw band. Headroom above LMAX_SKY so that lensed modes
#              near the band edge receive the power deflected down from above.
# LMAX_SEC   : secondary (phi/omega/beta) band. phi needs headroom for the deflection;
#              beta is a pure rotation and needs none beyond the analysis band.
# LMAX_GEOM  : quadrature geometry. Must exceed the largest band-limit acting on it,
#              with margin for products (rotation, remapping) and spin-raising, i.e. larger than LMAX_DRAW is a must! (I've seen QE completely failing when they match)
LMAX_SKY  = 5000
LMAX_DRAW = 6144 
LMAX_SEC  = 6144
LMAX_GEOM = LMAX_DRAW+1024 #LMAX_DRAW + 1024   # this must be larger than LMAX_DRAW!
GEOM_SIM  = ('thingauss', {'lmax': LMAX_GEOM, 'smax': 3})

DL_DEFAULT = {
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    # FIXME all lm_max need to be consistent no matter which flavour we start with.
    # better only have one lm_max in default and config file, and let l2p adapt accordingly?
    'data_source': {
        'gaussianized_sims': False,
        'flavour': 'pri',
        'libdir_suffix': 'generic',
        'geominfo': ('healpix', {'nside': 2048}),
        'fid_info': {
            'libdir': opj(os.path.dirname(delensalot.__file__), 'data', 'cls'),
            'fn': 'FFP10_wdipole_secondaries_lens_birefringence.dat',
            'libdir_sec': DNaV,
            'fn_sec': DNaV,
        },
        "CMB_info": {
            'space': 'cl',
            'libdir': DNaV,
            'fns': DNaV,
            'spin': 0,
            'lm_max': [LMAX_DRAW, LMAX_DRAW],
            'lm_max_sky': [LMAX_SKY, LMAX_SKY],
            'modifier': lambda x: x,
        },
        "sec_info": {
            'lensing': {
                'component': ['p', 'w'],
                'space': 'cl',
                'geominfo': GEOM_SIM,
                'LM_max': [LMAX_SEC, LMAX_SEC],
                'libdir': DNaV, 'fn': DNaV, 'scale': DNaV,
                'modifier': lambda x: x,
                'cl_modifier_factor': 1.0,
            },
            'birefringence': {
                'component': ['f'],
                'space': 'cl',
                'geominfo': GEOM_SIM,
                'LM_max': [LMAX_SKY, LMAX_SKY],
                'libdir': DNaV, 'fn': DNaV, 'scale': DNaV,
                'modifier': lambda x: x,
                'cl_modifier_factor': 4.0,
            },
        },
        "obs_info": {
            'noise_info': {
                'libdir': DNaV,
                'fns': DNaV,
                'nlev': {'P': 0.5, 'T': 0.5/np.sqrt(2)},
                'space': 'alm',
                'geominfo': ('healpix', {'nside': 2048}),
                'lm_max': [LMAX_SKY, LMAX_SKY],
            },
            'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=LMAX_SKY),
        },
        "operator_info": {
            'lensing': {
                'epsilon': 1e-12,
                'Lmin': 2,
                'lm_max': [LMAX_DRAW, LMAX_DRAW],
                'LM_max': [LMAX_SEC, LMAX_SEC],
                'lm_max_obs': [LMAX_SKY, LMAX_SKY],
                'geominfo': GEOM_SIM,
                'perturbative': False,
            },
            'birefringence': {
                'Lmin': 2,
                'lm_max': [LMAX_DRAW, LMAX_DRAW],
                'LM_max': [LMAX_SKY, LMAX_SKY],
                'lm_max_obs': [LMAX_SKY, LMAX_SKY],
                'geominfo': GEOM_SIM,
            },
        },
        'fixed_secondary_seed': None,
        'operator_order': ['birefringence', 'lensing'],
    },
    'analysis': { 
        'estimator_key': 'pwf_p',
        'idxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_CMBS4_jointsecrec',
        'Lmin': {'p':2, 'w': 2, 'f': 1}, 
        'LM_max': (4200, 4200), # NOTE this is max reconstructed secondary
        'lm_max_pri': (4000, 4000), # NOTE this is for CMB
        'lm_max_sky': (4000, 4000), # NOTE this is for CMB
        'lmin_teb': (2, 2, 200),
        'idxs_mf': [],
        'mask_fn': None,
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls_secondaries_lens_birefringence.dat'),
        'beam_FWHM': 1.0,
        'transfer_has_pixwindow': False,
                'secondary': {
            'lensing': {
                'geominfo': GEOM_SIM,
                'component': ['p', 'w'],
                'epsilon': 1e-12,
            },
            'birefringence': {
                'geominfo': GEOM_SIM,
                'component': ['f'],
            },
        },
        'seclist_sorted': ['birefringence', 'lensing'],
    },
    'qerec':{
        'tasks': ['calc_fields'],
        'TP_strategy': 'separate',
        'cg_tol': 1e-7,
        "subtract_QE_meanfield": True,
    },
    'maprec': {
        'tasks': ['calc_fields'],
        'itmax': 1,
        'cg_tol': 1e-7,
        'mfvar': '',
        'soltn_cond': lambda it: True,
        "use_QE_starting_point": True,
        "use_QE_for_lowL": False,
    },
    'noisemodel': {
        'spatial_type': 'isotropic',
        'spectrum_type': 'white',
        'OBD': 'trunc',
        'nlev': {'P': .5, 'T': 0.5/np.sqrt(2)},
        'rhits_normalised': None,
        'geominfo': ('healpix', {'nside': 2048}),
        'nivt_map': None,
        'nivp_map': None,
    },
    'madel': {
        'data_from_CFS': False,
        'edges': lc.cmbs4_edges,
        'nlevels': [np.inf],
        'dlm_mod': False,
        'iterations': [5],
        'masks_fn': [],
        'lmax': 1024,
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': "",
        'basemap': 'lens'
    },
    'phana': {
        'custom_WF_TEMP': None,
    },
    'computing': {
        'OMP_NUM_THREADS': np.max([0, int(psutil.cpu_count())-2]) #2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
    },
    'obd': {
        'libdir': DNaV,
        'rescale': 1,
        'tpl': 'template_dense',
        'nlev_dep': 1e4,
        'nside': 2048,
        'lmax': 200,
        'beam': 1.0,
    }
}