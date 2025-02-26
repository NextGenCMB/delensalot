import os
from os.path import join as opj
import numpy as np
import psutil

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV, DEFAULT_NotASTR

DL_DEFAULT = {
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    # FIXME all lm_max need to be consistent no matter which flavour we start with.
    # better only have one lm_max in default and config file, and let l2p adapt accordingly?
    'data_source': { 
        'flavour': 'pri',
        'libdir_suffix': 'generic',
        'geominfo': ('healpix',{'nside': 2048}), # NOTE this is the geometry for any map generated as the final result
        'maps': DNaV,
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
            'lm_max': [4096,4096],
            'modifier': lambda x: x,
        },
        "sec_info": { # NOTE if secondary already generated (space not 'cl'), this is needed. Otherwise, only the 'space' key is needed.
            'lensing':{
                'component': ['p','w'],
                'space': 'cl',
                'geominfo': ('thingauss', {'lmax': 4500, 'smax': 3}), # NOTE this is the geometry of the provided ssecondary maps
                'libdir': DNaV,
                'fn': DNaV,
                'scale': DNaV,
                'modifier': lambda x: x,
            },
            'birefringence':{
                'component': ['f'],
                'space': 'cl',
                'geominfo': ('thingauss', {'lmax': 4500, 'smax': 3}),
                'libdir': DNaV,
                'fn': DNaV,
                'scale': DNaV,
                'modifier': lambda x: x,
            },
        },
        "obs_info": {
            'noise_info': {
                'libdir': DNaV,
                'fns': DNaV,
                'nlev': {'P': 0.5, 'T': 0.5/np.sqrt(2)},
                'space': 'alm',
                'geominfo': ('healpix',{'nside': 2048}),
                'lm_max': [4096,4096],
            },
            'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
            },
        "operator_info": {
            'lensing': {
                'epsilon': 1e-7,
                'Lmin': 2,
                'lm_max': [4096,4096],
                'LM_max': [4096+1024,4096+1024],
                'lm_max_obs': [4096,4096],
                'geominfo': ('thingauss',{'lmax': 4500, 'smax': 3}),
                'perturbative': False,
            },
            'birefringence': {
                'Lmin': 2,
                'lm_max': [4096,4096],
                'LM_max': [4096,4096],
                'lm_max_obs': [4096,4096],
                'geominfo': ('thingauss',{'lmax': 4500, 'smax': 3}),
            },
        }
    },
    'analysis': { 
        'key': 'pwf_p',
        'idxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_CMBS4_jointsecrec',
        'Lmin': {'p':2, 'w': 2, 'f': 2}, 
        'LM_max': (4200, 4200), # NOTE this is max reconstructed secondary
        'lm_max_pri': (4000, 4000), # NOTE this is for CMB
        'lm_max_sky': (4000, 4000), # NOTE this is for CMB
        'lmin_teb': (2, 2, 200),
        'idxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'mask_fn': None,
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls_secondaries_lens_birefringence.dat'),
        'beam': 1.0,
        'transfunction_desc': 'gauss_no_pixwin',
        'secondary': {
            'lensing': {
                'geominfo': ('thingauss', {'lmax': 4500, 'smax': 3}),
                'LM_max': (4200, 4200), # NOTE this overwrites the global lm_max_sec
                'lm_max_pri': (4000, 4000), # NOTE this overwrites the global lm_max_pri
                'lm_max_sky': (4000, 4000), # NOTE this overwrites the global lm_max_sky
                'component': ['p', 'w'],
                'epsilon': 1e-7,
            },
            'birefringence': {
                'geominfo': ('thingauss', {'lmax': 4500, 'smax': 3}),
                'LM_max': (4200, 4200), # NOTE this overwrites the global lm_max_sec
                'lm_max_pri': (4000, 4000), # NOTE this overwrites the global lm_max_pri
                'lm_max_sky': (4000, 4000), # NOTE this overwrites the global lm_max_sky
                'component': ['f'],
            },
        },
    },
    'qerec':{
        'tasks': ['calc_fields'],
        'estimator_type': 'sepTP',
        'cg_tol': 1e-7,
        "subtract_QE_meanfield": True,
    },
    'maprec': {
        'tasks': ['calc_fields'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'mfvar': '',
        'soltn_cond': lambda it: True,
    },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spatial_type': 'isotropic',
        'spectrum_type': 'white',
        'OBD': 'trunc',
        'nlev': {'P': .5, 'T': 0.5/np.sqrt(2)},
        'rhits_normalised': None,
        'geominfo': ('healpix',{'nside': 2048}),
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
        'spectrum_calculator': pospace,
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