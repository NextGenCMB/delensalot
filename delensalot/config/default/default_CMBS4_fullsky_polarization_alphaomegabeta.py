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
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'flavour': 'unl',
        'geominfo': ('healpix',{'nside': 2048}),
        'maps': DNaV,
        "CMB_info": {
            'libdir': '',
            'fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_secondaries_lens_birefringence.dat'),
            'space': 'cl',
            'spin': 0,
            'lm_max': [4096,4096],
            'fns': DNaV,
            'modifier': lambda x: x,
        },
        "sec_info": {
            'phi':{
                'libdir': DNaV,
                'fn': DNaV,
                'components': ['pp', 'ww'],
                'space':'alm',
                'scale':'p',
                'modifier': lambda x: x,
                'lm_max': [4096+1024,4096+1024],
            },
            'bf':{
                'libdir': DNaV,
                'fn': DNaV,
                'components': ['ff'],
                'space':'alm',
                'scale':'p',
                'modifier': lambda x: x,
                'lm_max': [4096+1024,4096+1024],
            },
        },
        "obs_info": {
                'noise_info': {
                    'libdir': DNaV,
                    'fns': DNaV,
                    'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
                    'space': 'alm',
                    'geominfo': ('healpix',{'nside': 2048}),
                    'libdir_suffix': 'generic',
                },
                'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
            },
        "operator_info": {
            'lensing': {
                'epsilon': 1e-7,
                'components': ['p', 'w'],
                'Lmin': 1,
                'lm_max': [4096,4096],
                'lm_max_field': [4096+1024,4096+1024],
                'geominfo': ('thingauss',{'lmax': 4500, 'smax': 3}),
                'perturbative': False,
                'tr': 8,
                'field_fns': DNaV,
                'libdir': DNaV,
            },
            'birefringence': {
                'lm_max': [4096,4096],
                'lm_max_field': [4096+1024,4096+1024],
                'Lmin': 1,
                'field_fns': DNaV,
                'libdir': DNaV,
            },
        }
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_CMBS4_aob',
        'Lmin': 1, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024,1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_secondaries_lens_birefringence.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_secondaries_lens_birefringence.dat'),
        'beam': 1.0,
        'transfunction_desc': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'estimator_type': 'sepTP',
        'cg_tol': 1e-7,
        'filter_directional': 'isotropic',
        'geominfo': ('healpix',{'nside': 2048}),
        'lm_max_qlm': (4000,4000),
        'cl_analysis': False,
        'blt_pert': True,
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 2048,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },
        "subtract_QE_meanfield": True,
    },
    'itrec': {
        'stepper':{
            'typ': 'harmonicbump',
            'lmax_qlm': 4000,
            'mmax_qlm': 4000,
            'a': 0.5,
            'b': 0.499,
            'xa': 400,
            'xb': 1500
        },
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 2048,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geominfo': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeominfo': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': 'trunc',
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
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
        'OMP_NUM_THREADS': int(psutil.cpu_count()) #2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
    },
    'obd': {
        'libdir': DNaV,
        'rescale': 1,
        'tpl': 'template_dense',
        'nlev_dep': 1e4,
        'nside': 2048,
        'lmax': 200,
        'beam': 1.0,
    },
    'config': {
        'outdir_plot_root': opj(os.environ['HOME'], 'plots'),
        'outdir_plot_rel': ''
    }
}