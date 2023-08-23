import os
from os.path import join as opj
import numpy as np
import psutil

import delensalot
from delensalot import utils
from delensalot.utility.utils_hp import gauss_beam
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc
from delensalot.config.metamodel import DEFAULT_NotAValue, DEFAULT_NotASTR

DL_DEFAULT = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl',
        'lenjob_geominfo': ('thingauss',{'lmax': 4500, 'smax': 3}), 
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geominfo': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'spin': 0,
    },
    'analysis': { 
        'key': 'ptt',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'T_FS_CMBS4',
        'Lmin': 5, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024, 1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction_desc': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
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
        'cg_tol': 1e-6,
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
        'masks_fn': None,
        'lmax': 1024,
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
        'libdir': DEFAULT_NotAValue,
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
