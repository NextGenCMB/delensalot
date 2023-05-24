#!/usr/bin/env python

"""metamodel.__init__.py:
    The metamodel defines the available attributes and their valid values a delensalot configuration file can have.
    The init file is mainly a collection of default values for delensalot model.
    These dictionaries are, depending on the configuration, loaded at instantiation of a delensalot model.
"""

import os
from os.path import join as opj
import numpy as np
import psutil

import delensalot
from delensalot import utils
import delensalot.core.power.pospace as pospace
from delensalot.config.config_helper import LEREPI_Constants as lc

DEFAULT_NotAValue = -123456789
DEFAULT_NotASTR = '.-.,.-.,'
DEFAULT_NotValid = 9876543210123456789

DL_DEFAULT_CMBS4_FS_T = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-5,
        'nside': 2048,
        'class_parameters': {
            'lmax': 4096,
            'cls_unl': utils.camb_clfile(opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4096', 'nlevp_sqrt(2)')
        },
        'lmax_transf': 4500,
        'transferfunction': 'gauss_no_pixwin',
        'package_': 'delensalot', 
        'module_': 'sims.generic', 
        'class_': 'sims_cmb_len', 
    },
    'analysis': { 
        'key': 'ptt',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'T_FS_CMBS4',
        'Lmin': 1, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024,1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
    },
    'qerec':{
        'tasks': ['calc_phi'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'ninvjob_qe_geometry': 'healpix_geometry_qe',
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
        'tasks': ['calc_phi'],
        'itmax': 1,
        'cg_tol': 1e-3,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'isotropic',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'ninvjob_geometry': 'healpix_geometry',
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
    },
    'computing': {
        'OMP_NUM_THREADS': 2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
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

DL_DEFAULT_CMBS4_FS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-5,
        'nside': 2048,
        'class_parameters': {
            'lmax': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(delensalot.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4096', 'nlevp_sqrt(2)')
        },
        'lmax_transf': 4500,
        'transferfunction': 'gauss_no_pixwin',
        'package_': 'delensalot', 
        'module_': 'sims.generic', 
        'class_': 'sims_cmb_len', 
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_CMBS4',
        'Lmin': 1, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024,1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
    },
    'qerec':{
        'tasks': ['calc_phi'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'ninvjob_qe_geometry': 'healpix_geometry_qe',
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
        'tasks': ['calc_phi'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'isotropic',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'ninvjob_geometry': 'healpix_geometry',
    },
    'madel': {
        'data_from_CFS': False,
        'edges': lc.cmbs4_edges,
        'nlevels': [np.inf],
        'dlm_mod': False,
        'iterations': [5],
        'masks_fn': None,
        'lmax': 1024,
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
    },
    'computing': {
        'OMP_NUM_THREADS': 2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
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

DL_DEFAULT_CMBS4_MS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-5,
        'nside': 2048,
        'class_parameters': {
            'lmax': 4096,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(delensalot.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside2048', 'lmax4096', 'nlevp_sqrt(2)')
        },
        'lmax_transf': 4500,
        'transferfunction': 'gauss_no_pixwin',
        'package_': 'delensalot', 
        'module_': 'sims.generic', 
        'class_': 'sims_cmb_len', 
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_MS_CMBS4',
        'Lmin': 1, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024,1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
    },
    'qerec':{
        'tasks': ['calc_phi'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'ninvjob_qe_geometry': 'healpix_geometry_qe',
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
        'tasks': ['calc_phi'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'isotropic',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'ninvjob_geometry': 'healpix_geometry',
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
    },
    'computing': {
        'OMP_NUM_THREADS': 2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
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

DL_DEFAULT = dict({
    "T_FS_CMBS4": DL_DEFAULT_CMBS4_FS_T,
    "P_FS_CMBS4": DL_DEFAULT_CMBS4_FS_P,
    "P_MS_CMBS4": DL_DEFAULT_CMBS4_MS_P,
    })

DL_DEFAULT_TEMPLATE  = {
    'defaults_to': DEFAULT_NotValid,
    'meta': {
        'version': DEFAULT_NotValid,
        },
    'job': {
        'jobs': DEFAULT_NotValid,
    },
    'analysis': {
        'Lmin': DEFAULT_NotValid, 
        'TEMP_suffix': DEFAULT_NotValid, 
        'beam': DEFAULT_NotValid, 
        'cls_len': DEFAULT_NotValid, 
        'cls_unl': DEFAULT_NotValid, 
        'cpp': DEFAULT_NotValid, 
        'key': DEFAULT_NotValid, 
        'lm_max_ivf': DEFAULT_NotValid, 
        'lm_max_len': DEFAULT_NotValid, 
        'lmin_teb': DEFAULT_NotValid, 
        'mask': DEFAULT_NotValid, 
        'pbounds': DEFAULT_NotValid, 
        'simidxs': DEFAULT_NotValid, 
        'simidxs_mf': DEFAULT_NotValid, 
        'version': DEFAULT_NotValid, 
        'zbounds': DEFAULT_NotValid, 
        'zbounds_len': DEFAULT_NotValid, 
    },
    'data': {
        'beam': DEFAULT_NotValid, 
        'class_': DEFAULT_NotValid, 
        'class_parameters': DEFAULT_NotValid, 
        'lmax_transf': DEFAULT_NotValid, 
        'module_': DEFAULT_NotValid, 
        'nlev_p': DEFAULT_NotValid, 
        'nlev_t': DEFAULT_NotValid, 
        'nside': DEFAULT_NotValid, 
        'package_': DEFAULT_NotValid, 
        'transf_dat': DEFAULT_NotValid, 
        'transferfunction': DEFAULT_NotValid,
        'epsilon': DEFAULT_NotValid
    },
    'noisemodel': {
        'OBD': DEFAULT_NotValid, 
        'ninvjob_geometry': DEFAULT_NotValid, 
        'nlev_p': DEFAULT_NotValid, 
        'nlev_t': DEFAULT_NotValid, 
        'rhits_normalised': DEFAULT_NotValid, 
        'sky_coverage': DEFAULT_NotValid, 
        'spectrum_type': DEFAULT_NotValid
    },
    'qerec': {
        'blt_pert': DEFAULT_NotValid, 
        'cg_tol': DEFAULT_NotValid, 
        'chain': DEFAULT_NotValid, 
        'cl_analysis': DEFAULT_NotValid, 
        'filter_directional': DEFAULT_NotValid, 
        'lm_max_qlm': DEFAULT_NotValid, 
        'ninvjob_qe_geometry': DEFAULT_NotValid, 
        'qlm_type': DEFAULT_NotValid, 
        'tasks': DEFAULT_NotValid
    },
    'itrec': {
        'cg_tol': DEFAULT_NotValid, 
        'filter_directional': DEFAULT_NotValid, 
        'iterator_typ': DEFAULT_NotValid, 
        'itmax': DEFAULT_NotValid, 
        'lenjob_geometry': DEFAULT_NotValid, 
        'lenjob_pbgeometry': DEFAULT_NotValid, 
        'lm_max_qlm': DEFAULT_NotValid, 
        'lm_max_unl': DEFAULT_NotValid, 
        'mfvar': DEFAULT_NotValid, 
        'soltn_cond': DEFAULT_NotValid, 
        'stepper': DEFAULT_NotValid, 
        'tasks': DEFAULT_NotValid
    },
    'madel': {
        'Cl_fid': DEFAULT_NotValid, 
        'binning': DEFAULT_NotValid, 
        'dlm_mod': DEFAULT_NotValid, 
        'edges': DEFAULT_NotValid, 
        'iterations': DEFAULT_NotValid, 
        'libdir_it': DEFAULT_NotValid, 
        'lmax': DEFAULT_NotValid, 
        'masks': DEFAULT_NotValid, 
        'spectrum_calculator': DEFAULT_NotValid
    },
    'config': {
        'outdir_plot_rel': DEFAULT_NotValid, 
        'outdir_plot_root': DEFAULT_NotValid
    },
    'computing': {
        'OMP_NUM_THREADS': DEFAULT_NotValid,
    },
    'obd': {
        'beam': DEFAULT_NotValid, 
        'libdir': DEFAULT_NotValid, 
        'lmax': DEFAULT_NotValid, 
        'nlev_dep': DEFAULT_NotValid, 
        'nside': DEFAULT_NotValid, 
        'rescale': DEFAULT_NotValid, 
        'tpl': DEFAULT_NotValid
    }
}