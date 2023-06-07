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
from delensalot.utility.utils_hp import gauss_beam
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
    'simulationdata': {
        'space': 'cl', 
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 2048}),
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
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-3,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0/np.sqrt(2),
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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

DL_DEFAULT_CMBS4_FS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl',
        'phi_space': 'cl', 
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.},
        'geometry': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
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
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0/np.sqrt(2),
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
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

DL_DEFAULT_CMBS4_FS_TP = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl', 
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'analysis': { 
        'key': 'p_tp',
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
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'isotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
    },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
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

DL_DEFAULT_CMBS4_MS_T = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl',
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'analysis': { 
        'key': 'ptt',
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
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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

DL_DEFAULT_CMBS4_MS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_MS_CMBS4',
        'Lmin': 1, 
        'lm_max_ivf': (4000, 4000),
        'lm_max_blt': (1024, 1024),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1, 1),
        'zbounds_len': (-1, 1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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

DL_DEFAULT_CMBS4_MS_TP = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 4096,
        'phi_lmax': 5120,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=4096),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 2048}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'analysis': { 
        'key': 'p_tp',
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
        'transfunction': 'gauss_no_pixwin',
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
        'geometry': ('healpix',{'nside': 2048}),
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
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 4500, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'epsilon': 1e-7,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1.0,
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 2048}),
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
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
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

DL_DEFAULT_TEST_FS_T = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'analysis': { 
        'key': 'ptt',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'T_FS_TEST',
        'Lmin': 1, 
        'lm_max_ivf': (2048, 2048),
        'lm_max_blt': (512, 512),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (2048, 2048),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction': 'gauss_no_pixwin',
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 2048,
        'phi_lmax': 3160,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=2048),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 1024}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-5,
        'filter_directional': 'isotropic',
        'lm_max_qlm': (3000, 3000),
        'cl_analysis': False,
        'blt_pert': True,
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 1024,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },
    },
    'itrec': {
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 3,
        'cg_tol': 1e-5,
        'epsilon': 1e-7,
        'iterator_typ': 'fastWF',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 2248, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (2248, 2248),
        'lm_max_qlm': (3000, 3000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'stepper':{
            'typ': 'harmonicbump',
            'lmax_qlm': 3000,
            'mmax_qlm': 3000,
            'a': 0.5,
            'b': 0.499,
            'xa': 400,
            'xb': 1500
        },
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 1024,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },      
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1./np.sqrt(2),
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 1024}),
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
        'lmax': 512,
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
    },
    'computing': {
        'OMP_NUM_THREADS': int(psutil.cpu_count()) #2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
    },
    'obd': {
        'libdir': DEFAULT_NotAValue,
        'rescale': 1,
        'tpl': 'template_dense',
        'nlev_dep': 1e4,
        'nside': 1024,
        'lmax': 200,
        'beam': 1.0,
    },
    'config': {
        'outdir_plot_root': opj(os.environ['HOME'], 'plots'),
        'outdir_plot_rel': ''
    }
}

DL_DEFAULT_TEST_FS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_TEST',
        'Lmin': 1, 
        'lm_max_ivf': (1536, 1536),
        'lm_max_blt': (512, 512),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (1536, 1536),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction': 'gauss_no_pixwin',
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 1536,
        'phi_lmax': 1536,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=1536),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 1024}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-5,
        'filter_directional': 'isotropic',
        'lm_max_qlm': (1536, 1536),
        'cl_analysis': False,
        'blt_pert': True,
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 512,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },
    },
    'itrec': {
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 3,
        'cg_tol': 1e-5,
        'epsilon': 1e-7,
        'iterator_typ': 'fastWF',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 1536, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (1536, 1536),
        'lm_max_qlm': (1536, 1536),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'stepper':{
            'typ': 'harmonicbump',
            'lmax_qlm': 1536,
            'mmax_qlm': 1536,
            'a': 0.5,
            'b': 0.499,
            'xa': 400,
            'xb': 1500
        },
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 512,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },      
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1./np.sqrt(2),
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 1024}),
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
        'lmax': 512,
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
    },
    'computing': {
        'OMP_NUM_THREADS': int(psutil.cpu_count()) #2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
    },
    'obd': {
        'libdir': DEFAULT_NotAValue,
        'rescale': 1,
        'tpl': 'template_dense',
        'nlev_dep': 1e4,
        'nside': 1024,
        'lmax': 200,
        'beam': 1.0,
    },
    'config': {
        'outdir_plot_root': opj(os.environ['HOME'], 'plots'),
        'outdir_plot_rel': ''
    }
}

DL_DEFAULT_TEST_FS_TP = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'analysis': { 
        'key': 'p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'TP_FS_TEST',
        'Lmin': 1, 
        'lm_max_ivf': (2048, 2048),
        'lm_max_blt': (512, 512),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (2048, 2048),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
        'transfunction': 'gauss_no_pixwin',
    },
    'simulationdata': {
        'space': 'cl', 
        'phi_space': 'cl',
        'flavour': 'unl',
        'lmax': 2048,
        'phi_lmax': 3160,
        'transfunction': gauss_beam(1.0/180/60 * np.pi, lmax=2048),
        'nlev': {'P': 1.0, 'T': 1./np.sqrt(2)},
        'geometry': ('healpix',{'nside': 1024}),
        'phi_field': 'potential',
        'CMB_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'phi_fn': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'epsilon': 1e-7,
        'spin': 0,
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-5,
        'filter_directional': 'isotropic',
        'lm_max_qlm': (3000, 3000),
        'cl_analysis': False,
        'blt_pert': True,
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 1024,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },
    },
    'itrec': {
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 3,
        'cg_tol': 1e-5,
        'epsilon': 1e-7,
        'iterator_typ': 'fastWF',
        'filter_directional': 'isotropic',
        'lenjob_geometry': ('thingauss',{'lmax': 2248, 'smax': 3}),
        'lenjob_pbdgeometry': ('pbd', (0., 2*np.pi)),
        'lm_max_unl': (2248, 2248),
        'lm_max_qlm': (3000, 3000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        'stepper':{
            'typ': 'harmonicbump',
            'lmax_qlm': 3000,
            'mmax_qlm': 3000,
            'a': 0.5,
            'b': 0.499,
            'xa': 400,
            'xb': 1500
        },
        'chain': {
            'p0': 0,
            'p1': ["diag_cl"],
            'p2': None,
            'p3': 1024,
            'p4': np.inf,
            'p5': None,
            'p6': 'tr_cg',
            'p7': 'cache_mem'
        },      
        },
    'noisemodel': {
        'sky_coverage': 'unmasked',
        'spectrum_type': 'white',
        'OBD': False,
        'nlev_t': 1./np.sqrt(2),
        'nlev_p': 1.0,
        'rhits_normalised': None,
        'geometry': ('healpix',{'nside': 1024}),
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
        'lmax': 512,
        'lmax_mask': lc.cmbs4_edges[-1],
        'Cl_fid': 'ffp10',
        'libdir_it': None,
        'binning': 'binned',
        'spectrum_calculator': pospace,
        'basemap': 'lens'
    },
    'computing': {
        'OMP_NUM_THREADS': int(psutil.cpu_count()) #2*int(psutil.cpu_count()/psutil.cpu_count(logical=False))
    },
    'obd': {
        'libdir': DEFAULT_NotAValue,
        'rescale': 1,
        'tpl': 'template_dense',
        'nlev_dep': 1e4,
        'nside': 1024,
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
    "T_MS_CMBS4": DL_DEFAULT_CMBS4_MS_T,
    "P_FS_CMBS4": DL_DEFAULT_CMBS4_FS_P,
    "P_MS_CMBS4": DL_DEFAULT_CMBS4_MS_P,
    "TP_FS_CMBS4": DL_DEFAULT_CMBS4_FS_TP,
    "P_MS_CMBS4": DL_DEFAULT_CMBS4_MS_TP,
    "T_FS_TEST": DL_DEFAULT_TEST_FS_T,
    "P_FS_TEST": DL_DEFAULT_TEST_FS_P,
    "TP_FS_TEST": DL_DEFAULT_TEST_FS_TP,
})