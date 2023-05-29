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
        'epsilon': 1e-7,
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
        'tasks': ['calc_phi', 'calc_blt'],
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
        'tasks': ['calc_phi', 'calc_blt'],
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
        'sky_coverage': 'unmasked',
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
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-7,
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
        'tasks': ['calc_phi', 'calc_blt'],
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
        'tasks': ['calc_phi', 'calc_blt'],
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
        'sky_coverage': 'unmasked',
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
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-7,
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
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
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
        'tasks': ['calc_phi', 'calc_blt'],
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
        'sky_coverage': 'unmasked',
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
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-7,
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
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (4000, 4000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
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
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
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
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-7,
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
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
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
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
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
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': 1.,
        'epsilon': 1e-7,
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
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-4,
        'filter_directional': 'anisotropic',
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
        'tasks': ['calc_phi', 'calc_blt'],
        'itmax': 1,
        'cg_tol': 1e-5,
        'iterator_typ': 'constmf',
        'filter_directional': 'anisotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (4500,4500),
        'lm_max_qlm': (4000,4000),
        'mfvar': '',
        'soltn_cond': lambda it: True,
        },
    'noisemodel': {
        'sky_coverage': 'masked',
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

DL_DEFAULT_TEST_FS_P = {
    'meta': {
        'version': "0.2"
    },
    'job':{
        'jobs': ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    },
    'data': {
        'beam': 1.,
        'nlev_t': 1.,
        'nlev_p': np.sqrt(2),
        'epsilon': 1e-7,
        'nside': 1024,
        'class_parameters': {
            'lmax': 3000,
            'cls_unl': utils.camb_clfile(opj(opj(os.path.dirname(delensalot.__file__), 'data', 'cls'), 'FFP10_wdipole_lenspotentialCls.dat')),
            'lib_dir': opj(os.environ['SCRATCH'], 'sims', 'generic', 'nside1024', 'lmax3000', 'nlevp_sqrt(2)')
        },
        'lmax_transf': 3000,
        'transferfunction': 'gauss_no_pixwin',
        'package_': 'delensalot', 
        'module_': 'sims.generic', 
        'class_': 'sims_cmb_len', 
    },
    'analysis': { 
        'key': 'p_p',
        'version': 'noMF',
        'simidxs': np.arange(0,1),
        'TEMP_suffix': 'P_FS_TEST',
        'Lmin': 1, 
        'lm_max_ivf': (3000, 3000),
        'lm_max_blt': (512, 512),
        'lmin_teb': (2, 2, 200),
        'simidxs_mf': [],
        'zbounds': (-1,1),
        'zbounds_len': (-1,1),
        'pbounds': (0., 2*np.pi),
        'lm_max_len': (3000, 3000),
        'mask': None,
        'cls_unl': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'cls_len': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lensedCls.dat'),
        'cpp': opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'),
        'beam': 1.0,
    },
    'qerec':{
        'tasks': ['calc_phi', 'calc_blt'],
        'qlm_type': 'sepTP',
        'cg_tol': 1e-5,
        'filter_directional': 'isotropic',
        'ninvjob_qe_geometry': 'healpix_geometry_qe',
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
        'iterator_typ': 'fastWF',
        'filter_directional': 'isotropic',
        'lenjob_geometry': 'thin_gauss',
        'lenjob_pbgeometry': 'pbdGeometry',
        'lm_max_unl': (3200, 3200),
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
        'nlev_t': 1.0,
        'nlev_p': np.sqrt(2),
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
    "P_FS_TEST": DL_DEFAULT_TEST_FS_P,
})