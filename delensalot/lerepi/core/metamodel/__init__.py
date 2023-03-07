DEFAULT_NotAValue = -123456789
DEFAULT_NotValid = 9876543210123456789

DL_DEFAULT_T = {
    'meta': {
        'version': "0.1"
    },
    'data': {
        'beam': 10,
        'nlev_t': 10,
        'nlev_p': 10,
    }
}

DL_DEFAULT_P = {
    'meta': {
        'version': "0.2"
    },
    'data': {
        'beam': 10,
        'nlev_t': 10,
        'nlev_p': 10,
    }
}

DL_DEFAULT_CMBS4_FS_P = {
    'meta': {
        'version': "0.2"
    },
    'data': {
        'beam': 10,
        'nlev_t': 10,
        'nlev_p': 10,
    }
}



DL_DEFAULT = dict({
    "T": DL_DEFAULT_T,
    "P": DL_DEFAULT_P,
    "default": DL_DEFAULT_P,
    "FS_CMB-S4_Pol": DL_DEFAULT_CMBS4_FS_P,
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
        'transferfunction': DEFAULT_NotValid
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
        'lensres': DEFAULT_NotValid, 
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
        'OMP_NUM_THREADS'
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