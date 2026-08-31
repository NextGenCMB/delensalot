import numpy as np
import healpy as hp
import os
from os.path import join as opj

from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.metamodel.delensalot_mm import *

def func(data):
    return data * 1e6

delensalot_model = DELENSALOT_Model(
    defaults_to = 'default_jointrec',
    job = DELENSALOT_Job(
        jobs = ["generate_sim", "QE_lensrec", "MAP_lensrec"]
    ),
    analysis = DELENSALOT_Analysis(
        TEMP_suffix = 'jointrec_aniso_mwe_fullsky',
        estimator_key = 'pwf_p',
        idxs = np.arange(1),
        idxs_mf = np.array([]),
        beam_FWHM = 1.5,
        LM_max = (4000, 4000),
        lm_max_pri = (3000, 3000),
        lm_max_sky = (3000, 3000),
        Lmin = {'p': 2, 'w': 2, 'f': 2},
        lmin_teb = (30, 30, 200),
        transfer_has_pixwindow = False,
        operator_order = ['birefringence', 'lensing'],
    ),
    data_source = DELENSALOT_DataSource(
        flavour = 'pri',
        sec_info = {
            'lensing': {'component': ['p','w'],},
            'birefringence': {'component': ['f']},
        },
        obs_info = {
            'noise_info': {
                'nlev': {'P': 1.5, 'T': 1.5/np.sqrt(2)},
            },
            'transfunction': gauss_beam(1.5/180/60 * np.pi, lmax=4096),
        },
        operator_order = ['birefringence', 'lensing'],
        fixed_secondary_seed = None,
    ),
    noisemodel = DELENSALOT_Noisemodel(
        nlev = {'P': 1.5, 'T': 1.5/np.sqrt(2)},
        geominfo = ('healpix', {'nside': 2048}),
    ),
    qerec = DELENSALOT_QErec(
        tasks = ["calc_fields"],
        filtering_type = 'isotropic',
        cg_tol = 1e-6,
        subtract_QE_meanfield = False,
    ),
    maprec = DELENSALOT_MAPrec(
        tasks = ["calc_fields"],
        filtering_type = 'anisotropic',
        itmax = 5,
        cg_tol = 1e-6,
        use_QE_for_lowL = False,
        use_QE_starting_point = True,
    ),
    computing = DELENSALOT_Computing(
        OMP_NUM_THREADS = 32
    ),
)