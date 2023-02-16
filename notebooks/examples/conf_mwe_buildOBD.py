import os
from os.path import join as opj

import numpy as np
from lenscarf.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_OBD"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),
    analysis = DLENSALOT_Analysis(
        mask = opj(os.environ['SCRATCH'], 'OBDmatrix', 'nside512', 'lmax1024', 'lcut100', 'small_mask', 'mask.fits'),
        lmin_teb = (10, 10, 100)
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'OBDmatrix', 'nside512', 'lmax1024', 'lcut100', 'small_mask', 'rhits.fits'), np.inf)
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'small_mask'),
        nside = 512,
        nlev_dep = 1e4,
        beam = 1,
        lmax = 1024
    )
)