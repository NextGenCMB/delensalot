import os
from os.path import join as opj

import numpy as np
from lenscarf.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_OBD"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 2
    ),
    analysis = DLENSALOT_Analysis(
        mask = opj(os.environ['SCRATCH'], 'OBDmatrix', 'nside512_lmax1024_lcut100/mask.fits')
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 100),
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'OBDmatrix', 'nside512_lmax1024_lcut100/rhits.fits'), np.inf),
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix', 'nside512_lmax1024_lcut100'),
        nside = 512,
        nlev_dep = 1e4,
        beam = 1,
        lmax = 1024
    )
)