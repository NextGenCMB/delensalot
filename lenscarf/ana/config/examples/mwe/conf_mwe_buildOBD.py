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
        mask = opj(os.environ['SCRATCH'],'OBDmatrix_nside128_lmax200/mask.fits')
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        lmin_teb = (10, 10, 200),
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = (opj(os.environ['SCRATCH'],'OBDmatrix_nside128_lmax200/rhits.fits'), np.inf),
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'OBDmatrix_nside128_lmax200'),
        nside = 128,
        nlev_dep = 1e4,
        beam = 1,
        lmax = 200
    )
)