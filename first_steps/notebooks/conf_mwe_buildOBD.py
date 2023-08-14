import os
from os.path import join as opj

import numpy as np
import delensalot
from delensalot.config.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_OBD"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),
    analysis = DLENSALOT_Analysis(
        mask = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'mask.fits'),
        lmin_teb = (10, 10, 100)
    ),
    noisemodel = DLENSALOT_Noisemodel(
        OBD = 'OBD',
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev = {'P': np.sqrt(2), 'T': np.sqrt(1)},
        geominfo = ('healpix', {'nside': 512}),
        rhits_normalised = (opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100', 'rhits.fits'), np.inf)
    ),
    obd = DLENSALOT_OBD(
        libdir = opj(os.environ['SCRATCH'], 'analysis', 'OBDmatrix', 'my_first_dlensalot_analysis', 'nside512', 'lmax1024', 'lcut100'),
        nlev_dep = 1e4,
    )
)