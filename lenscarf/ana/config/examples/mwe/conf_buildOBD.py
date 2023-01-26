import numpy as np
from dlensalot.lerepi.core.metamodel.dlensalot_mm import *

dlensalot_model = DLENSALOT_Model(
    job = DLENSALOT_Job(
        jobs = ["build_OBD"]
    ),
    computing = DLENSALOT_Computing(
        OMP_NUM_THREADS = 4
    ),
    analysis = DLENSALOT_Analysis(
        TEMP_suffix = 'my_first_OBDmatrix',
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        BMARG_LCUT = 200,
        nlev_dep = 10000.,
        inf = 1e4,
        lmin_teb = (10, 10, 200),
        nlev_t = 0.25/np.sqrt(2),
        nlev_p = 0.25,
        mask = ('nlev', np.inf),
        BMARG_LIBDIR = '<>',
        rhits_normalised = ('/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits', np.inf),
        tpl = 'template_dense'
    )
)