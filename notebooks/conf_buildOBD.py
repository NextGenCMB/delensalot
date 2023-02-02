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
        TEMP_suffix = 'my_first_OBDmatrix',
        mask = '/global/cscratch1/sd/sebibel/dlensalot/lenscarf/generic/sims_cmb_len_lminB30_my_first_dlensalot_analysis_maskedsky/mask.fits'
    ),
    noisemodel = DLENSALOT_Noisemodel(
        sky_coverage = 'masked',
        spectrum_type = 'white',
        nlev_dep = 10000.,
        inf = 1e4,
        lmin_teb = (10, 10, 200),
        nlev_t = 1.00,
        nlev_p = np.sqrt(2),
        rhits_normalised = ('/global/cscratch1/sd/sebibel/dlensalot/lenscarf/generic/sims_cmb_len_lminB30_my_first_dlensalot_analysis_maskedsky/mask.fits', np.inf),
    ),
    obd = DLENSALOT_OBD(
        libdir = '',
        nside = 2048,
        nlev_dep = 1e4,
        tpl = 'template_dense'
    )
)