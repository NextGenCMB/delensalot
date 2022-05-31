import os
import numpy as np
import healpy as hp

from lenscarf.iterators import steps
from lenscarf import utils_scarf
from lerepi.data.dc08 import data_08d as if_s
from lerepi.core import helper

# DATA_LIBDIR: 
DATA_LIBDIR = '/global/project/projectdirs/cmbs4/awg/lowellbb/'

# BMARG_LIBDIR: 
BMARG_LIBDIR = os.path.join(DATA_LIBDIR, 'reanalysis/mapphi_intermediate/s08d/')

# BMARG_LCUT: 
BMARG_LCUT = 200

# THIS_CENTRALNLEV_UKAMIN: 
THIS_CENTRALNLEV_UKAMIN = 0.59 
nlev_t = THIS_CENTRALNLEV_UKAMIN
nlev_p = THIS_CENTRALNLEV_UKAMIN/np.sqrt(2)

# Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
# This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
QE_LENSING_CL_ANALYSIS = False

# Change the following block only if exotic transferfunctions are desired
STANDARD_TRANSFERFUNCTION = True

# Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
FILTER = 'cinv_sepTP'

# Change the following block only if exotic chain descriptor are desired
CHAIN_DESCRIPTOR = 'default'

# Change the following block only if other than sepTP for QE is desired
FILTER_QE = 'sepTP' 

# The following block defines various multipole limits. Change as desired
lmax_transf = 4000 # can be distinct from lmax_filt for iterations
lmax_filt = 4096 # unlensed CMB iteration lmax
lmin_tlm, lmin_elm, lmin_blm = (30, 30, 200)
lmax_qlm, mmax_qlm = (4000, 4000)
lmax_unl, mmax_unl = (4000, 4000)
lmax_ivf, mmax_ivf = (3000, 3000)
lmin_ivf, mmin_ivf = (10, 10)
lensres = 1.7  # Deflection operations will be performed at this resolution
Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted

# Meanfield, OBD, and tol settings
cg_tol = 1e-5
tol = 5
nsims_mf = 10
isOBD = True

# rhits:
rhits = hp.read_map(os.path.join(DATA_LIBDIR, 'expt_xx/08d/rhits/n2048.fits'))

# The following block defines which data and mask to use
fg = '00'
sims = if_s.ILC_May2022(fg,mask_suffix=2)
mask = sims.get_mask_path()

# The following block defines about which area the lensing will be performed. Zbounds is the latitude bounds, pbounds is for longitude
zbounds = helper.get_zbounds(hp.read_map(mask))
zbounds_len = helper.extend_zbounds(zbounds) # Outside of these bounds the reconstructed maps are assumed to be zero
pb_ctr, pb_extent = (0., 2 * np.pi) # Longitude cuts, if any, in the form (center of patch, patch extent)
lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(lmax_unl, 2, zbounds=zbounds_len)
lenjob_pbgeometry = utils_scarf.pbdGeometry(lenjob_geometry, utils_scarf.pbounds(pb_ctr, pb_extent))

# nside:
nside = 2048

# beam:
beam = 2.3

# transf:
transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)

# ninvjob_geometry:
ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(nside, zbounds=zbounds)

# stepper:
stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, xa=400, xb=1500) #reduce the gradient by 0.5 for large scale and by 0.1 for small scales to improve convergence in regimes where the deflection field is not invertible
