"""Script for calculating Btemplates using precalculated plms as input. Use 'Generate_plm_mm.py' for calculating plm input
"""
import os, sys

import healpy as hp
import numpy as np

from plancklens import utils
from plancklens.filt import filt_cinv
from plancklens.qcinv import cd_solve
from plancklens.helpers import mpi

from cmbs4 import sims_08b

import itercurv
from itercurv.iterators.statics import rec as Rec
from itercurv.remapping.utils import alm_copy
from itercurv.filt import utils_cinv_p

import argparse
parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=500, help='Maximal simulation index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
parser.add_argument('-bs', dest='blm_suffix', type=str, default='', help='Suffix string. Defines where the plms are found and where the blm will be stored.')
parser.add_argument('-it', dest='itid', type=str, default='QEMAP', help='iteration identifier, either QE, MAP or QEMAP')
args = parser.parse_args()

simid_lower = args.simid_lower
simid_upper = args.simid_upper
fg = args.fg
blm_suffix = args.blm_suffix
if args.itid == 'QE':
    iteration = [0]
elif args.itid == 'QEMAP':
    iteration = [0,12]
elif args.itid == 'MAP':
    iteration = [12]
simids = np.arange(simid_lower,simid_upper)

sims_may = sims_08b.caterinaILC_May12(fg)
sims_sep = sims_08b.caterinaILC_Sep12(fg)

BMARG_LCUT=200
BMARG_CENTRALNLEV_UKAMIN = 0.350500 # central pol noise level in map used to build the (TniT) inverse matrix
THIS_CENTRALNLEV_UKAMIN = 0.42# central pol noise level in this pameter file noise sims. The template matrix willbe rescaled
BMARG_LIBDIR  = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/'

tniti_rescal = (THIS_CENTRALNLEV_UKAMIN / BMARG_CENTRALNLEV_UKAMIN) ** 2
nlev_p = 0.42   #NB: cinv_p gives me this value cinv_p::noiseP_uk_arcmin = 0.429
nlev_t = nlev_p / np.sqrt(2.)
beam = 2.3
lmax_ivf_qe = 3000
lmin_ivf_qe = 10
lmax_qlm = 4096
lmax_transf = 4000 # can be distinct from lmax_filt for iterations
lmax_filt = 4096 # unlensed CMB iteration lmax
nside = 2048
tol=1e-3

nsims = 200

transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)
cls_path = os.path.join(os.path.dirname(itercurv.__file__), 'data', 'cls')
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
zbounds = sims_08b.get_zbounds(np.inf)

TEMPMFbase =  '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s/'%(fg)
TEMPMFbase_lmax =  '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/'%(fg)
libdir_cinvt = os.path.join(TEMPMFbase_lmax, 'cinvt')
libdir_cinvp = os.path.join(TEMPMFbase_lmax, 'cinvp')
libdir_ivfs = os.path.join(TEMPMFbase_lmax, 'ivfs')  


chain_descr = [[0, ["diag_cl"], lmax_ivf_qe, nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

ivmat_path = os.path.join(TEMPMFbase_lmax, 'itvmap.fits')
ninv_t = [ivmat_path]
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf_qe, nside, cls_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[], chain_descr=chain_descr)

ivmap_path = os.path.join(TEMPMFbase_lmax, 'ipvmap.fits')
ninv_p = [[ivmap_path]]
cinv_p_OBD = utils_cinv_p.cinv_p(libdir_cinvp.replace('cinvp', 'cinvpOBD'), lmax_ivf_qe, nside, cls_len, transf, ninv_p,
                                 chain_descr=chain_descr, bmarg_lmax=BMARG_LCUT, zbounds=zbounds, _bmarg_lib_dir=BMARG_LIBDIR, _bmarg_rescal=tniti_rescal)


#---Add 5 degrees to mask zbounds:
zbounds_len = [np.cos(np.arccos(zbounds[0]) + 5. / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - 5. / 180 * np.pi)]
zbounds_len[0] = max(zbounds_len[0], -1.)
zbounds_len[1] = min(zbounds_len[1],  1.)

#-- Add 5 degress to mask pbounds
pbounds_len = np.array((113.20399439681668, 326.79600560318335)) #hardcoded

from itercurv.remapping import remapping

for simid in simids[mpi.rank::mpi.size]:
    for iti, it in enumerate(iteration):
        if simid>=200:
            sims = sims_sep
        else:
            sims = sims_may

        ivfs_raw_OBD    = filt_cinv.library_cinv_sepTP(libdir_ivfs.replace('ivfs', 'ivfsOBD'), sims, cinv_t, cinv_p_OBD, cls_len)
        lib_dir_iterator = TEMPMFbase_lmax + 'zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % simid
        lib_dir = lib_dir_iterator + '/ffi_p_it{}{}'.format(it, blm_suffix)
        rm = remapping.cached_deflection(lib_dir, nside, 1, facres=-1, zbounds=zbounds_len, pbounds=pbounds_len)
        if blm_suffix == '':
            plm_lensc = Rec.load_plms(lib_dir_iterator, [it])[0]
        else:
            plm_lensc = np.load(lib_dir_iterator+'/ffi_p_it{}{}/'.format(it, blm_suffix)+'plm_mm_%04d_it%d.npy'%(simid, it))
        rm.prepare_filtering(plm_lensc)   
        if it == 0:
            wflm0 = lambda : alm_copy(ivfs_raw_OBD.get_sim_emliklm(simid), lmax=lmax_filt)
            elm = wflm0()
        else:
            elm = Rec.load_elm(lib_dir_iterator, it-1)
        blm = Rec.get_btemplate(lib_dir_iterator, elm, it, pbounds_len, zbounds_len, cache=True, lmax_b=2048, ffi_suffix=blm_suffix)
        np.save(lib_dir+'/blm%s_%04d_it%d.npy'%(blm_suffix, simid, it), blm)
        print('  .. it {}/{}'.format(iti, len(iteration)))
    print('simulation {}/{} done.'.format(simid+1, len(simids)))
    

        
