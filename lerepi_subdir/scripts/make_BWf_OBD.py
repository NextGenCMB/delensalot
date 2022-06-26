import os, sys
import numpy as np
import healpy as hp

import scarf
from plancklens.filt import  filt_util, filt_cinv
from plancklens import qest, qresp, utils
from plancklens.qcinv import cd_solve, opfilt_pp
import itercurv

from itercurv import healpix_hack as hph
from itercurv.filt import opfilt_ee_wl
from itercurv.iterators import cs_iterator
from itercurv.filt import utils_cinv_p

from itercurv.remapping.utils import alm_copy
from cmbs4 import sims_08b

from plancklens.helpers import mpi

fg = '00'
TEMP =  '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/'%fg

TEMP3000  =  TEMP.replace('_lmax4000', '')
TEMP_qlmsddOBD = TEMP.replace('_lmax4000', '')

THIS_CENTRALNLEV_UKAMIN = 0.42# central pol noise level in this pameter file noise sims. The template matrix willbe rescaled

nlev_p = 0.42   #NB: cinv_p gives me this value cinv_p::noiseP_uk_arcmin = 0.429
nlev_t = nlev_p / np.sqrt(2.)
beam = 2.3
lmax_ivf_qe = 3000
lmin_ivf_qe = 10
lmax_qlm = 4096
lmax_transf = 4000 # can be distinct from lmax_filt for iterations
lmax_filt = 4096 # unlensed CMB iteration lmax
nside = 2048
nsims = 200
tol=1e-3

# The gradient spectrum seems to saturate with 1e-3 after roughly this number of iteration
tol_iter = lambda itr : 1e-3 if itr <= 10 else 1e-4
soltn_cond = lambda itr: True

#---- Redfining opfilt_pp shts to include zbounds
zbounds = sims_08b.get_zbounds(np.inf)
sht_threads = 32
opfilt_pp.alm2map_spin = lambda eblm, nside_, spin, lmax, **kwargs: scarf.alm2map_spin(eblm, spin, nside_, lmax, **kwargs, nthreads=sht_threads, zbounds=zbounds)
opfilt_pp.map2alm_spin = lambda *args, **kwargs: scarf.map2alm_spin(*args, **kwargs, nthreads=sht_threads, zbounds=zbounds)

#---Add 5 degrees to mask zbounds:
zbounds_len = [np.cos(np.arccos(zbounds[0]) + 5. / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - 5. / 180 * np.pi)]
zbounds_len[0] = max(zbounds_len[0], -1.)
zbounds_len[1] = min(zbounds_len[1],  1.)
#-- Add 5 degress to mask pbounds
pbounds_len = np.array((113.20399439681668, 326.79600560318335)) #hardcoded


# --- masks: here we test apodized at ratio 10 and weightmap
if not os.path.exists(TEMP):
    os.makedirs(TEMP)
ivmap_path = os.path.join(TEMP, 'ipvmap.fits')
if not os.path.exists(ivmap_path):
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08b/rhits/n2048.fits'))
    pixlev = THIS_CENTRALNLEV_UKAMIN / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    print("Pmap center pixel pol noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmap_path,  1./ pixlev ** 2 * rhits)
ivmat_path = os.path.join(TEMP, 'itvmap.fits')
if not os.path.exists(ivmat_path):
    pixlev= 0.27 * np.sqrt(2) / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08b/rhits/n2048.fits'))
    rhits = np.where(rhits > 0., rhits, 0.)  # *(~np.isnan(rhits))
    print("Pmap center pixel T noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmat_path,  1./ pixlev ** 2 * rhits)

cls_path = os.path.join(os.path.dirname(itercurv.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_grad = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_weights_qe =  utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)

sims_cmbs4May = sims_08b.caterinaILC_May12(fg)
sims_cmbs4Sep = sims_08b.caterinaILC_Sep12(fg)

simmax = 500
simids = np.arange(200,simmax)
if simmax>200:
    sims = sims_cmbs4Sep
else:
    sims = sims_cmbs4May

#------- For sep_TP
ftl_stp_nocut = utils.cli(cls_len['tt'][:lmax_ivf_qe + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe+1]  ** 2)
fel_stp_nocut = utils.cli(cls_len['ee'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe+1]  ** 2)
fbl_stp_nocut = utils.cli(cls_len['bb'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe+1]  ** 2)
filt_t = np.ones(lmax_ivf_qe + 1, dtype=float); filt_t[:lmin_ivf_qe] = 0.
filt_e = np.ones(lmax_ivf_qe + 1, dtype=float); filt_e[:lmin_ivf_qe] = 0.
filt_b = np.ones(lmax_ivf_qe + 1, dtype=float); filt_b[:lmin_ivf_qe] = 0.

ftl_stp = ftl_stp_nocut * filt_t
fel_stp = fel_stp_nocut * filt_e
fbl_stp = fbl_stp_nocut * filt_b


#------- We cache all modes inclusive of the low-ell, and then refilter on the fly with filt_t, filt_e, filt_b
libdir_cinvt = os.path.join(TEMP3000, 'cinvt')
libdir_cinvp = os.path.join(TEMP3000, 'cinvp')
libdir_ivfs = os.path.join(TEMP3000, 'ivfs')

chain_descr = [[0, ["diag_cl"], lmax_ivf_qe, nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
chain_descr_f = lambda cgtol: [[0, ["diag_cl"], lmax_ivf_qe, nside, np.inf, cgtol, cd_solve.tr_cg, cd_solve.cache_mem()]]


ninv_t = [ivmat_path]
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_t,
                     marge_monopole=True, marge_dipole=True, marge_maps=[], chain_descr=chain_descr)

ninv_p = [[ivmap_path]]
cinv_p_OBD_nobmarg = utils_cinv_p.cinv_p(libdir_cinvp.replace('cinvp', 'cinvpOBD'), lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_p,
                                 chain_descr=chain_descr, zbounds=zbounds)

ivfs_raw_OBD    = filt_cinv.library_cinv_sepTP(libdir_ivfs.replace('ivfs', 'ivfsOBD'), sims, cinv_t, cinv_p_OBD_nobmarg, cls_len)
ivfs_OBD   = filt_util.library_ftl(ivfs_raw_OBD, lmax_ivf_qe, filt_t, filt_e, filt_b)


for simidx, simid in enumerate(simids[mpi.rank::mpi.size]):

    lib_dir_iterator = TEMP + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % simid
    BWf_dir = lib_dir_iterator+'/ffi_p_it0/'
    bwflm = ivfs_OBD.get_sim_bmliklm(simid) # b maximum likelihood lm
    np.save(BWf_dir+'bwflm_%04d.npy'%(simid), bwflm)
    
    
    