import numpy as np
import os
from time import time
import scarf
from lenscarf import remapping
from lenscarf import utils_config, utils_hp, utils, utils_sht
from lenscarf import cachers
from lenscarf.opfilt import opfilt_ee_wl
from lenscarf.utils_scarf import Geom, scarfjob
from plancklens.utils import camb_clfile


lmax_dlm = 4096
lmax_unl = 4000
lmax_len = 4000
sht_threads = 8
fftw_threads = 8
nside = 2048
fwhm = 2.3
targetres_amin=2.
mmax_is_lmax = False

#ninvjob, pbds, zbounds_len, zbounds_ninv  = utils_config.cmbs4_08b_healpix()#_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen
ninvjob, pbds, zbounds_len, zbounds_ninv  = utils_config.cmbs4_08b_healpix_onp()#_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen
#ninvjob, pbds, zbounds_len, zbounds_ninv = utils_config.cmbs4_08b_healpix_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen

ninvgeom = ninvjob.geom
lenjob = scarfjob()

IPVMAP = '/global/cscratch1/sd/jcarron/cmbs4/temp/s08b/cILC2021_00/ipvmap.fits'
if not os.path.exists(IPVMAP):
    IPVMAP = '/Users/jcarron/OneDrive - unige.ch/cmbs4/inputs/ipvmap.fits'

if not mmax_is_lmax:
    tht_mmax = np.min(np.abs(ninvgeom.theta - np.pi * 0.5) ) + np.pi * 0.5
    mmax_unl = int(np.ceil(max(utils_sht.st2mmax(2, tht_mmax, lmax_unl), utils_sht.st2mmax(-2, tht_mmax, lmax_unl))))
    mmax_len = int(np.ceil(max(utils_sht.st2mmax(2, tht_mmax, lmax_len), utils_sht.st2mmax(-2, tht_mmax, lmax_len))))
    mmax_dlm = int(np.ceil(max(utils_sht.st2mmax(2, tht_mmax, lmax_dlm), utils_sht.st2mmax(-2, tht_mmax, lmax_dlm))))
    mmax_unl = min(lmax_unl, mmax_unl)
    mmax_len = min(lmax_len, mmax_len)
    mmax_dlm = min(lmax_dlm, mmax_dlm)
else:
    mmax_unl, mmax_len, mmax_dlm = (lmax_unl, lmax_len, lmax_dlm)

print("Setting mmax unl, len, dlm to %s %s %s"%(mmax_unl, mmax_len, mmax_dlm))

# build slice for zbounded hp:
hp_geom = scarf.healpix_geometry(2048, 1)
hp_start = hp_geom.ofs[np.where(hp_geom.theta == np.min(ninvgeom.theta))[0]][0]
hp_end = hp_start + Geom.npix(ninvgeom).astype(hp_start.dtype) # Somehow otherwise makes a float out of int64 and uint64 ???

# deflection instance:
cldd = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax_dlm + 1]
cldd *= np.sqrt(np.arange(lmax_dlm + 1) *  np.arange(1, lmax_dlm + 2))
#dlm = hp.synalm(cldd, lmax=lmax_dlm, mmax=mmax_dlm) # get segfault with nontrivial mmax and new=True ?!
dlm = utils_hp.synalm(cldd, lmax_dlm, mmax_dlm)

#cacher = cachers.cacher_npy('/Users/jcarron/OneDrive - unige.ch/lenscarf/temp/test_opfilt')
cacher = cachers.cacher_mem()


# ninv filter:
transf = utils_hp.gauss_beam(fwhm, lmax_len)
n_inv = [np.nan_to_num(utils.read_map(IPVMAP)[hp_start:hp_end])]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test of lenscarf-based opfilt')
    parser.add_argument('-lensgeom', dest='lensgeom', type=str, default='healpix', help='minimal sim index')
    args = parser.parse_args()

    if args.lensgeom == 'healpix':
        lenjob.set_healpix_geometry(nside, zbounds=zbounds_len)
    elif args.lensgeom == 'thingauss':
        lenjob.set_thingauss_geometry(max(lmax_unl, lmax_len), 2, zbounds=zbounds_len)
    else:
        assert 0, args.lensgeom + ' not implemented'
    d = remapping.deflection(lenjob.geom, targetres_amin, pbds, dlm, sht_threads, fftw_threads, mmax=mmax_dlm, cacher=cacher)


    t0 = time()
    d._init_d1()
    print('init d1: %.2fs' % (time() - t0))

    t0 = time()
    d._bwd_angles()
    print('inverse deflection: %.2fs' % (time() - t0))
    opfilt = opfilt_ee_wl.alm_filter_ninv_wl(ninvgeom, n_inv, d, transf, (lmax_unl, mmax_unl), (lmax_len, mmax_len),
                                              sht_threads, verbose=True)
    elm = np.zeros(utils_hp.Alm.getsize(lmax_unl, mmax_unl), dtype=complex)
    d.tim.reset()
    opfilt.apply_alm(elm)
    print("2nd version without plans etc")
    d.tim.reset()
    opfilt.apply_alm(elm)

"""comp to NERSC  THREADS 16
                       scarf ecp job setup:  [00h:00m:00s:000ms] 
                             scarf alm2map:  [00h:00m:01s:986ms] 
                             fftw planning:  [00h:00m:00s:003ms] 
                                  fftw fwd:  [00h:00m:00s:421ms] 
                 bicubic prefilt, fftw bwd:  [00h:00m:00s:605ms] 
                             Interpolation:  [00h:00m:01s:551ms] 
                     Polarization rotation:  [00h:00m:00s:334ms] 
 gclm2lensedmap spin 2 lmax glm 4096 lmax dlm 4096 Total :  [00h:00m:05s:334ms] 
 [00:00:05] gclm2lensedmap spin 2 lmax glm 4000 lmax dlm 4096 > 00%
(2048, 4096) [00:03:25] (7, 0.00187220)
 [00:00:05] gclm2lensedmap spin 2 lmax glm 4096 lmax dlm 4096 > 00%
                       scarf ecp job setup:  [00h:00m:00s:000ms] 
                             scarf alm2map:  [00h:00m:01s:884ms] 
                             fftw planning:  [00h:00m:00s:003ms] 
                                  fftw fwd:  [00h:00m:00s:420ms] 
                 bicubic prefilt, fftw bwd:  [00h:00m:00s:600ms] 
                             Interpolation:  [00h:00m:01s:551ms] 
                     Polarization rotation:  [00h:00m:00s:334ms] 
 gclm2lensedmap spin 2 lmax glm 4000 lmax dlm 4096 Total :  [00h:00m:05s:224ms] 
 """
