import numpy as np
import os
from time import time
import scarf
from delensalot import remapping
from delensalot import utils_config, utils, utils_sht, utils_scarf
from delensalot.core import cachers
from delensalot.utility import utils_hp
from delensalot.core.opfilt import opfilt_ee_wl, opfilt_pp
from delensalot.core.helper.utils_scarf import Geom, scarfjob, pbdGeometry, pbounds
from plancklens.utils import camb_clfile
from delensalot import qest_wl
# FIXME: qest_wl part of opfilt
# FIXME: make opfilt classes instead of files?
# FIXME: fix multigrid always assuming mmax=lmax

lmax_dlm = 4096
lmax_unl = 4000
lmax_len = 4000

nside = 2048
fwhm = 2.3
targetres_amin=1.7
mmax_is_lmax = False # False allows for mmax cuts
w_lensing = True
full_sky = False
config = 'cmbs4_08b_healpix_onp' #cmbs4_08b_healpix, cmbs4_08b_healpix_onp, cmbs4_08b_healpix_oneq, full_sky_healpix

if not full_sky:
    #ninvjob, pbds, zbounds_len, zbounds_ninv  = getattr(utils_config, config)() #_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen
    #ninvjob, pbds, zbounds_len, zbounds_ninv  = utils_config.cmbs4_08b_healpix_onp()#_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen
    #ninvjob, pbds, zbounds_len, zbounds_ninv = utils_config.cmbs4_08b_healpix_oneq() #FIXME: can have distinct zbounds for ninv and zboundslen
    IPVMAP = '/global/cscratch1/sd/jcarron/cmbs4/temp/s08b/cILC2021_00/ipvmap.fits'
    if not os.path.exists(IPVMAP):
        IPVMAP = '/Users/jcarron/OneDrive - unige.ch/cmbs4/inputs/ipvmap.fits'
    if config in ['cmbs4_08b_healpix_onp']:
        IPVMAP = '/Users/jcarron/OneDrive - unige.ch/cmbs4/inputs/ipvmap_onpole.fits'
        ninvjob, pbds, zbounds_len, zbounds_ninv = utils_config.cmbs4_08b_healpix_onp()
else:
    ninvjob, pbds, zbounds_len, zbounds_ninv = utils_config.full_sky_healpix()
    import healpy as hp
    nlevp = 1.
    area = hp.nside2pixarea(nside, degrees=False)
    IPVMAP = np.ones(12 * nside ** 2) / ((nlevp / 180 / 60 * np.pi) ** 2 / area)

ninvgeom = ninvjob.geom
lenjob = scarfjob()


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

transf = utils_hp.gauss_beam(fwhm / 180 / 60 * np.pi, lmax_len)
n_inv = [np.nan_to_num(utils.read_map(IPVMAP)[hp_start:hp_end])]

# deflection instance:
cldd = w_lensing * camb_clfile('../delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax_dlm + 1]
cldd *= np.sqrt(np.arange(lmax_dlm + 1) *  np.arange(1, lmax_dlm + 2))
#dlm = hp.synalm(cldd, lmax=lmax_dlm, mmax=mmax_dlm) # get segfault with nontrivial mmax and new=True ?!
dlm = utils_hp.synalm(cldd, lmax_dlm, mmax_dlm)

#cacher = cachers.cacher_npy('/Users/jcarron/OneDrive - unige.ch/delensalot/temp/test_opfilt')
cacher = cachers.cacher_mem()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test of delensalot-based opfilt')
    parser.add_argument('-lensgeom', dest='lensgeom', type=str, default='healpix', help='lens geometry choice')
    parser.add_argument('-run', dest='run', action='store_true', help='build and filter one sim')
    parser.add_argument('-sht', dest='sht', type=int, default=8, help='threads number for SHTs')
    parser.add_argument('-fftw', dest='fftw',type=int, default=8, help='threads number for FFTs')

    args = parser.parse_args()
    sht_threads = args.sht
    fftw_threads = args.fftw
    if args.lensgeom == 'healpix':
        lenjob.set_healpix_geometry(nside, zbounds=zbounds_len)
    elif args.lensgeom == 'thingauss':
        lenjob.set_thingauss_geometry(max(lmax_unl, lmax_len), 2, zbounds=zbounds_len)
    else:
        assert 0, args.lensgeom + ' not implemented'
    d_geom = pbdGeometry(lenjob.geom, pbounds(pbds[0], pbds[1]))
    d = remapping.deflection(d_geom, targetres_amin, dlm, mmax_dlm, sht_threads, fftw_threads, cacher=cacher)


    t0 = time()
    d._init_d1()
    print('init d1: %.2fs' % (time() - t0))

    t0 = time()
    d._bwd_angles()
    print('inverse deflection: %.2fs' % (time() - t0))
    opfilt = opfilt_ee_wl.alm_filter_ninv_wl(ninvgeom, n_inv, d, transf, (lmax_unl, mmax_unl), (lmax_len, mmax_len),
                                              sht_threads, None, verbose=True)
    elm = np.zeros(utils_hp.Alm.getsize(lmax_unl, mmax_unl), dtype=complex)
    d.tim.reset()
    opfilt.apply_alm(elm)
    print("2nd version without plans etc")
    d.tim.reset()
    opfilt.apply_alm(elm)
    if args.run: # build and filter one Gaussian sim
        print("Building G. sims")

        cl_len = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lensedCls.dat')
        elm = utils_hp.synalm(cl_len['ee'], lmax_len, mmax_len)
        blm = utils_hp.synalm(cl_len['bb'], lmax_len, mmax_len) * w_lensing
        utils_hp.almxfl(elm, transf, mmax_len, True)
        utils_hp.almxfl(blm, transf, mmax_len, True)
        ninvjob.set_triangular_alm_info(lmax_len, mmax_len)
        ninvjob.set_nthreads(sht_threads)
        qu_dat = ninvjob.alm2map_spin([elm, blm], 2)
        qu_dat += np.random.standard_normal(qu_dat.shape) * utils.cli(np.sqrt(utils.read_map(n_inv[0])))

        from plancklens.qcinv import cd_solve, multigrid
        itermax = 100
        opfilt.verbose = False
        d.verbose = False
        d.tim.reset()
        opfilt.tim.reset()
        chain_descr = [[0, ["diag_cl"], lmax_unl, nside, itermax, 1e-3, cd_solve.tr_cg, cd_solve.cache_mem()]]
        if w_lensing:
            soltn = np.zeros(utils_hp.Alm.getsize(lmax_unl, mmax_unl), dtype=complex)
            opfilt_file = opfilt_ee_wl
        else:
            soltn = np.zeros((2, utils_hp.Alm.getsize(lmax_unl, mmax_unl)), dtype=complex)
            opfilt_file = opfilt_pp
            opfilt = opfilt_pp.alm_filter_ninv(ninvgeom, n_inv, transf, (lmax_unl, mmax_unl),
                                                     (lmax_len, mmax_len), sht_threads, verbose=True)
        chain = multigrid.multigrid_chain(opfilt_file, chain_descr, cl_len, opfilt)
        chain.solve(soltn, qu_dat, dot_op=opfilt_file.dot_op(lmax_unl, mmax_unl))

        QE_geom = utils_scarf.pbdGeometry(lenjob.geom, utils_scarf.pbounds(np.pi, 2 * np.pi))
        qlms_g, qlms_c = qest_wl.get_qlms_wl(qu_dat, soltn, opfilt, QE_geom)
        qlms_g_EB, qlms_c_EB = qest_wl.get_qlms_wl(qu_dat, soltn, opfilt, QE_geom, wEE=False)

        opfilt.tim.add('qest_wl')

        import pylab as pl
        from plancklens import n0s
        #FIXME:
        N0 = n0s.get_N0(beam_fwhm=fwhm, nlev_t=1./ np.sqrt(2.), nlev_p=1.,
                        lmax_CMB=lmax_len,  lmin_CMB=10, lmax_out=lmax_dlm)[0]['p_p']

        ls = np.arange(2, lmax_dlm + 1)
        cl_unl = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat')
        w = ls ** 2 * (ls + 1) ** 2 * 1e7 /2./np.pi
        pl.loglog(ls, w *  utils_hp.alm2cl(qlms_g, qlms_g, lmax_dlm, mmax_dlm, lmax_dlm)[ls] * N0[ls] ** 2)
        pl.loglog(ls, w *  utils_hp.alm2cl(qlms_g_EB, qlms_g_EB, lmax_dlm, mmax_dlm, lmax_dlm)[ls] * N0[ls] ** 2)

        pl.show()
        print(opfilt._nlevp)
        print(d.tim)
        print(opfilt.tim)
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
