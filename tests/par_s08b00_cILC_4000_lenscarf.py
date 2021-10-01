"""Iterative reconstruction for s06b fg 00, using Caterina ILC maps



"""
import os, sys
import numpy as np
import healpy as hp
import plancklens
from lenscarf.iterators import cs_iterator as scarf_iterator, steps
from lenscarf.utils_hp import  alm_copy
from lenscarf.opfilt import bmodes_ninv as bni
from plancklens import  qresp, utils
from plancklens.qcinv import cd_solve
from lenscarf.opfilt import opfilt_ee_wl as opfilt_ee_wl_scarf
from cmbs4.plotrec_utils import get_bb_amplitude
import scarf
from lenscarf import remapping, utils_scarf
from cmbs4 import sims_08b


#-------- reference reconstruction parfiles that will give me the mean-field
#         (and previous recs to compare with in this case)
from cmbs4.params.s08b_00 import par_s08b00_cILC_4000 as ref_parfile
assert hasattr(ref_parfile, 'qlms_dd')      # for mean-field
assert hasattr(ref_parfile, 'ivfs_raw_OBD') # for first iteration starting point E-mode
assert hasattr(ref_parfile, 'sims') # for data map
TEMP =  ref_parfile.TEMP + '/lenscarf_recs'
if not os.path.exists(TEMP):
    os.makedirs(TEMP)

#------ low-ell B-modes marginalization parameters:
BMARG_LIBDIR  = '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/'
BMARG_LCUT=200
BMARG_CENTRALNLEV_UKAMIN = 0.350500 # central pol noise level in map used to build the (TniT) inverse matrix
THIS_CENTRALNLEV_UKAMIN = 0.42# central pol noise level in this pameter file noise sims. The template matrix willbe rescaled
tniti_rescal = (THIS_CENTRALNLEV_UKAMIN / BMARG_CENTRALNLEV_UKAMIN) ** 2
#------

#----- MAP rec. parameters
nlev_p = 0.42   #NB: cinv_p gives me this value cinv_p::noiseP_uk_arcmin = 0.429
nlev_t = nlev_p / np.sqrt(2.)
beam = 2.3
lmax_ivf_qe = ref_parfile.lmax_ivf_qe
lmin_ivf_qe = ref_parfile.lmin_ivf_qe
lmax_qlm = 4096
lmax_transf = 4000 # can be distinct from lmax_filt for iterations
lmax_filt = 4096 # unlensed CMB iteration lmax
nside = 2048
nsims = 200  # number of sims for the mean-field
tol=1e-3

# --- cg iterations parameters
tol_iter = lambda itr : 1e-3 if itr <= 10 else 1e-4
soltn_cond = lambda itr: True

# --- Here we extract zbounds to speed up the spherical transforms
zbounds = sims_08b.get_zbounds(np.inf) # sharp zbounds of inverse noise variance maps
# --- We also build zbounds outside which the lensing is not performed at all, assuming everything is zero
#     we do this by padding the zbounds by 5 degrees
#     (could try to remove this, just to check)
zbounds_len = [np.cos(np.arccos(zbounds[0]) + 5. / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - 5. / 180 * np.pi)]
zbounds_len[0] = max(zbounds_len[0], -1.)
zbounds_len[1] = min(zbounds_len[1],  1.)
# -- We also build here longitude bounds, outside of which the lensing is not performed at all
# -- (Maybe this actually slows things down, not sure, can try to set pb_ctr, pb_extent to (np.pi, 2* np.pi)
pbounds_len = np.array((113.20399439681668, 326.79600560318335)) # This was built also using a 5 degrees buffer
pb_ctr = np.mean([-(360. - pbounds_len[1]), pbounds_len[0]]) # centre of patch
pb_extent = pbounds_len[0] + (360. - pbounds_len[1])   # extent of patch
pb_ctr, pb_extent = (pb_ctr / 180 * np.pi, pb_extent / 180 * np.pi)
# -- scarf geometries
tr = int(os.environ.get('OMP_NUM_THREADS', 8)) # threads
ninv_job = utils_scarf.scarfjob()  # input data geometry etc
ninv_job.set_healpix_geometry(nside, zbounds=zbounds)  # input maps are given to us on a healpix geom zero outside of zbounds
ninvgeom = ninv_job.geom

# -- now we just get the pixels defining the slice to take into the input maps to get the non-zero part
hp_geom = scarf.healpix_geometry(2048, 1)
hp_start = hp_geom.ofs[np.where(hp_geom.theta == np.min(ninvgeom.theta))[0]][0]
hp_end = hp_start + utils_scarf.Geom.npix(ninvgeom).astype(hp_start.dtype)  # Somehow otherwise makes a float out of int64 and uint64 ???
# --- scarf geometry of the lensing jobs at each iteration
lenjob = utils_scarf.scarfjob()
mmax_filt = None# we could reduce that since we are not too far from the pole. From command line args
mmax_qlm = lmax_qlm
#NB: the lensing jobs geom are specified in the command line arguments


def bp(x, xa, a, xb, b, scale=50): # helper function to build step-length
    "Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale"
    x0 = (xa + xb) * 0.5
    r = lambda x: np.arctan(np.sign(b - a) * (x - x0) / scale) + np.sign(b - a) * np.pi * 0.5
    return a + r(x) * (b - a) / r(xb)


def step_length(iter, norm_incr):
    return bp(np.arange(4097), 400, 0.5, 1500, 0.1, scale=50)


def get_itlib(qe_key, DATIDX,  vscarf='p', mmax_is_lmax=True):
    #assert vscarf in [False, '', 'd', 'k', 'p'], vscarf
    lib_dir = TEMP
    lib_dir_iterator = lib_dir + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20'%DATIDX
    if vscarf not in [False, '']:
        lib_dir_iterator += vscarf
    assert qe_key == 'p_p'
    cls_path = os.path.join(os.path.dirname(plancklens.__file__), 'data', 'cls')
    cls_weights_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)
    cls_weights_unl = utils.camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)
    cls_unl = utils.camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))
    cls_filt = utils.camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))
    cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_grad = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
    cls_weights_len['bb'] *= 0.

    transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)
    ftl_stp = utils.cli(cls_len['tt'][:lmax_ivf_qe + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    fel_stp = utils.cli(cls_len['ee'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    fbl_stp = utils.cli(cls_len['bb'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    ftl_stp_unl = utils.cli(cls_unl['tt'][:lmax_ivf_qe + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    fel_stp_unl = utils.cli(cls_unl['ee'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    fbl_stp_unl = utils.cli(cls_unl['bb'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
    ftl_stp[:lmin_ivf_qe] *= 0.
    fel_stp[:lmin_ivf_qe] *= 0.
    fbl_stp[:lmin_ivf_qe] *= 0.
    ftl_stp_unl[:lmin_ivf_qe] *= 0.
    fel_stp_unl[:lmin_ivf_qe] *= 0.
    fbl_stp_unl[:lmin_ivf_qe] *= 0.

    respG_unl, _, _, _ = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_weights_unl, cls_unl,
                                             {'t': ftl_stp_unl, 'e': fel_stp_unl, 'b': fbl_stp_unl}, lmax_qlm=lmax_qlm)
    respG_grad, _, _, _ = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_weights_len, cls_grad,
                                              {'t': ftl_stp, 'e': fel_stp, 'b': fbl_stp}, lmax_qlm=lmax_qlm)


    N0_len = utils.cli(respG_grad)
    H0_unl = respG_unl
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
    clwf = cpp * utils.cli(cpp + N0_len)
    qnorm = utils.cli(respG_grad)
    mc_sims_mf = np.arange(nsims)
    mf0 = ref_parfile.qlms_dd.get_sim_qlm_mf('p_p', mc_sims=mc_sims_mf)
    if DATIDX in mc_sims_mf:
        mf0 =  (mf0 * len(mc_sims_mf) - ref_parfile.qlms_dd.get_sim_qlm('p_p', DATIDX)) / (len(mc_sims_mf) - 1.)
    plm0 = hp.almxfl(utils.alm_copy(ref_parfile.qlms_dd.get_sim_qlm(qe_key, DATIDX), lmax=lmax_qlm) - mf0, qnorm * clwf)

    chain_descr = [[0, ["diag_cl"], lmax_filt, nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

    #--- killing L = 1, 2 and 3 with prior
    cpp[:4] *= 1e-5


    wflm0 = lambda : alm_copy(ref_parfile.ivfs_raw_OBD.get_sim_emliklm(DATIDX), None, lmax_filt, mmax_filt)

    if 'h' not in vscarf:
        lenjob.set_thingauss_geometry(max(lmax_filt, lmax_transf), 2, zbounds=zbounds_len)
    else:
        lenjob.set_healpix_geometry(2048, zbounds=zbounds_len)
    if 'f' in vscarf:
        k_geom = scarf.healpix_geometry(2048, 1)
    else:
        k_geom = lenjob.geom
    if not mmax_is_lmax:
        from lenscarf import utils_sht
        tht_mmax = np.min(np.abs(lenjob.geom.theta - np.pi * 0.5)) + np.pi * 0.5
        mmax_unl = int(
            np.ceil(max(utils_sht.st2mmax(2, tht_mmax, lmax_filt), utils_sht.st2mmax(-2, tht_mmax, lmax_filt))))
        mmax_filt = min(lmax_filt, mmax_unl)
    else:
        mmax_filt = lmax_filt
    print("lmax filt, mmax filt %s %s"%(lmax_filt, mmax_filt))
    ninv_sc = [hp.read_map(ref_parfile.ivmap_path)[hp_start:hp_end]]
    if 'r' in vscarf:
        if vscarf[0] == 'k':
            h2k = np.ones(lmax_qlm + 1, dtype=float)
        elif vscarf[0] == 'p':
            h2k = np.arange(lmax_qlm + 1) * np.arange(1, lmax_qlm + 2) * 0.5
        elif vscarf[0] == 'd':
            h2k = np.sqrt(np.arange(lmax_qlm + 1) * np.arange(1, lmax_qlm + 2))
        else:
            assert 0
        stepper = steps.hmapprescal(lmax_qlm, mmax_qlm, h2k, ninv_sc[0], (0.1, 0.5), ninvgeom, tr)
    else:
        stepper = steps.harmonicbump(lmax_qlm, mmax_qlm)

    dat = ref_parfile.sims.get_sim_pmap(DATIDX)
    dat = np.array([da[hp_start:hp_end] for da in dat])
    assert dat[0].size == utils_scarf.Geom.npix(ninvgeom), (dat[0].size,utils_scarf.Geom.npix(ninvgeom) )

    tpl = bni.template_dense(BMARG_LCUT, ninvgeom, tr, _lib_dir=BMARG_LIBDIR, rescal=tniti_rescal)
    pbd_geom = utils_scarf.pbdGeometry(lenjob.geom, utils_scarf.pbounds(pb_ctr, pb_extent))
    ffi = remapping.deflection(pbd_geom, 1.7, np.zeros_like(plm0), mmax_qlm, tr, tr)

    filtr = opfilt_ee_wl_scarf.alm_filter_ninv_wl(ninvgeom, ninv_sc, ffi, transf, (lmax_filt, mmax_filt), (lmax_transf, lmax_transf), tr, tpl)
    if '0' in vscarf:
        mf0 *= 0.
    itlib = scarf_iterator.iterator_cstmf(lib_dir_iterator, vscarf[0], (lmax_qlm, mmax_qlm), dat,
                                        plm0, mf0, H0_unl, cpp, cls_filt, filtr, k_geom, chain_descr, stepper, wflm0=wflm0)
    itlib.newton_step_length = step_length
    return itlib


def build_Bampl(this_itlib, this_itr, datidx, cache_b=False):
    from lenscarf import cachers
    from plancklens.sims import planck2018_sims
    e_fname = 'wflm_%s_it%s' % ('p', this_itr - 1)
    assert this_itlib.wf_cacher.is_cached(e_fname)
    # loading deflection field at the wanted iter:
    dlm = this_itlib.get_hlm(this_itr, 'p')
    this_itlib.hlm2dlm(dlm, True)
    ffi = this_itlib.filter.ffi.change_dlm([dlm, None], this_itlib.mmax_qlm, cachers.cacher_mem())
    this_itlib.filter.set_ffi(ffi)

    # loading e-mode map:
    elm = this_itlib.wf_cacher.load('wflm_%s_it%s' % ('p', this_itr - 1))
    lmax_b, mmax_b = (2048, 2048)
    b_fname =this_itlib.lib_dir + '/blm_%04d_%s_lmax%s.fits' % (this_itr, utils.clhash(elm.real), lmax_b)
    _, blm = this_itlib.filter.ffi.lensgclm(np.array([elm, elm * 0]), this_itlib.mmax_filt, 2, lmax_b, mmax_b, False)
    if cache_b:
        hp.write_alm(b_fname, blm)
        print('Cached ', b_fname)
    cls_path = os.path.join(os.path.dirname(plancklens.__file__), 'data', 'cls')
    cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    blm_in = alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(datidx), None, lmax_b, mmax_b)
    print("BB ampl itr " + str(this_itr))
    Abb = get_bb_amplitude(sims_08b.get_nlev_mask(2.), cls_len, blm, blm_in)
    f = open(itlib.lib_dir + "/BBampl.txt", "a")
    f.write("%4s %.5f" % (this_itr, Abb))
    f.close()
    return Abb


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-imin', dest='imin', type=int, default=0, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=0, help='maximal sim index')
    parser.add_argument('-btempl', dest='btempl', action='store_true', help='build B-templ for last iter > 0')
    parser.add_argument('-scarf', dest='scarf', type=str, default='p', help='further iterator options')
    parser.add_argument('-mmax', dest='mmax',  action='store_true', help='reduces mmax to some value')
    parser.add_argument('-BB', dest='BB',  action='store_true', help='calc BB ampls at each iter')

    #vscarf: 'p' 'k' 'd' for bfgs variable
    # add a 'f' to use full sky in once-per iteration kappa thingy
    # add a 'r' for real space attenuation of the step instead of harmonic space
    # add a '0' for no mf

    args = parser.parse_args()
    from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier
    from itercurv.iterators.statics import rec as Rec


    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = TEMP + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % idx + args.scarf
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append( (idx, Rec.maxiterdone(lib_dir_iterator)) )
            print(lib_dir_iterator)

    for idx, itdone in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = TEMP + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % idx + args.scarf
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib('p_p', idx,  vscarf=args.scarf, mmax_is_lmax=not args.mmax)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                chain_descr = [[0, ["diag_cl"], lmax_filt, nside, np.inf, tol_iter(i), cd_solve.tr_cg, cd_solve.cache_mem()]]
                itlib.chain_descr  = chain_descr
                itlib.soltn_cond = soltn_cond

                print("doing iter " + str(i))
                itlib.iterate(i, 'p')
                if args.BB and i > 0 and i > itdone:
                    print(build_Bampl(itlib, i, idx, cache_b= args.btempl * (i == args.imax)))

