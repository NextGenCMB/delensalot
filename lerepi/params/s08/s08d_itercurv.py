"""Iterative reconstruction for s08d fg 00 / 07, using Caterina ILC maps

Steps to run this successfully:
    1. calculate tniti
    2. get central noise level
    3. choose filter
    4. check dir structure
    
    NOTE: Issue when first run with mpi:
        ivpmat and resp files file access issue
        mf calculation, barrier at end doesn't seem to work
        
"""

import os, sys
import numpy as np
import healpy as hp
import argparse

from plancklens.helpers import mpi
from plancklens.filt import  filt_util, filt_cinv
from itercurv.filt import utils_cinv_p as iterc_cinv_p
from plancklens import qest, qresp
from plancklens.qcinv import cd_solve, opfilt_pp

import scarf

import lenscarf

from lenscarf import utils
from lenscarf.iterators.statics import rec
from itercurv.filt import opfilt_ee_wl
from lenscarf.opfilt import opfilt_ee_wl


from lenscarf.iterators import cs_iterator as lenscarf_it
from itercurv.iterators import cs_iterator as itercurv_it

from itercurv.remapping.utils import alm_copy

from lerepi.data.dc08 import data_08d as sims_if
from lerepi.survey_config.dc08 import sc_08d as survey_config

isOBD = True
qe_key = 'p_p'
fg = '00'
#TODO remove itercurv

TEMP_OBD =  '/global/cscratch1/sd/sebibel/cmbs4/s08d/cILC_%s_nozboundslen/'%fg
TEMP_noOBD =  '/global/cscratch1/sd/sebibel/cmbs4/s08d/cILC_%s_noOBD/'%fg
if isOBD:
    TEMP = TEMP_OBD
else:
    TEMP = TEMP_noOBD
TEMP_it = TEMP + '/iterator_p_p_%04d_OBD'

BMARG_LIBDIR = survey_config.BMARG_LIBDIR
BMARG_LCUT = survey_config.BMARG_LCUT
beam = survey_config.get_ILC_beam()

nlev_p = survey_config.THIS_CENTRALNLEV_UKAMIN 
nlev_t = nlev_p / np.sqrt(2.)



lmax_ivf_qe = 3000
lmin_ivf_qe = 10
lmax_qlm, mmax_qlm = (4096, 4096)
lmax_transf = 4000 # can be distinct from lmax_filt for iterations
lmax_filt = 4096 # unlensed CMB iteration lmax
nside = 2048
lmax_unl, mmax_unl = (4096, 4096)
nsims = 2

tol = 1e-3
# The gradient spectrum seems to saturate with 1e-3 after roughly this number of iteration
tol_iter = lambda itr : 1e-3 if itr <= 10 else 1e-4
soltn_cond = lambda itr: True

zbounds = survey_config.get_zbounds()
# zbounds = survey_config.get_zbounds()
# zbounds = survey_config.extend_zbounds(zbounds) #---Add 5 degrees to mask zbounds:

pbounds=[0,0]
zbounds_len = [-1,1]
sht_threads = 32
opfilt_pp.alm2map_spin = lambda eblm, nside_, spin, lmax, **kwargs: scarf.alm2map_spin(eblm, spin, nside_, lmax, **kwargs, nthreads=sht_threads, zbounds=zbounds)
opfilt_pp.map2alm_spin = lambda *args, **kwargs: scarf.map2alm_spin(*args, **kwargs, nthreads=sht_threads, zbounds=zbounds)

if not os.path.exists(TEMP):
    os.makedirs(TEMP)
ivmap_path = os.path.join(TEMP, 'ipvmap.fits')
if not os.path.exists(ivmap_path):
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits')) #TODO this should come from survey_conf
    pixlev = nlev_p / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    print("Pmap center pixel pol noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmap_path,  1./ pixlev ** 2 * rhits)  #TODO this should be provided to app level
ivmat_path = os.path.join(TEMP, 'itvmap.fits')
if not os.path.exists(ivmat_path):
    pixlev= 0.27 * np.sqrt(2) / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits')) #TODO this should come from survey_conf
    rhits = np.where(rhits > 0., rhits, 0.)  # *(~np.isnan(rhits))
    print("Pmap center pixel T noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmat_path,  1./ pixlev ** 2 * rhits)  #TODO this should be provided to app level

cls_path = os.path.join(os.path.dirname(lenscarf.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_grad = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_weights_qe =  utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_weights_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)
cls_weights_len['bb'] *= 0.
cls_weights_unl = utils.camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)

transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)
sims = sims_if.ILC_May2022(fg)

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
libdir_cinvt = os.path.join(TEMP, 'cinvt')
libdir_cinvp = os.path.join(TEMP, 'cinvpOBD')
libdir_ivfs = os.path.join(TEMP, 'ivfsOBD')

chain_descr = [[0, ["diag_cl"], lmax_ivf_qe, nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

ninv_t = [ivmat_path]
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[], chain_descr=chain_descr)

ninv_p = [[ivmap_path]]
if isOBD:
    cinv_p = iterc_cinv_p.cinv_p(libdir_cinvp, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_p,
                                 chain_descr=chain_descr, bmarg_lmax=BMARG_LCUT, zbounds=zbounds, _bmarg_lib_dir=BMARG_LIBDIR)
else: 
    cinv_p = iterc_cinv_p.cinv_p(libdir_cinvp, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_p,
                                 chain_descr=chain_descr, zbounds=zbounds)

ivfs_raw = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cls_len)
ivfs = filt_util.library_ftl(ivfs_raw, lmax_ivf_qe, filt_t, filt_e, filt_b)

#-----: This is a filtering instance shuffling simulation indices according to 'ss_dict'.
ddresp = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp_dd_stp'), lmax_ivf_qe, cls_weights_qe, cls_grad,
                               {'tt':ftl_stp.copy(), 'ee':fel_stp.copy(),'bb':fbl_stp.copy()}, lmax_qlm)
qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_ddOBD'), ivfs, ivfs, cls_len['te'], nside, lmax_qlm=lmax_qlm, resplib=ddresp)                        

qlibs = [qlms_dd]
qcls_dd, qcls_ds, qcls_ss = (None, None, None)


def load_resp():
    if not os.path.exists(TEMP + '/resp_len_%s.dat' % qe_key):
        respG_len, respC_len, _, _ = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_weights_len, cls_len,
                                                        {'t': ftl_stp, 'e': fel_stp, 'b': fbl_stp}, lmax_qlm=lmax_qlm)
        np.savetxt(TEMP + '/resp_len_%s.dat' % qe_key, np.array([respG_len, respC_len]).transpose())

    if not os.path.exists(TEMP + '/resp_grad_%s.dat' % qe_key):
        respG_len, respC_len, _, _ = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_weights_len, cls_grad,
                                                        {'t': ftl_stp, 'e': fel_stp, 'b': fbl_stp}, lmax_qlm=lmax_qlm)
        np.savetxt(TEMP + '/resp_grad_%s.dat' % qe_key, np.array([respG_len, respC_len]).transpose())

    if not os.path.exists(TEMP + '/resp_unl_%s.dat' % qe_key):
        respG_unl, respC_unl, _, _ = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_weights_unl, cls_unl,
                                                        {'t': ftl_stp_unl, 'e': fel_stp_unl, 'b': fbl_stp_unl}, lmax_qlm=lmax_qlm)
        np.savetxt(TEMP + '/resp_unl_%s.dat' % qe_key, np.array([respG_unl, respC_unl]).transpose())

    respG_len, respC_len = np.loadtxt(TEMP + '/resp_len_%s.dat' % qe_key).transpose()
    respG_unl, respC_unl = np.loadtxt(TEMP + '/resp_unl_%s.dat' % qe_key).transpose()
    respG_grad, respC_grad = np.loadtxt(TEMP + '/resp_grad_%s.dat' % qe_key).transpose()

    return respG_len, respC_len, respG_unl, respC_unl, respG_grad, respC_grad

ftl_stp_unl = utils.cli(cls_unl['tt'][:lmax_ivf_qe + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
fel_stp_unl = utils.cli(cls_unl['ee'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)
fbl_stp_unl = utils.cli(cls_unl['bb'][:lmax_ivf_qe + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax_ivf_qe + 1] ** 2)

ftl_stp_unl[:lmin_ivf_qe] *= 0.
fel_stp_unl[:lmin_ivf_qe] *= 0.
fbl_stp_unl[:lmin_ivf_qe] *= 0.

respG_len, respC_len, respG_unl, respC_unl, respG_grad, respC_grad = load_resp()

def get_itlib(qe_key, DATIDX):

    assert qe_key == 'p_p'

    TEMP_it = TEMP + '/iterator_p_p_%04d_OBD'%DATIDX

    if not os.path.exists(TEMP + '/mfresp_unl_%s.dat' % qe_key):
        lmax = min(lmax_transf, lmax_filt)
        cls_ivfs = {
            'tt': utils.cli(cls_unl['tt'][:lmax + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax+ 1] ** 2),
            'ee': utils.cli(cls_unl['ee'][:lmax + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2),
            'bb': utils.cli(cls_unl['bb'][:lmax + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2), }

        cls_cmb = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
        for cl in cls_ivfs.values():
            cl[:lmin_ivf_qe] *= 0.
        np.savetxt(TEMP + '/mfresp_unl_%s_cmblmax%s.dat' % (qe_key, len(cls_cmb['tt']-1)),
                   np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(lmax_filt, lmax_transf), lmax_qlm)).transpose())
        for cl in cls_cmb.values():
            cl[lmax_filt + 1:] *= 0.
        np.savetxt(TEMP + '/mfresp_unl_%s.dat' % qe_key,  np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(lmax_filt, lmax_transf), lmax_qlm)).transpose())

    if not os.path.exists(TEMP + '/mfresp_len_%s.dat' % qe_key):
        lmax = min(lmax_transf, lmax_filt)
        cls_ivfs = {
            'tt': utils.cli(cls_len['tt'][:lmax + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2),
            'ee': utils.cli(cls_len['ee'][:lmax + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2),
            'bb': utils.cli(cls_len['bb'][:lmax+ 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2), }
        cls_cmb = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
        for cl in cls_ivfs.values():
            cl[:lmin_ivf_qe] *= 0.
        for cl in cls_cmb.values():
            cl[lmax_filt + 1:] *= 0.
        np.savetxt(TEMP + '/mfresp_len_%s.dat' % qe_key,
                   np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(lmax_filt, lmax_transf), lmax_qlm)).transpose())

    N0_len = utils.cli(respG_len)
    H0_unl = respG_unl
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
    clwf = cpp * utils.cli(cpp + N0_len)
    qnorm = utils.cli(respG_grad)

    #TODO needs reworking for mpi.
    mc_sims_mf = np.arange(nsims)
    mf0 = qlms_dd.get_sim_qlm_mf_mpi('p_p', mc_sims=mc_sims_mf, mpi=mpi)
    if DATIDX in mc_sims_mf:
        mf0 =  (mf0 * len(mc_sims_mf) - qlms_dd.get_sim_qlm('p_p', DATIDX)) / (len(mc_sims_mf) - 1.)
    
    plm0 = hp.almxfl(alm_copy(qlms_dd.get_sim_qlm(qe_key, DATIDX), lmax=lmax_qlm) - mf0, qnorm * clwf)
    dat = sims.get_sim_pmap(DATIDX)

    if qe_key == 'p_p':
        pixn_inv = [hp.read_map(ivmap_path)]
    else:
        assert 0

    def opfilt(libdir, plm, olm=None):
        if isOBD:
            return opfilt_ee_wl.alm_filter_ninv_wl(libdir, pixn_inv, transf, lmax_filt, plm, olm=olm, nside_lens=2048, nbands_lens=1, facres=-1,zbounds=zbounds, zbounds_len=zbounds,  pbounds=pbounds)
        else:
            return opfilt_ee_wl.alm_filter_ninv_wl(libdir, pixn_inv, transf, lmax_filt, plm, bmarg_lmax=BMARG_LCUT, _bmarg_lib_dir=BMARG_LIBDIR, olm=olm, nside_lens=2048, nbands_lens=1, facres=-1,zbounds=zbounds, zbounds_len=zbounds_len, pbounds=pbounds)

    chain_descr = [[0, ["diag_cl"], lmax_filt, nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

    cls_filt = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    #--- trying to kill L = 1, 2 and 3 with prior
    cpp[:4] *= 1e-5

    def bp(x, xa, a, xb, b, scale=50):
        "Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale"
        x0 = (xa + xb) * 0.5
        r = lambda x: np.arctan(np.sign(b - a) * (x - x0) / scale) + np.sign(b - a) * np.pi * 0.5
        return a + r(x) * (b - a) / r(xb)

    def step_length(iter, norm_incr):
        return bp(np.arange(4097), 400, 0.5, 1500, 0.1, scale=50)

    wflm0 = lambda : alm_copy(ivfs_raw.get_sim_emliklm(DATIDX), lmax=lmax_filt)
    
    # TODO switch to lenscarf iterator
    itlib_itercurv =  itercurv_it.iterator_cstmf(TEMP_it ,{'p_p': 'QU', 'p': 'TQU', 'ptt': 'T'}[qe_key],
                        dat, plm0, mf0, H0_unl, cpp, cls_filt, lmax_filt, wflm0=wflm0, chain_descr=chain_descr,  ninv_filt=opfilt)
    itlib_itercurv.newton_step_length = step_length

    # mf_resp = qresp.get_mf_resp(qe_key, cls_unl, {'ee': fel_stp_unl, 'bb': fbl_stp_unl}, lmax_ivf_qe, lmax_qlm)[0]
    # R_unl = qresp.get_response(qe_key, lmax_ivf_qe, 'p', cls_unl, cls_unl,  {'e': fel_stp_unl, 'b': fbl_stp_unl, 't':ftl_stp_unl}, lmax_qlm=lmax_qlm)[0]
    # itlib_lenscarf = lenscarf_it.iterator_pertmf(TEMP_it, 'p', (lmax_qlm, lmax_qlm), dat,
    #         plm0, mf_resp, R_unl, cpp, cls_unl, opfilt, opfilt.ffi.geom, chain_descr, step_length,
    #         mf0=mf0, wflm0=lambda : alm_copy(ivfs.get_sim_emliklm(DATIDX), None, lmax_filt, lmax_filt))

    return itlib_itercurv#, itlib_lenscarf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-btempl', dest='btempl', action='store_true', help='build B-templ for last iter > 0')

    args = parser.parse_args()

    mpi.barrier = lambda : 1 # redefining the barrier

    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        TEMP_it_loc = TEMP_it%idx
        if rec.maxiterdone(TEMP_it_loc) < args.itmax:
            jobs.append(idx)    

    for idx in jobs[mpi.rank::mpi.size]:
        TEMP_it_loc = TEMP_it%idx
        if args.itmax >= 0 and rec.maxiterdone(TEMP_it_loc) < args.itmax:
            itlib = get_itlib(qe_key, idx)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                chain_descr = [[0, ["diag_cl"], lmax_filt, nside, np.inf, tol_iter(i), cd_solve.tr_cg, cd_solve.cache_mem()]]
                itlib.chain_descr  = chain_descr
                itlib.soltn_cond = soltn_cond(i)

                print("doing iter " + str(i))
                itlib.iterate(i, 'p')
                
    # Produces B-template for last iteration
    if args.btempl:
        from itercurv.iterators.statics import rec as Rec
        elm = Rec.load_elm(TEMP_it_loc, args.itmax - 1)
        Rec.get_btemplate(TEMP_it_loc, elm, args.itmax, pbounds_len, zbounds_len,  cache=True, lmax_b=2048)
        itlib = get_itlib(qe_key, 0)
        blm = itlib.get_template_blm(args.itmax, args.itmax, pbounds_len=pbounds, lmaxb=2048, lmin_plm=1)