"""Anything related to quadratic estimators should be made accessible via this module

Returns:
    _type_: _description_
"""
import numpy as np
import os, sys

from plancklens.filt import  filt_util, filt_cinv
from plancklens import qest, qresp
from itercurv.remapping.utils import alm_copy

from itercurv.filt import utils_cinv_p as iterc_cinv_p

import lenscarf
from lenscarf import utils

cls_path = os.path.join(os.path.dirname(lenscarf.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_grad = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_weights_qe =  utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_weights_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)
cls_weights_len['bb'] *= 0.
cls_weights_unl = utils.camb_clfile(os.path.join(cls_path,'FFP10_wdipole_lenspotentialCls.dat'))  # QE fiducial weights (here identical to lensed CMB spectra)


def get_ivma_paths(TEMP):
    ivmap_path = os.path.join(TEMP, 'ipvmap.fits')
    if not os.path.exists(ivmap_path):
        print('Run scripts/buildivmapt.py first')
        sys.exit()

    ivmat_path = os.path.join(TEMP, 'itvmap.fits')
    if not os.path.exists(ivmat_path):
        print('Run scripts/buildivmapt.py first')
        sys.exit()

def set_all_responses(TEMP, qe_key, sc):
    ftl_stp_unl = utils.cli(cls_unl['tt'][:sc.lmax_ivf_qe + 1] + (sc.nlev_t / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe + 1] ** 2)
    fel_stp_unl = utils.cli(cls_unl['ee'][:sc.lmax_ivf_qe + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe + 1] ** 2)
    fbl_stp_unl = utils.cli(cls_unl['bb'][:sc.lmax_ivf_qe + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe + 1] ** 2)

    ftl_stp_unl[:sc.lmin_ivf_qe] *= 0.
    fel_stp_unl[:sc.lmin_ivf_qe] *= 0.
    fbl_stp_unl[:sc.lmin_ivf_qe] *= 0.
    if not os.path.exists(TEMP + '/resp_len_%s.dat' % qe_key):
        respG_len, respC_len, _, _ = qresp.get_response(qe_key, sc.lmax_ivf_qe, 'p', cls_weights_len, cls_len,
                                                        {'t': ftl_stp, 'e': fel_stp, 'b': fbl_stp}, lmax_qlm=sc.lmax_qlm)
        np.savetxt(TEMP + '/resp_len_%s.dat' % qe_key, np.array([respG_len, respC_len]).transpose())

    if not os.path.exists(TEMP + '/resp_grad_%s.dat' % qe_key):
        respG_len, respC_len, _, _ = qresp.get_response(qe_key, sc.lmax_ivf_qe, 'p', cls_weights_len, cls_grad,
                                                        {'t': ftl_stp, 'e': fel_stp, 'b': fbl_stp}, lmax_qlm=sc.lmax_qlm)
        np.savetxt(TEMP + '/resp_grad_%s.dat' % qe_key, np.array([respG_len, respC_len]).transpose())

    if not os.path.exists(TEMP + '/resp_unl_%s.dat' % qe_key):
        respG_unl, respC_unl, _, _ = qresp.get_response(qe_key, sc.lmax_ivf_qe, 'p', cls_weights_unl, cls_unl,
                                                        {'t': ftl_stp_unl, 'e': fel_stp_unl, 'b': fbl_stp_unl}, lmax_qlm=sc.lmax_qlm)
        np.savetxt(TEMP + '/resp_unl_%s.dat' % qe_key, np.array([respG_unl, respC_unl]).transpose())


def init(TEMP, qe_key, sc):
    set_all_responses(TEMP, sc)
    respG_len, respC_len = np.loadtxt(TEMP + '/resp_len_%s.dat' % qe_key).transpose()
    respG_unl, respC_unl = np.loadtxt(TEMP + '/resp_unl_%s.dat' % qe_key).transpose()
    respG_grad, respC_grad = np.loadtxt(TEMP + '/resp_grad_%s.dat' % qe_key).transpose()

    lmax = min(sc.lmax_transf, sc.lmax_filt)
    cls_ivfs = {
        'tt': utils.cli(cls_unl['tt'][:lmax + 1] + (sc.nlev_t / 60. / 180. * np.pi) ** 2 / sc.transf[:lmax+ 1] ** 2),
        'ee': utils.cli(cls_unl['ee'][:lmax + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:lmax + 1] ** 2),
        'bb': utils.cli(cls_unl['bb'][:lmax + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:lmax + 1] ** 2)}

    if not os.path.exists(TEMP + '/mfresp_unl_%s.dat' % qe_key):
        cls_cmb = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
        for cl in cls_ivfs.values():
            cl[:sc.lmin_ivf_qe] *= 0.
        np.savetxt(TEMP + '/mfresp_unl_%s_cmblmax%s.dat' % (qe_key, len(cls_cmb['tt']-1)),
                   np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(sc.lmax_filt, sc.lmax_transf), sc.lmax_qlm)).transpose())
        for cl in cls_cmb.values():
            cl[sc.lmax_filt + 1:] *= 0.
        np.savetxt(
            TEMP + '/mfresp_unl_%s.dat' % qe_key,
            np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(sc.lmax_filt, sc.lmax_transf), sc.lmax_qlm)).transpose())

    if not os.path.exists(TEMP + '/mfresp_len_%s.dat' % qe_key):
        cls_cmb = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
        for cl in cls_ivfs.values():
            cl[:sc.lmin_ivf_qe] *= 0.
        for cl in cls_cmb.values():
            cl[sc.lmax_filt + 1:] *= 0.
        np.savetxt(
            TEMP + '/mfresp_len_%s.dat' % qe_key,
            np.array(qresp.get_mf_resp(qe_key, cls_cmb, cls_ivfs, min(sc.lmax_filt, sc.lmax_transf), sc.lmax_qlm)).transpose())

    N0_len = utils.cli(respG_len)
    H0_unl = respG_unl
    cpp = np.copy(cls_unl['pp'][:sc.lmax_qlm + 1])
    #--- trying to kill L = 1, 2 and 3 with prior
    cpp[:4] *= 1e-5
    clwf = cpp * utils.cli(cpp + N0_len)
    qnorm = utils.cli(respG_grad)

    return N0_len, H0_unl, cpp, clwf, qnorm


def get_qlms_dd(TEMP, sc):
    #------- For sep_TP
    ftl_stp_nocut = utils.cli(cls_len['tt'][:sc.lmax_ivf_qe + 1] + (sc.nlev_t / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe+1]  ** 2)
    fel_stp_nocut = utils.cli(cls_len['ee'][:sc.lmax_ivf_qe + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe+1]  ** 2)
    fbl_stp_nocut = utils.cli(cls_len['bb'][:sc.lmax_ivf_qe + 1] + (sc.nlev_p / 60. / 180. * np.pi) ** 2 / sc.transf[:sc.lmax_ivf_qe+1]  ** 2)
    filt_t = np.ones(sc.lmax_ivf_qe + 1, dtype=float); filt_t[:sc.lmin_ivf_qe] = 0.
    filt_e = np.ones(sc.lmax_ivf_qe + 1, dtype=float); filt_e[:sc.lmin_ivf_qe] = 0.
    filt_b = np.ones(sc.lmax_ivf_qe + 1, dtype=float); filt_b[:sc.lmin_ivf_qe] = 0.

    ftl_stp = ftl_stp_nocut * filt_t
    fel_stp = fel_stp_nocut * filt_e
    fbl_stp = fbl_stp_nocut * filt_b

    #------- We cache all modes inclusive of the low-ell, and then refilter on the fly with filt_t, filt_e, filt_b
    libdir_cinvt = os.path.join(TEMP, 'cinvt')
    libdir_cinvp = os.path.join(TEMP, 'cinvpOBD')
    libdir_ivfs = os.path.join(TEMP, 'ivfsOBD')

    chain_descr = [[0, ["diag_cl"], sc.lmax_ivf_qe, sc.nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

    ninv_t = [ivmat_path]
    cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_t,
                            marge_monopole=True, marge_dipole=True, marge_maps=[], chain_descr=chain_descr)

    ninv_p = [[ivmap_path]]
    cinv_p = iterc_cinv_p.cinv_p(libdir_cinvp, lmax_ivf_qe, nside, cls_len, transf[:lmax_ivf_qe+1], ninv_p,
                                    chain_descr=chain_descr, bmarg_lmax=BMARG_LCUT, zbounds=zbounds, _bmarg_lib_dir=BMARG_LIBDIR)

    ivfs_raw = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cls_len)

    ivfs = filt_util.library_ftl(ivfs_raw, lmax_ivf_qe, filt_t, filt_e, filt_b)

    #-----: This is a filtering instance shuffling simulation indices according to 'ss_dict'.
    ddresp = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp_dd_stp'), lmax_ivf_qe, cls_weights_qe, cls_grad,
                                {'tt':ftl_stp.copy(), 'ee':fel_stp.copy(),'bb':fbl_stp.copy()}, lmax_qlm)
    qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_ddOBD'), ivfs, ivfs, cls_len['te'], nside,
                                        lmax_qlm=lmax_qlm, resplib=ddresp)   
    return qlms_dd


def get_wflm0(DATIDX):
    wflm0 = lambda : alm_copy(ivfs_raw.get_sim_emliklm(DATIDX), lmax=lmax_filt)
    return wflm0
