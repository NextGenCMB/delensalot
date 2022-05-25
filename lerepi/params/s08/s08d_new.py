"""Iterative reconstruction for s08d, using Caterina ILC maps

Steps to run this successfully:
    1. calculate tniti
    2. get central noise level
    3. choose filter
    4. check dir structure
"""

import os, sys
import numpy as np
import healpy as hp

from plancklens.qcinv import cd_solve

#TODO only one or two imports would be appreciated
import lenscarf
import lenscarf.interface_user as if_u 
from lenscarf import utils
from lenscarf import interface_qe as iqe
from lenscarf.iterators import steps

from lenscarf.opfilt import opfilt_ee_wl
from lenscarf.iterators import cs_iterator

from lerepi.data.dc08 import data_08d as sims_if
from lerepi.survey_config.dc08 import sc_08d as sc

cls_path = os.path.join(os.path.dirname(lenscarf.__file__), 'data', 'cls')

qe_key = 'p_p'
fg = '00'
TEMP =  '/global/cscratch1/sd/sebibel/cmbs4/s08d/cILC_%s_test/'%fg

if not os.path.exists(TEMP):
    os.makedirs(TEMP)

nsims = 100

nlev_p = sc.THIS_CENTRALNLEV_UKAMIN 
nlev_t = nlev_p / np.sqrt(2.)

zbounds, zbounds_len = sc.get_zbounds()
transf = sc.transf
sims = sims_if.ILC_May2022(fg)
mc_sims_mf = np.arange(nsims)

tol=1e-3
tol_iter = lambda itr : 1e-3 if itr <= 10 else 1e-4 # The gradient spectrum seems to saturate with 1e-3 after roughly this number of iteration
soltn_cond = lambda itr: True

N0_len, H0_unl, cpp, clwf, qnorm = iqe.init(TEMP, qe_key, sc)
qlms_dd = iqe.get_qlms_dd(TEMP)
ivmat_path, ivmap_path = iqe.get_ivma_paths()
pixn_inv = [hp.read_map(ivmap_path)]

#TODO if this file isn't precalculated, either terminate param file, or turn this into mpi tasks.
mf0 = qlms_dd.get_sim_qlm_mf('p_p', mc_sims=mc_sims_mf)


def get_itlib(qe_key, DATIDX):

    assert qe_key == 'p_p'

    TEMP_it = TEMP + '/iterator_p_p_%04d_OBD'%DATIDX
    wflm0 = iqe.get_wflm0(DATIDX)

    if DATIDX in mc_sims_mf:
        mf0 =  (mf0 * len(mc_sims_mf) - qlms_dd.get_sim_qlm('p_p', DATIDX)) / (len(mc_sims_mf) - 1.)
    plm0 = hp.almxfl(utils.alm_copy(qlms_dd.get_sim_qlm(qe_key, DATIDX), lmax=sc.lmax_qlm) - mf0, qnorm * clwf)
    dat = sims.get_sim_pmap(DATIDX)

    def opfilt(libdir, plm, olm=None):
        return opfilt_ee_wl.alm_filter_ninv_wl(libdir, pixn_inv, transf, sc.lmax_filt, plm, bmarg_lmax=sc.BMARG_LCUT, _bmarg_lib_dir=sc.BMARG_LIBDIR,
                    olm=olm, sc.nside_lens=2048, nbands_lens=1, facres=-1,zbounds=zbounds, zbounds_len=zbounds_len)

    chain_descr = [[0, ["diag_cl"], sc.lmax_filt, sc.nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

    cls_filt = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))

    itlib =  cs_iterator.iterator_cstmf(TEMP_it ,{'p_p': 'QU', 'p': 'TQU', 'ptt': 'T'}[qe_key],
                        dat, plm0, mf0, H0_unl, cpp, cls_filt, sc.lmax_filt, wflm0=wflm0, chain_descr=chain_descr,  ninv_filt=opfilt)
    itlib.newton_step_length = steps.bp(np.arange(4097), 400, 0.5, 1500, 0.1, scale=50)
    return itlib

if __name__ == '__main__':
    args = if_u.get_parser()
    jobs = if_u.collect_jobs(TEMP, args)
    if_u.run(TEMP, args, jobs)