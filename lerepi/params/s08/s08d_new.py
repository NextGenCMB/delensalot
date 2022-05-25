"""Iterative reconstruction for s08d fg 00 / 07, using Caterina ILC maps

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

import lenscarf
import lenscarf.interface_user as if_u 
from lenscarf.opfilt import interface_opfilt as if_o

from lenscarf import utils
from lenscarf import interface_qe as iqe
from lenscarf.iterators import steps

from lenscarf.iterators import cs_iterator

from lerepi.data.dc08 import data_08d as sims_if
from lerepi.survey_config.dc08 import sc_08d as sc

cls_path = os.path.join(os.path.dirname(lenscarf.__file__), 'data', 'cls')

qe_key = 'p_p'
fg = '00'
TEMP =  '/global/cscratch1/sd/sebibel/cmbs4/s08d/cILC_%s_test/'%fg

nsims = 100

sims = sims_if.ILC_May2022(fg)
mc_sims_mf = np.arange(nsims)

tol = 1e-3
tol_iter = lambda itr : 1e-3 if itr <= 10 else 1e-4 # The gradient spectrum seems to saturate with 1e-3 after roughly this number of iteration
soltn_cond = lambda itr: True

N0_len, H0_unl, cpp, clwf, qnorm = iqe.init(TEMP, qe_key, sc)
qlms_dd = iqe.get_qlms_dd(TEMP)
ivmat_path, ivmap_path = iqe.get_ivma_paths()
pixn_inv = [hp.read_map(ivmap_path)]

#TODO if this file isn't precalculated, either terminate param file, or turn this into mpi tasks.
mf0 = qlms_dd.get_sim_qlm_mf('p_p', mc_sims=mc_sims_mf)

def get_itlib(qe_key, DATIDX):
    TEMP_it = TEMP + '/iterator_'+qe_key+'_%04d_OBD'%DATIDX
    wflm0 = iqe.get_wflm0(DATIDX)
    if DATIDX in mc_sims_mf:
        mf0 =  (mf0 * len(mc_sims_mf) - qlms_dd.get_sim_qlm(qe_key, DATIDX)) / (len(mc_sims_mf) - 1.)
    plm0 = hp.almxfl(utils.alm_copy(qlms_dd.get_sim_qlm(qe_key, DATIDX), lmax=sc.lmax_qlm) - mf0, qnorm * clwf)
    dat = sims.get_sim_pmap(DATIDX)

    chain_descr = [[0, ["diag_cl"], sc.lmax_filt, sc.nside, np.inf, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

    cls_filt = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    itlib = cs_iterator.iterator(TEMP, cs)
    itlib =  cs_iterator.iterator_cstmf(TEMP_it ,'QU', dat, plm0, mf0, H0_unl, cpp, cls_filt, sc.lmax_filt, wflm0=wflm0, chain_descr=chain_descr,  ninv_filt=if_o.get_ninv_opfilt())
    itlib.newton_step_length = steps.bp(np.arange(4097), 400, 0.5, 1500, 0.1, scale=50)
    return itlib

if __name__ == '__main__':
    args = if_u.get_parser()
    jobs = if_u.collect_jobs(TEMP, args)
    if_u.run(TEMP, get_itlib, jobs, args)