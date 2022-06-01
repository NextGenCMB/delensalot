#!/usr/bin/env python

"""handler.py: This module receives input from lerepi.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


from os.path import join as opj
import logging
from logdecorator import log_on_start, log_on_end

import numpy as np

from plancklens import utils, qresp
from plancklens.helpers import mpi

from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators.statics import rec as Rec
from lenscarf.iterators import iteration_handler


class MAP_delensing():
    def __init__(self, qe, lensing_config):
        self.qe = qe
        self.lensing_config = lensing_config
        self.libdir_iterators = lambda qe_key, simidx, version: opj(lensing_config.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.ith = iteration_handler.transformer(lensing_config.iterator)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        jobs = []
        for idx in np.arange(self.lensing_config.imin, self.lensing_config.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.lensing_config.k, idx, self.lensing_config.v)
            if Rec.maxiterdone(lib_dir_iterator) < self.lensing_config.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.lensing_config.k, idx, self.lensing_config.v)
            if self.lensing_config.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < self.lensing_config.itmax:
                itlib = self.ith(self.qe, self.lensing_config.k, idx, self.lensing_config.v, self.libdir_iterators, self.lensing_config, self.lensing_config, self.lensing_config)
                itlib_iterator = itlib.get_iterator()
                for i in range(self.lensing_config.itmax + 1):
                    # print("Rank {} with size {} is starting iteration {}".format(mpi.rank, mpi.size, i))
                    print("****Iterator: setting cg-tol to %.4e ****"%self.lensing_config.tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%self.lensing_config.soltn_cond(i))
                    itlib_iterator.chain_descr  = self.lensing_config.chain_descr(self.lensing_config.lmax_unl, self.lensing_config.tol_iter(i))
                    itlib_iterator.soltn_cond   = self.lensing_config.soltn_cond(i)
                    print("doing iter " + str(i))
                    itlib_iterator.iterate(i, 'p')


class QE_delensing():
    def __init__(self, lensing_config):
        self.lensing_config = lensing_config
        self.mf0 = self.get_meanfield_it0()
        self.plm0 = self.get_plm_it0()
        self.mf_resp = self.get_meanfield_response_it0()
        self.wflm0 = lambda simidx: alm_copy(lensing_config.ivfs.get_sim_emliklm(simidx), None, lensing_config.lmax_unl, lensing_config.mmax_unl)
        self.R_unl = qresp.get_response(self.k, self.lensing_config.lmax_ivf, 'p', self.lensing_config.cls_unl, self.lensing_config.cls_unl,  {'e': self.lensing_config.fel_unl, 'b': self.lensing_config.fbl_unl, 't':self.lensing_config.ftl_unl}, lmax_qlm=self.lensing_config.lmax_qlm)[0]


    @log_on_start(logging.INFO, "Start of get_meanfield_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_it0()")
    def get_meanfield_it0(self):
        # QE mean-field fed in as constant piece in the iteration steps:
        mf_sims = np.unique(self.lensing_config.mc_sims_mf_it0 if not 'noMF' in self.version else np.array([]))
        mf0 = self.lensing_config.qlms_dd.get_sim_qlm_mf(self.k, mf_sims)  # Mean-field to subtract on the first iteration:
        if self.simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(mf_sims)
            mf0 = (mf0 - self.lensing_config.qlms_dd.get_sim_qlm(self.k, int(self.simidx)) / Nmf) * (Nmf / (Nmf - 1))
        
        return mf0


    @log_on_start(logging.INFO, "Start of get_plm_it0()")
    @log_on_end(logging.INFO, "Finished get_plm_it0()")
    def get_plm_it0(self):
        path_plm0 = opj(self.libdir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm0):
            # We now build the Wiener-filtered QE here since not done already
            plm0  = self.lensing_config.qlms_dd.get_sim_qlm(self.k, int(self.simidx))  #Unormalized quadratic estimate:
            plm0 -= self.mf0  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            R = qresp.get_response(self.k, self.lensing_config.lmax_ivf, 'p', self.lensing_config.cls_len, self.lensing_config.cls_len, {'e': self.lensing_config.fel, 'b': self.lensing_config.fbl, 't':self.lensing_config.ftl}, lmax_qlm=self.lensing_config.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.lensing_config.cpp * utils.cli(self.lensing_config.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, self.lensing_config.lmax_qlm, self.lensing_config.mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), self.lensing_config.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, self.lensing_config.mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, self.lensing_config.cpp > 0, self.lensing_config.mmax_qlm, True)
            np.save(path_plm0, plm0)

        return np.load(path_plm0)


    @log_on_start(logging.INFO, "Start of get_meanfield_response_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_response_it0()")
    def get_meanfield_response_it0(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version :
            mf_resp = qresp.get_mf_resp(self.k, self.lensing_config.cls_unl, {'ee': self.lensing_config.fel_unl, 'bb': self.lensing_config.fbl_unl}, self.lensing_config.lmax_ivf, self.lensing_config.lmax_qlm)[0]
        else:
            print('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lensing_config.lmax_qlm + 1, dtype=float)

        return mf_resp