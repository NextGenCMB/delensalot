#!/usr/bin/env python

"""handler.py: This module receives input from lerepi, handles delensing jobs and runs them.
    Two main classes define the delensing; the plancklens QE delenser, and the D.lensalot MAP delensing.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os
from os.path import join as opj
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np

from plancklens import utils, qresp
from plancklens.helpers import mpi

from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators.statics import rec as Rec
from lenscarf.iterators import iteration_handler


class QE_delensing():
    def __init__(self, dlensalot_model):
        self.libdir_iterators = lambda qe_key, simidx, version: opj(dlensalot_model.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.dlensalot_model = dlensalot_model
        self.version = dlensalot_model.version
        self.k = dlensalot_model.k

        self.mf0 = lambda simidx: self.get_meanfield_it0(simidx)
        self.plm0 = lambda simidx: self.get_plm_it0(simidx)
        self.mf_resp = lambda: self.get_meanfield_response_it0()
        self.wflm0 = lambda simidx: alm_copy(self.dlensalot_model.ivfs.get_sim_emliklm(simidx), None, self.dlensalot_model.lmax_unl, self.dlensalot_model.mmax_unl)
        self.R_unl = lambda: qresp.get_response(self.k, self.dlensalot_model.lmax_ivf, 'p', self.dlensalot_model.cls_unl, self.dlensalot_model.cls_unl,  {'e': self.dlensalot_model.fel_unl, 'b': self.dlensalot_model.fbl_unl, 't':self.dlensalot_model.ftl_unl}, lmax_qlm=self.dlensalot_model.lmax_qlm)[0]


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        jobs = []
        # TODO could check if mf has already been calculated
        for idx in self.dlensalot_model.mc_sims_mf_it0:
            jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        # TODO this triggers the creation of all files for the MAP input, defined by the job array. MAP later needs to request the corresponding values separately via the getter
        # Can I think of something better?
        for idx in self.jobs[mpi.rank::mpi.size]:
            logging.info('{}/{}, Starting job {}'.format(mpi.rank,mpi.size,idx))
            self.get_sim_qlm(idx)
            self.get_meanfield_response_it0()
            self.get_wflm0(idx)
            self.get_R_unl()
            logging.info('{}/{}, Finished job {}'.format(mpi.rank,mpi.size,idx))
        print('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
        mpi.barrier()
        print('All ranks finished qe ivfs tasks.')
        for idx in self.jobs[mpi.rank::mpi.size]:
            self.get_meanfield_it0(idx)
            self.get_plm_it0(idx)
        print('{} finished qe mf-calc tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
        mpi.barrier()
        print('All ranks finished qe mf-calc tasks.')


    @log_on_start(logging.INFO, "Start of get_sim_qlm()")
    @log_on_end(logging.INFO, "Finished get_sim_qlm()")
    def get_sim_qlm(self, idx):

        return self.dlensalot_model.qlms_dd.get_sim_qlm(self.k, idx)


    @log_on_start(logging.INFO, "Start of get_wflm0()")
    @log_on_end(logging.INFO, "Finished get_wflm0()")    
    def get_wflm0(self, simidx):

        return lambda: alm_copy(self.dlensalot_model.ivfs.get_sim_emliklm(simidx), None, self.dlensalot_model.lmax_unl, self.dlensalot_model.mmax_unl)


    @log_on_start(logging.INFO, "Start of get_R_unl()")
    @log_on_end(logging.INFO, "Finished get_R_unl()")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.dlensalot_model.lmax_ivf, 'p', self.dlensalot_model.cls_unl, self.dlensalot_model.cls_unl,  {'e': self.dlensalot_model.fel_unl, 'b': self.dlensalot_model.fbl_unl, 't':self.dlensalot_model.ftl_unl}, lmax_qlm=self.dlensalot_model.lmax_qlm)[0]


    @log_on_start(logging.INFO, "Start of get_meanfield_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_it0()")
    def get_meanfield_it0(self, simidx):
        mf0 = self.dlensalot_model.qlms_dd.get_sim_qlm_mf(self.k, self.dlensalot_model.mc_sims_mf_it0)  # Mean-field to subtract on the first iteration:
        if simidx in self.dlensalot_model.mc_sims_mf_it0:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(self.dlensalot_model.mc_sims_mf_it0)
            mf0 = (mf0 - self.dlensalot_model.qlms_dd.get_sim_qlm(self.k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))
        
        return mf0


    @log_on_start(logging.INFO, "Start of get_plm_it0()")
    @log_on_end(logging.INFO, "Finished get_plm_it0()")
    def get_plm_it0(self, simidx):
        lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        if not os.path.exists(lib_dir_iterator):
            os.makedirs(lib_dir_iterator)
        path_plm0 = opj(lib_dir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm0):
            # We now build the Wiener-filtered QE here since not done already
            plm0  = self.dlensalot_model.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            plm0 -= self.mf0(simidx)  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            R = qresp.get_response(self.k, self.dlensalot_model.lmax_ivf, 'p', self.dlensalot_model.cls_len, self.dlensalot_model.cls_len, {'e': self.dlensalot_model.fel, 'b': self.dlensalot_model.fbl, 't':self.dlensalot_model.ftl}, lmax_qlm=self.dlensalot_model.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.dlensalot_model.cpp * utils.cli(self.dlensalot_model.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, self.dlensalot_model.lmax_qlm, self.dlensalot_model.mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), self.dlensalot_model.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, self.dlensalot_model.mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, self.dlensalot_model.cpp > 0, self.dlensalot_model.mmax_qlm, True)
            np.save(path_plm0, plm0)

        return np.load(path_plm0)


    # TODO this could be done before, inside p2d
    @log_on_start(logging.INFO, "Start of get_meanfield_response_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_response_it0()")
    def get_meanfield_response_it0(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version :
            mf_resp = qresp.get_mf_resp(self.k, self.dlensalot_model.cls_unl, {'ee': self.dlensalot_model.fel_unl, 'bb': self.dlensalot_model.fbl_unl}, self.dlensalot_model.lmax_ivf, self.dlensalot_model.lmax_qlm)[0]
        else:
            print('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.dlensalot_model.lmax_qlm + 1, dtype=float)
        return mf_resp


class MAP_delensing():
    def __init__(self, dlensalot_model):
        # TODO not entirely happy how QE dependence is put into MAP_delensing but cannot think of anything better at the moment.
        self.qe = QE_delensing(dlensalot_model)
        self.dlensalot_model = dlensalot_model
        self.libdir_iterators = lambda qe_key, simidx, version: opj(dlensalot_model.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
        self.ith = iteration_handler.transformer(dlensalot_model.iterator)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        self.qe.collect_jobs()
        jobs = []
        for idx in np.arange(self.dlensalot_model.imin, self.dlensalot_model.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.dlensalot_model.k, idx, self.dlensalot_model.version)
            if Rec.maxiterdone(lib_dir_iterator) < self.dlensalot_model.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        self.qe.run()
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.dlensalot_model.k, idx, self.dlensalot_model.version)
            if self.dlensalot_model.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < self.dlensalot_model.itmax:
                itlib = self.ith(self.qe, self.dlensalot_model.k, idx, self.dlensalot_model.version, self.libdir_iterators, self.dlensalot_model)
                itlib_iterator = itlib.get_iterator()
                for i in range(self.dlensalot_model.itmax + 1):
                    print("****Iterator: setting cg-tol to %.4e ****"%self.dlensalot_model.tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%self.dlensalot_model.soltn_cond(i))
                    itlib_iterator.chain_descr  = self.dlensalot_model.chain_descr(self.dlensalot_model.lmax_unl, self.dlensalot_model.tol_iter(i))
                    itlib_iterator.soltn_cond   = self.dlensalot_model.soltn_cond(i)
                    itlib_iterator.iterate(i, 'p')


    @log_on_start(logging.INFO, "Start of init_ith()")
    @log_on_end(logging.INFO, "Finished init_ith()")
    def get_ith_sim(self, simidx):
        return self.ith(self.qe, self.dlensalot_model.k, simidx, self.dlensalot_model.version, self.libdir_iterators, self.dlensalot_model)


class inspect_result():
    def __init__(self, qe, dlensalot_model):
        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):
        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        assert 0, "Implement if needed"


# TODO B_template construction could be independent of the iterator(), and could be simplified when get_template_blm is moved to a 'query' module
class B_template_construction():
    def __init__(self, qe, dlensalot_model):
        self.qe = qe
        self.dlensalot_model = dlensalot_model
        self.libdir_iterators = lambda qe_key, simidx, version: opj(dlensalot_model.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
        self.ith = iteration_handler.transformer(dlensalot_model.iterator)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):
        jobs = []
        for idx in np.arange(self.dlensalot_model.imin, self.dlensalot_model.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.dlensalot_model.k, idx, self.dlensalot_model.version)
            if Rec.maxiterdone(lib_dir_iterator) == self.dlensalot_model.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.dlensalot_model.k, idx, self.dlensalot_model.version)
            if self.dlensalot_model.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) == self.dlensalot_model.itmax:
                itlib = self.ith(self.qe, self.dlensalot_model.k, idx, self.dlensalot_model.version, self.libdir_iterators, self.dlensalot_model)
                itlib_iterator = itlib.get_iterator()
                for it in range(0, self.dlensalot_model.itmax + 1):
                    itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1)

