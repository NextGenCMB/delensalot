#!/usr/bin/env python

"""handler.py: This module receives input from lerepi, handles D.lensalot jobs and runs them.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os
from os.path import join as opj
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import datetime
import getpass

import numpy as np
import healpy as hp

from plancklens import utils, qresp
from plancklens.helpers import mpi
from plancklens.sims import planck2018_sims

from MSC import pospace as ps

from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators.statics import rec as Rec
from lenscarf.iterators import iteration_handler
from lenscarf.opfilt.bmodes_ninv import template_bfilt

from lerepi.config.cmbs4.data import data_08d as sims_if


class OBD_builder():
    def __init__(self, OBD_model):
        self.__dict__.update(OBD_model.__dict__)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        # This is faking the collect/run structure, as bpl takes care of MPI 
        jobs = [1]
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        for job in self.jobs:
            bpl = template_bfilt(self.BMARG_LCUT, self.geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=self.TEMP)
            if not os.path.exists(self.TEMP + '/tnit.npy'):
                bpl._get_rows_mpi(self.ninv_p[0], prefix='')  # builds all rows in parallel
            mpi.barrier()
            if mpi.rank == 0:
                int(os.environ.get('OMP_NUM_THREADS', 32)) # TODO not sure if this resets anything..
                tnit = bpl._build_tnit()
                np.save(self.TEMP + '/tnit.npy', tnit)
                tniti = np.linalg.inv(tnit + np.diag((1. / (self.nlev_dep / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
                np.save(self.TEMP + '/tniti.npy', tniti)
                readme = '{}: This tniti has been created from user {} using lerepi/D.lensalot with the following settings: {}'.format(getpass.getuser(), datetime.date.today(), self.__dict__)
                np.save(self.TEMP + '/README.txt', readme)
                int(os.environ.get('OMP_NUM_THREADS', 8))  # TODO not sure if this resets anything..
        mpi.barrier()


class QE_lr():
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)

        self.mf0 = lambda simidx: self.get_meanfield_it0(simidx)
        self.plm0 = lambda simidx: self.get_plm_it0(simidx)
        self.mf_resp = lambda: self.get_meanfield_response_it0()
        self.wflm0 = lambda simidx: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lmax_unl, self.mmax_unl)
        self.R_unl = lambda: qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_unl, self.cls_unl,  {'e': self.fel_unl, 'b': self.fbl_unl, 't':self.ftl_unl}, lmax_qlm=self.lmax_qlm)[0]


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        jobs = []
        for idx in np.arange(self.imin, self.imax + 1):
            jobs.append(idx)
        self.jobs = jobs
        # TODO could check if mf has already been calculated, or all get_X() have created the files they need, before appending. Fixes (1)
        # for idx in self.mc_sims_mf_it0:
        #     jobs.append(idx)
        # self.jobs = jobs


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

        return self.qlms_dd.get_sim_qlm(self.k, idx)


    @log_on_start(logging.INFO, "Start of get_wflm0()")
    @log_on_end(logging.INFO, "Finished get_wflm0()")    
    def get_wflm0(self, simidx):

        return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lmax_unl, self.mmax_unl)


    @log_on_start(logging.INFO, "Start of get_R_unl()")
    @log_on_end(logging.INFO, "Finished get_R_unl()")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_unl, self.cls_unl,  {'e': self.fel_unl, 'b': self.fbl_unl, 't':self.ftl_unl}, lmax_qlm=self.lmax_qlm)[0]


    @log_on_start(logging.INFO, "Start of get_meanfield_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_it0()")
    def get_meanfield_it0(self, simidx):
        mf0 = self.qlms_dd.get_sim_qlm_mf(self.k, self.mc_sims_mf_it0)  # Mean-field to subtract on the first iteration:
        if simidx in self.mc_sims_mf_it0:  # We dont want to include the sim we consider in the mean-field...
            Nmf = len(self.mc_sims_mf_it0)
            mf0 = (mf0 - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))
        
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
            plm0  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            plm0 -= self.mf0(simidx)  # MF-subtracted unnormalized QE
            # Isotropic normalization of the QE
            R = qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_len, self.cls_len, {'e': self.fel, 'b': self.fbl, 't':self.ftl}, lmax_qlm=self.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, self.lmax_qlm, self.mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
            almxfl(plm0, utils.cli(R), self.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, self.mmax_qlm, True)           # Wiener-filter QE
            almxfl(plm0, self.cpp > 0, self.mmax_qlm, True)
            np.save(path_plm0, plm0)

        return np.load(path_plm0)


    # TODO this could be done before, inside p2d
    @log_on_start(logging.INFO, "Start of get_meanfield_response_it0()")
    @log_on_end(logging.INFO, "Finished get_meanfield_response_it0()")
    def get_meanfield_response_it0(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version :
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.fel_unl, 'bb': self.fbl_unl}, self.lmax_ivf, self.lmax_qlm)[0]
        else:
            print('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lmax_qlm + 1, dtype=float)
        return mf_resp


class MAP_lr():
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        # TODO Only needed to hand over to ith(). It would be better if a ith model was prepared already, and only hand over that
        self.dlensalot_model = dlensalot_model
        # TODO not entirely happy how QE dependence is put into MAP_lr but cannot think of anything better at the moment.
        self.qe = QE_lr(dlensalot_model)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
        self.ith = iteration_handler.transformer(self.iterator)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        self.qe.collect_jobs()
        jobs = []
        for idx in np.arange(self.imin, self.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
            if Rec.maxiterdone(lib_dir_iterator) < self.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        # TODO if all qe jobs already done before, this still takes a while. Fix, checking if files exist should be quick. (1)
        self.qe.run()
        # TODO within first srun, this doesn't start.. at least one job hangs in qe.
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
            if self.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < self.itmax:
                itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                itlib_iterator = itlib.get_iterator()
                for i in range(self.itmax + 1):
                    print("****Iterator: setting cg-tol to %.4e ****"%self.tol_iter(i))
                    print("****Iterator: setting solcond to %s ****"%self.soltn_cond(i))
                    itlib_iterator.chain_descr  = self.chain_descr(self.lmax_unl, self.tol_iter(i))
                    itlib_iterator.soltn_cond = self.soltn_cond(i)
                    itlib_iterator.iterate(i, 'p')


    @log_on_start(logging.INFO, "Start of init_ith()")
    @log_on_end(logging.INFO, "Finished init_ith()")
    def get_ith_sim(self, simidx):
        
        return self.ith(self.qe, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)


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
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        # TODO Only needed to hand over to ith(). It would be better if a ith model was prepared already, and only hand over that
        self.dlensalot_model = dlensalot_model
        self.qe = QE_lr(dlensalot_model)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
        self.ith = iteration_handler.transformer(self.iterator)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):
        self.qe.collect_jobs()
        jobs = []
        for idx in np.arange(self.imin, self.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
            if Rec.maxiterdone(lib_dir_iterator) >= self.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        self.qe.run()
        for idx in self.jobs[mpi.rank::mpi.size]:
            lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
            itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
            itlib_iterator = itlib.get_iterator()
            for it in range(0, self.itmax + 1):
                if it <= Rec.maxiterdone(lib_dir_iterator):
                    itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1)


class map_delensing():
    """Script for calculating delensed ILC and Blens spectra,
    using precaulculated Btemplates as input. Use 'Generate_Btemplate.py' for calulcating Btemplate input.
    """

    def __init__(self, bmd_model):
        self.bmd_model = bmd_model
        self.libdir_iterators = lambda qe_key, simidx, version: opj(bmd_model.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.lib_nm = dict()
        self.bcl_L_nm, self.bcl_cs_nm, self.bwfcl_cs_nm = np.zeros(shape=(len(bmd_model.nlevels),len(bmd_model.edges))), np.zeros(shape=(len(bmd_model.nlevels),len(bmd_model.edges))), np.zeros(shape=(len(bmd_model.nlevels),len(bmd_model.edges)))


    @log_on_start(logging.INFO, "Start of getfn_blm_lensc()")
    @log_on_end(logging.INFO, "Finished getfn_blm_lensc()")
    def getfn_blm_lensc(self, ana_p, simidx, it):
        '''Lenscarf output using Catherinas E and B maps'''
        # TODO remove hardcoding
        rootstr = '/global/cscratch1/sd/sebibel/cmbs4/'

        return rootstr+ana_p+'/p_p_sim%04d/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(simidx, it, it)

            
    @log_on_start(logging.INFO, "Start of getfn_qumap_cs()")
    @log_on_end(logging.INFO, "Finished getfn_qumap_cs()")
    def getfn_qumap_cs(self, simidx):
        '''Component separated polarisation maps lm, i.e. lenscarf input'''

        return self.bmd_model.sims.get_sim_pmap(simidx)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):
        jobs = []
        for idx in np.arange(self.bmd_model.imin, self.bmd_model.imax + 1):
            lib_dir_iterator = self.libdir_iterators(self.bmd_model.k, idx, self.bmd_model.version)
            if Rec.maxiterdone(lib_dir_iterator) >= self.bmd_model.itmax:
                jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        outputdata = np.zeros(shape=(6,len(self.bmd_model.nlevels),len(self.bmd_model.edges)-1))
        for nlev in self.bmd_model.nlevels:
            sims_may  = sims_if.ILC_May2022(self.bmd_model.fg, mask_suffix=int(nlev))
            nlev_mask = sims_may.get_mask() 
            self.lib_nm.update({nlev: ps.map2cl_binned(nlev_mask, self.bmd_model.clc_templ[:self.bmd_model.lmax_lib], self.bmd_model.edges, self.bmd_model.lmax_lib)})
        # TODO remove hardcoding
        dirroot = '/global/cscratch1/sd/sebibel/cmbs4/'+self.bmd_model.analysis_path+'/plotdata/'
        if not(os.path.isdir(dirroot + '{}'.format(self.bmd_model.dirid))):
            os.makedirs(dirroot + '{}'.format(self.bmd_model.dirid))

        for idx in self.jobs[mpi.rank::mpi.size]:
            file_op = dirroot + '{}'.format(self.bmd_model.dirid) + '/ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(idx, self.bmd_model.fg)
            print('will store file at:', file_op)
            
            qumap_cs_buff = self.getfn_qumap_cs(idx)
            eblm_cs_buff = hp.map2alm_spin(qumap_cs_buff*self.bmd_model.base_mask, 2, self.bmd_model.lmax_cl)
            bmap_cs_buff = hp.alm2map(eblm_cs_buff[1], self.bmd_model.nside)
            for nlevi, nlev in enumerate(self.bmd_model.nlevels):
                sims_may  = sims_if.ILC_May2022(self.bmd_model.fg, mask_suffix=int(nlev))
                nlev_mask = sims_may.get_mask() 
                bcl_cs_nm = self.lib_nm[nlev].map2cl(bmap_cs_buff)
                blm_L_buff = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(idx), lmax=self.bmd_model.lmax_cl), self.bmd_model.transf)
                bmap_L_buff = hp.alm2map(blm_L_buff, self.bmd_model.nside)
                bcl_L_nm = self.lib_nm[nlev].map2cl(bmap_L_buff)

                blm_lensc_MAP_buff = np.load(self.getfn_blm_lensc(self.bmd_model.analysis_path, idx, self.bmd_model.itmax))
                bmap_lensc_MAP_buff = hp.alm2map(blm_lensc_MAP_buff, nside=self.bmd_model.nside)
                blm_lensc_QE_buff = np.load(self.getfn_blm_lensc(self.bmd_model.analysis_path, idx, 0))
                bmap_lensc_QE_buff = hp.alm2map(blm_lensc_QE_buff, nside=self.bmd_model.nside)
    
                bcl_Llensc_MAP_nm = self.lib_nm[nlev].map2cl(bmap_L_buff-bmap_lensc_MAP_buff)    
                bcl_Llensc_QE_nm = self.lib_nm[nlev].map2cl(bmap_L_buff-bmap_lensc_QE_buff)

                bcl_cslensc_MAP_nm = self.lib_nm[nlev].map2cl(bmap_cs_buff-bmap_lensc_MAP_buff)
                bcl_cslensc_QE_nm = self.lib_nm[nlev].map2cl(bmap_cs_buff-bmap_lensc_QE_buff)
                
                outputdata[0][nlevi] = bcl_L_nm
                outputdata[1][nlevi] = bcl_cs_nm
                
                outputdata[2][nlevi] = bcl_Llensc_MAP_nm
                outputdata[3][nlevi] = bcl_cslensc_MAP_nm  
                
                outputdata[4][nlevi] = bcl_Llensc_QE_nm           
                outputdata[5][nlevi] = bcl_cslensc_QE_nm
            np.save(file_op, outputdata)
