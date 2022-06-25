#!/usr/bin/env python

"""handler.py: This module receives input from lerepi, handles D.lensalot jobs and runs them.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os, sys
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
from lenscarf.iterators.statics import rec as rec
from lenscarf.iterators import iteration_handler
from lenscarf.opfilt.bmodes_ninv import template_bfilt


class OBD_builder():
    def __init__(self, OBD_model):
        self.__dict__.update(OBD_model.__dict__)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        jobs = [1]
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        for job in self.jobs:
            bpl = template_bfilt(self.BMARG_LCUT, self.geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=self.TEMP)
            if not os.path.exists(self.TEMP + '/tnit.npy'):
                bpl._get_rows_mpi(self.ninv_p[0], prefix='')
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
        if 'overwrite_libdir' in dlensalot_model.__dict__:
            pass
        else:
            self.overwrite_libdir = None
        if self.overwrite_libdir is None:
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
            self.mf0 = lambda simidx: self.get_meanfield_it0(simidx)
            self.plm0 = lambda simidx: self.get_plm_it0(simidx)
            self.mf_resp = lambda: self.get_meanfield_response_it0()
            self.wflm0 = lambda simidx: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lmax_unl, self.mmax_unl)
            self.R_unl = lambda: qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_unl, self.cls_unl,  {'e': self.fel_unl, 'b': self.fbl_unl, 't':self.ftl_unl}, lmax_qlm=self.lmax_qlm)[0]
        else:
            # TODO hack. Only want to access old s08b sim result lib and generate B wf
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'zb_terator_%s_%04d_nofg_OBD_solcond_3apr20'%(qe_key, simidx) + version)
      

    @log_on_start(logging.INFO, "Start of collect_jobs(): id = {id}, overwrite_libdir = {self.overwrite_libdir}")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self, id=''):
        if self.overwrite_libdir is None:
            mc_sims = self.mc_sims_mf_it0
            mf_fname = os.path.join(self.TEMP, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(mc_sims)))
            if os.path.isfile(mf_fname):
                # Can safely skip QE jobs. MF exists, so we know QE was run before
                self.jobs = []
            if id == "None":
                self.jobs = []
            if id == 'All':
                jobs = []
                for idx in np.arange(self.imin, self.imax + 1):
                    jobs.append(idx)
                self.jobs = jobs
        else:
            # TODO hack. Only want to access old s08b sim result lib and generate B wf
            jobs = []
            for idx in np.arange(self.imin, self.imax + 1):
                jobs.append(idx)
            self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        if self.overwrite_libdir is None:
            # TODO this triggers the creation of all files for the MAP input, defined by the job array. MAP later needs the corresponding values separately via the getter
            # Can I think of something better?
            for idx in self.jobs[mpi.rank::mpi.size]:
                logging.info('{}/{}, Starting job {}'.format(mpi.rank,mpi.size,idx))
                self.get_sim_qlm(idx)
                self.get_meanfield_response_it0()
                self.get_wflm0(idx)
                self.get_R_unl()
                self.get_B_wf(idx)
                logging.info('{}/{}, Finished job {}'.format(mpi.rank,mpi.size,idx))
            if len(self.jobs)>0:
                log.info('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                mpi.barrier()
                # Tunneling the meanfield-calculation, so only rank 0 calculates it. Otherwise,
                # some processes will try accessing it too fast, or calculate themselves, which results in
                # an io error
                log.info("Done waiting. Rank 0 going to calculate meanfield-file.. everyone else waiting.")
                if mpi.rank == 0:
                    self.get_meanfield_it0(mpi.rank)
                    log.info("rank finsihed calculating meanfield-file.. everyone else waiting.")
                mpi.barrier()
                log.info("Starting mf-calc task")
            for idx in self.jobs[mpi.rank::mpi.size]:
                self.get_meanfield_it0(idx)
                self.get_plm_it0(idx)
            if len(self.jobs)>0:
                log.info('{} finished qe mf-calc tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                mpi.barrier()
                log.info('All ranks finished qe mf-calc tasks.')
        else:
            # TODO hack. Only want to access old s08b sim result lib and generate B wf
            for idx in self.jobs[mpi.rank::mpi.size]:
                self.get_B_wf(idx)

    @log_on_start(logging.INFO, "Start of get_sim_qlm()")
    @log_on_end(logging.INFO, "Finished get_sim_qlm()")
    def get_sim_qlm(self, idx):

        return self.qlms_dd.get_sim_qlm(self.k, idx)


    @log_on_start(logging.INFO, "Start of get_B_wf()")
    @log_on_end(logging.INFO, "Finished get_B_wf()")    
    def get_B_wf(self, simidx):
        fn = self.libdir_iterators(self.k, simidx, self.version)+'/bwf_qe_%04d.npy'%simidx
        if os.path.isfile(fn):
            bwf = self.ivfs.get_sim_bmliklm(simidx)
        else:
            log.info(fn)
            bwf = self.ivfs.get_sim_bmliklm(simidx)
            np.save(fn, bwf)
        return bwf


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
        if self.mfvar == None:
            mf0 = self.qlms_dd.get_sim_qlm_mf(self.k, self.mc_sims_mf_it0)
            if simidx in self.mc_sims_mf_it0:
                Nmf = len(self.mc_sims_mf_it0)
                mf0 = (mf0 - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))
        else:
            mf0 = hp.read_alm(self.mfvar)

        return mf0


    @log_on_start(logging.INFO, "Start of get_plm_it0()")
    @log_on_end(logging.INFO, "Finished get_plm_it0()")
    def get_plm_it0(self, simidx):
        lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        if not os.path.exists(lib_dir_iterator):
            os.makedirs(lib_dir_iterator)
        path_plm0 = opj(lib_dir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm0):
            plm0  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            plm0 -= self.mf0(simidx)  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_len, self.cls_len, {'e': self.fel, 'b': self.fbl, 't':self.ftl}, lmax_qlm=self.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
            plm0 = alm_copy(plm0,  None, self.lmax_qlm, self.mmax_qlm)
            almxfl(plm0, utils.cli(R), self.mmax_qlm, True) # Normalized QE
            almxfl(plm0, WF, self.mmax_qlm, True) # Wiener-filter QE
            almxfl(plm0, self.cpp > 0, self.mmax_qlm, True)
            np.save(path_plm0, plm0)

        return np.load(path_plm0)


    # TODO this could be done before, inside p2d()
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
        # TODO Only needed to hand over to ith(). in p2d(), prepare an ith model for it
        self.dlensalot_model = dlensalot_model
        # TODO not entirely happy how QE dependence is put into MAP_lr but cannot think of anything better at the moment.
        self.qe = QE_lr(dlensalot_model)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
        self.ith = iteration_handler.transformer(self.iterator_typ)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        jobs = list(range(len(self.tasks)))
        for taski, task in enumerate(self.tasks):
            _jobs = []
            if task == 'calc_meanfield_it':
                self.qe.collect_jobs()
                if rec.maxiterdone(lib_dir_iterator) < self.itmax:
                    _jobs.append(0)
            elif task == 'calc_btemplate':
                self.qe.collect_jobs()
                for idx in np.arange(self.imin, self.imax + 1):
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if rec.maxiterdone(lib_dir_iterator) >= self.itmax:
                        _jobs.append(idx)
            elif task == 'lens_rec':
                self.qe.collect_jobs()
                for idx in np.arange(self.imin, self.imax + 1):
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        _jobs.append(idx)

            jobs[taski] = _jobs
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        for taski, task in enumerate(self.tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))
            if task == 'calc_meanfield':
                self.qe.run()
                self.get_meanfields_it(np.arange(self.itmax+1), calc=True)
                mpi.barrier()
            elif task == 'calc_btemplate':
                self.qe.run()
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if self.dlm_mod_bool:
                        dlm_mod = self.get_meanfields_it(np.arange(self.itmax+1), calc=False)
                        # assuming mf includes all plms from simindices in config
                        dlm_mod = (dlm_mod * (self.imax - self.imin) - rec.load_plms(lib_dir_iterator, np.arange(self.itmax+1)))/(self.nsims_mf-1)
                    else:
                        dlm_mod = [None for n in range(self.itmax+1)]
                    itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                    itlib_iterator = itlib.get_iterator()
                    for it in range(0, self.itmax + 1):
                        if it <= rec.maxiterdone(lib_dir_iterator):
                            if it == 0:
                                  # this has already been done at QE level
                                itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1, dlm_mod=None, calc=True)
                            else:
                                itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1, dlm_mod=dlm_mod[it], calc=True)
                    log.info("{}: finished simulation {}".format(mpi.rank, idx))
            elif task == 'lens_rec':
                self.qe.run()
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if self.itmax >= 0 and rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                        itlib_iterator = itlib.get_iterator()
                        for i in range(self.itmax + 1):
                            log.info("****Iterator: setting cg-tol to %.4e ****"%self.tol_iter(i))
                            log.info("****Iterator: setting solcond to %s ****"%self.soltn_cond(i))
                            itlib_iterator.chain_descr  = self.chain_descr(self.lmax_unl, self.tol_iter(i))
                            itlib_iterator.soltn_cond = self.soltn_cond(i)
                            itlib_iterator.iterate(i, 'p')
                            log.info('{}, simidx {} done with iteration {}'.format(mpi.rank, idx, i))


    @log_on_start(logging.INFO, "Start of init_ith()")
    @log_on_end(logging.INFO, "Finished init_ith()")
    def get_ith_sim(self, simidx):
        
        return self.ith(self.qe, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)


    @log_on_start(logging.INFO, "Start of get_plm_it()")
    @log_on_end(logging.INFO, "Finished get_plm_it()")
    def get_plm_it(self, simidx, iterations):

        plms = rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), iterations)

        return plms


    @log_on_start(logging.INFO, "Start of get_meanfield_it()")
    @log_on_end(logging.INFO, "Finished get_meanfield_it()")
    def get_meanfield_it(self, iteration, calc=False):
        fn = opj(self.TEMP, 'mf', 'mf_it%03d.npy'%(iteration))
        if not calc:
            if os.path.isfile(fn):
                mf = np.load(fn)
            else:
                self.get_meanfield_it(self, iteration, calc=True)
        else:
            plm = rec.load_plms(self.libdir_iterators(self.k, 0, self.version), [0])[-1]
            
            mf = np.zeros_like(plm)
            for simidx in range(self.imin, self.imax):
                log.info(" iteration {}: simulation {}/{} done".format(iteration, simidx, self.imax-self.imin))
                mf += rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), [iteration])[-1]
                np.save(fn, mf/(self.imax-self.imin))

        return mf


    @log_on_start(logging.INFO, "Start of get_meanfields_it()")
    @log_on_end(logging.INFO, "Finished get_meanfields_it()")
    def get_meanfields_it(self, iterations, calc=False):
        plm = rec.load_plms(self.libdir_iterators(self.k, 0, self.version), [0])[-1]
        mfs = np.zeros(shape=(len(iterations),*plm.shape), dtype=np.complex128)
        if calc==True:
            for iti, it in enumerate(iterations[mpi.rank::mpi.size]):
                mfs[iti] = self.get_meanfield_it(it, calc=calc)
            mpi.barrier()
        for iti, it in enumerate(iterations[mpi.rank::mpi.size]):
            mfs[iti] = self.get_meanfield_it(it, calc=False)

        return mfs


class Inspector():
    def __init__(self, qe, model):

        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):

        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):

        assert 0, "Implement if needed"


class Map_delenser():
    """Script for calculating delensed ILC and Blens spectra using precaulculated Btemplates as input.
    """

    def __init__(self, bmd_model):
        self.__dict__.update(bmd_model.__dict__)

        # TODO hack. Remove and think of a better way of including old data without existing config file
        if 'libdir_iterators' in bmd_model.__dict__:
            pass
        else:
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.lib = dict()


    # @log_on_start(logging.INFO, "Start of getfn_blm_lensc()")
    # @log_on_end(logging.INFO, "Finished getfn_blm_lensc()")
    def getfn_blm_lensc(self, simidx, it):
        '''Lenscarf output using Catherinas E and B maps'''
        # TODO this needs cleaner implementation
        if self.libdir_iterators == 'overwrite':
            if it==12:
                rootstr = '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/lt_recons/'
                if self.fg == '00':
                    return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '07':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '09':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            elif it==0:
                return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(self.fg, simidx, simidx)    
        else:
            # TODO this belongs via config to p2d
            if self.dlm_mod_bool:
                return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod.npy'%(it, it)
            else:
                return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(it, it)

            
    # @log_on_start(logging.INFO, "Start of getfn_qumap_cs()")
    # @log_on_end(logging.INFO, "Finished getfn_qumap_cs()")
    def getfn_qumap_cs(self, simidx):

        '''Component separated polarisation maps lm, i.e. lenscarf input'''

        return self.sims.get_sim_pmap(simidx)


    # @log_on_start(logging.INFO, "Start of getfn_qumap_cs()")
    # @log_on_end(logging.INFO, "Finished getfn_qumap_cs()")
    def get_B_wf(self, simidx):
        '''Component separated polarisation maps lm, i.e. lenscarf input'''
        # TODO this is a quickfix and works only for already existing bwflm's for 08bb
        bw_fn = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/bwf_qe_%04d.npy'%(self.fg,simidx,simidx)
        if os.path.isfile(bw_fn):

            return np.load(bw_fn)
        else:
            assert 0, "File {} doesn't exist".format(bw_fn)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs(): {self.jobs}")
    def collect_jobs(self):
        jobs = []
        for idx in np.arange(self.imin, self.imax + 1):
            # Overwriting test
            if self.libdir_iterators == 'overwrite':
                jobs.append(idx)
            else:
                if idx not in self.droplist:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    log.info(lib_dir_iterator)
                    log.info(rec.maxiterdone(lib_dir_iterator))
                    if rec.maxiterdone(lib_dir_iterator) >= self.iterations[-1]:
                        jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        if self.jobs != []:
            for edgesi, edges in enumerate(self.edges):
                outputdata = np.zeros(shape=(2, 2+len(self.iterations), len(self.nlevels), len(edges)-1))
                for nlev in self.nlevels:
                    self.lib.update({nlev: ps.map2cl_binned(self.nlev_mask[nlev], self.clc_templ[:self.lmax_lib], edges, self.lmax_lib)})

                for idx in self.jobs[mpi.rank::mpi.size]:
                    _file_op = self.file_op(idx, self.fg, edgesi)
                    log.info('will store file at: {}'.format(_file_op))

                    qumap_cs_buff = self.getfn_qumap_cs(idx)
                    eblm_cs_buff = hp.map2alm_spin(qumap_cs_buff*self.base_mask, 2, self.lmax_cl)
                    bmap_cs_buff = hp.alm2map(eblm_cs_buff[1], self.nside)

                    blm_lensc_QE_buff = np.load(self.getfn_blm_lensc(idx, 0))
                    bmap_lensc_QE_buff = hp.alm2map(blm_lensc_QE_buff, nside=self.nside)

                    if self.getfn_blm_lensc(idx, 0).endswith('npy'):
                        blm_lensc_MAP_buff = np.array([np.load(self.getfn_blm_lensc(idx, it)) for it in self.iterations])
                    else:
                        blm_lensc_MAP_buff = np.array([hp.read_alm(self.getfn_blm_lensc(idx, it)) for it in self.iterations])
                    bmap_lensc_MAP_buff = np.array([hp.alm2map(blm_lensc_MAP_buff[iti], nside=self.nside) for iti in range(len(self.iterations))])
                    for nlevi, nlev in enumerate(self.nlevels):
                        bcl_cs = self.lib[nlev].map2cl(bmap_cs_buff)
                        # TODO fiducial choice should happen at transformer
                        blm_L_buff = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(idx), lmax=self.lmax_cl), self.transf)
                        bmap_L_buff = hp.alm2map(blm_L_buff, self.nside)
                        bcl_L = self.lib[nlev].map2cl(bmap_L_buff)

                        outputdata[0][0][nlevi] = bcl_L
                        outputdata[1][0][nlevi] = bcl_cs

                        bcl_Llensc_QE = self.lib[nlev].map2cl(bmap_L_buff-bmap_lensc_QE_buff)
                        bcl_cslensc_QE = self.lib[nlev].map2cl(bmap_cs_buff-bmap_lensc_QE_buff)

                        outputdata[0][1][nlevi] = bcl_Llensc_QE
                        outputdata[1][1][nlevi] = bcl_cslensc_QE

                        for iti, it in enumerate(self.iterations):
                            bcl_Llensc_MAP = self.lib[nlev].map2cl(bmap_L_buff-bmap_lensc_MAP_buff[iti])    
                            bcl_cslensc_MAP = self.lib[nlev].map2cl(bmap_cs_buff-bmap_lensc_MAP_buff[iti])

                            outputdata[0][2+iti][nlevi] = bcl_Llensc_MAP
                            outputdata[1][2+iti][nlevi] = bcl_cslensc_MAP      
            
                    np.save(_file_op, outputdata)
