#!/usr/bin/env python

"""handler.py: This module collects the delensalot jobs. It receives the delensalot model build for the respective job. They all initialize needed modules and directories, collect the computing-jobs, and run the computing-jobs, with MPI support, if available.
    
"""
import os
from os.path import join as opj

import numpy as np
import healpy as hp

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from logdecorator import log_on_start, log_on_end
import datetime, getpass, copy, importlib

from plancklens import qresp, qest, qecl, utils
from plancklens.sims import maps, phas
from plancklens.qcinv import opfilt_pp
from plancklens.filt import filt_util, filt_cinv, filt_simple

from delensalot.utils import read_map
from delensalot.utility import utils_qe, utils_sims
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam

from delensalot.sims.generic import parameter_sims

from delensalot.config.visitor import transform, transform3d
from delensalot.config.config_helper import data_functions as df
from delensalot.config.metamodel import DEFAULT_NotAValue
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Concept

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI
from delensalot.core.opfilt.opfilt_handler import QE_transformer, MAP_transformer
from delensalot.core.iterator.iteration_handler import iterator_transformer
from delensalot.core.iterator.statics import rec as rec
from delensalot.core.decorator.exception_handler import base as base_exception_handler
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD
from delensalot.core.opfilt.bmodes_ninv import template_bfilt


class Basejob():
    """
    Base class for all jobs, i.e. convenience functions go in here as they should be accessible from anywhere
    """
    def __str__(self):
        _str = ''
        for key, val in self.__dict__.items():
            keylen = len(str(key))
            if type(val) in [list, np.ndarray, np.array, dict]:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(type(val))
            else:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(val)
            _str += '\n'
        return _str


    def __init__(self, model):
        self.__dict__.update(model.__dict__)

        self.libdir_QE = opj(self.TEMP, 'QE')
        if not os.path.exists(self.libdir_QE):
            os.makedirs(self.libdir_QE)
        self.libdir_MAP = lambda qe_key, simidx, version: opj(self.TEMP, 'MAP/%s'%(qe_key), 'sim%04d'%(simidx) + version)
        for simidx in self.simidxs:
            libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            if not os.path.exists(libdir_MAPidx):
                os.makedirs(libdir_MAPidx)
        self.libdir_MAP_blt = opj(self.TEMP, 'MAP/BLT/')
        if not os.path.exists(self.libdir_MAP_blt):
            os.makedirs(self.libdir_MAP_blt)
         
        self.config_model = model
        self.jobs = []


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):

        assert 0, "Overwrite"


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def run(self):

        assert 0, "Implement if needed"


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_qlm_it(self, simidx, it):

        assert 0, "Implement if needed"


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), its)

        return plms


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_mf_it(self, simidx, it, normalized=True):

        assert 0, "Implement if needed"


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_blt_it(self, simidx, it):
        # TODO if data on CFS (or elsewhere), I need to redirect this. but how to do it nicely?
        if self.data_from_CFS:
            fn_blt = self.libdir_blt_MAP_CFS(self.k, simidx, self.version)
        else:
            if it == 0:
                fn_blt = os.path.join(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
            elif it >0:
                fn_blt = os.path.join(self.libdir_MAP_blt, 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it, self.lm_max_blt[0]) + '.npy')
        return np.load(fn_blt)
    

    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_ivf(self, simidx, it, field):

        assert 0, "Implement if needed"


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_wf(self, simidx, it, field):

        assert 0, "Implement if needed"
    

    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def get_fiducial_sim(self, simidx, field):

        assert 0, "Implement if needed"


    #@log_on_start(logging.INFO, "get_filter() started")
    #@log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self): 
        assert 0, 'overwrite'


class OBD_builder(Basejob):
    """OBD matrix builder Job. Calculates the OBD matrix, used to correctly deproject the B-modes at a masked sky.
    """
    @check_MPI
    def __init__(self, OBD_model, diasable_mpi=False):
        self.__dict__.update(OBD_model.__dict__)
        # self.tpl = template_dense(self.lmin_teb[2], self.geom, self.tr, _lib_dir=self.libdir_QE, rescal=self.rescale)
        b_transf = gauss_beam(df.a2r(self.beam), lmax=self.lmax) # TODO ninv_p doesn't depend on this anyway, right?
        self.ninv_p = np.array(opfilt_pp.alm_filter_ninv(self.ninv_p_desc, b_transf, marge_qmaps=(), marge_umaps=()).get_ninv())


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished")
    def collect_jobs(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        jobs = [1]
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "run() started")
    #@log_on_end(logging.INFO, "run() finished")
    def run(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        for job in self.jobs:
            bpl = template_bfilt(self.lmin_b, self.geom, self.tr, _lib_dir=self.libdir)
            if not os.path.exists(self.libdir_QE+ '/tnit.npy'):
                bpl._get_rows_mpi(self.ninv_p, prefix='')
            mpi.barrier()
            if mpi.rank == 0:
                if not os.path.exists(self.libdir_QE+ '/tnit.npy'):
                    tnit = bpl._build_tnit()
                    np.save(self.libdir_QE+ '/tnit.npy', tnit)
                else:
                    tnit = np.load(self.libdir_QE+ '/tnit.npy')
                if not os.path.exists(self.libdir_QE+ '/tniti.npy'):
                    log.info('inverting')
                    tniti = np.linalg.inv(tnit + np.diag((1. / (self.nlev_dep / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
                    np.save(self.libdir_QE+ '/tniti.npy', tniti)
                    readme = '{}: tniti.npy. created from user {} using lerepi/delensalot with the following settings: {}'.format(getpass.getuser(), datetime.date.today(), self.__dict__)
                    with open(self.libdir_QE+ '/README.txt', 'w') as f:
                        f.write(readme)
                else:
                    log.info('Matrix already created')
        mpi.barrier()


class Sim_generator(Basejob):
    """Simulation generation Job. Generates simulations for the requested configuration.
        * If any libdir exists, then a flavour of data is provided. Therefore, can only check by making sure flavour == obs, and fns exist.
    """
    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)
        if self.simulationdata.flavour == 'obs' and self.simulationdata.libdir != DEFAULT_NotAValue and self.simulationdata.fns != DEFAULT_NotAValue:
            # Here, obs data is provided and nothing needs to be generated
            pass
        else:
            # some flavour may be provided, but we need to generate the obs maps from this. so Sim_generator() gets its own libdir and fns, and later updates simhandler with it.
            self.libdir = opj(os.environ['SCRATCH'], 'simulation/', str(self.simulationdata.geometry), str(self.simulationdata.nlev))
            self.fns = ['Qmapobs_{}.npy', 'Umapobs_{}.npy']
            first_rank = mpi.bcast(mpi.rank)
            if first_rank == mpi.rank:
                if not os.path.exists(self.libdir):
                    os.makedirs(self.libdir)
                for n in range(mpi.size):
                    if n != mpi.rank:
                        mpi.send(1, dest=n)
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)
        if self.simulationdata.flavour != 'sky':
            # it is handy to store the sky simulations, as often, only noise levels and beam change for an analysis.
            self.libdir_sky = opj(os.environ['SCRATCH'], 'simulation/', str(self.simulationdata.geometry))
            self.fns_sky = ['Qmapsky_{}.npy', 'Umapsky_{}.npy']
        
        # if Sim_generator() already produced the obs maps, then the jobmodel of course doesnt know of it, so need to update jobmodel manually. Since sim_lib hasnt gotten the libdir and fns info yet, need to check this manually. self.libdir and self.fns only exists when Sim_generator() believes it should generate data.
        if self.libdir != DEFAULT_NotAValue and self.fns != DEFAULT_NotAValue:
            simidxs_ = np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))))
            check_ = True  
            for simidx in simidxs_:
                if self.k in ['p_p', 'p_eb', 'peb', 'p_be']: 
                    if os.path.exists(opj(self.libdir, self.fns[0].format(simidx))) and os.path.exists(opj(self.libdir, self.fns[1].format(simidx))):
                        pass
                    else:
                        check_ = False
                if self.k in ['ptt', 'p']:
                    if os.path.exists(opj(self.libdir, self.fns.format(simidx))):
                        pass
                    else:
                        check_ = False
            if check_:
                self.postrun()


    # @base_exception_handler
    #@log_on_start(logging.INFO, "Sim.collect_jobs() started")
    #@log_on_end(logging.INFO, "Sim.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        jobs = []
        simidxs_ = np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))))
        for simidx in simidxs_:
            if not self.simulationdata.isdone(simidx, field='polarization', spin=2):
                jobs.append(simidx)
        self.jobs = jobs
        return self.jobs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "Sim.run() started")
    #@log_on_end(logging.INFO, "Sim.run() finished")
    def run(self):
        for simidx in self.jobs[mpi.rank::mpi.size]:
            log.info("rank {} (size {}) generating sim {}".format(mpi.rank, mpi.size, simidx))
            self.generate_sim(simidx)
            log.info("rank {} (size {}) generated sim {}".format(mpi.rank, mpi.size, simidx))
        self.postrun()


    def postrun(self):
        self.simulationdata.libdir = self.libdir
        self.simulationdata.fns = self.fns

        self.simulationdata.obs_lib.fns = self.fns
        self.simulationdata.obs_lib.libdir = self.libdir
        self.simulationdata.obs_lib.space = 'map'
        self.simulationdata.obs_lib.spin = 2

        if self.simulationdata.flavour != 'sky':
            # only overwrite if sky data has not been provided already
            self.simulationdata.len_lib.fns = self.fns_sky
            self.simulationdata.len_lib.libdir = self.libdir_sky
            self.simulationdata.len_lib.space = 'map'
            self.simulationdata.len_lib.spin = 2


    #@log_on_start(logging.INFO, "Sim.generate_sim(simidx={simidx}) started")
    #@log_on_end(logging.INFO, "Sim.generate_sim(simidx={simidx}) finished")
    def generate_sim(self, simidx):
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be']:
            QUobs = self.simulationdata.get_sim_obs(simidx, spin=2, space='map', field='polarization')
            np.save(opj(self.libdir, self.fns[0].format(simidx)), QUobs[0])
            np.save(opj(self.libdir, self.fns[1].format(simidx)), QUobs[1])

            if self.simulationdata.flavour != 'sky':
                # only save sky maps if data has not been provided by user. We only get here if sim_generator generates simulations.
                QUsky = self.simulationdata.get_sim_sky(simidx, spin=2, space='map', field='polarization')
                np.save(opj(self.libdir_sky, self.fns_sky[0].format(simidx)), QUsky[0])
                np.save(opj(self.libdir_sky, self.fns_sky[1].format(simidx)), QUsky[1])

        if self.k in ['ptt', 'p']:
            Tobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='map', field='temperature')
            np.save(opj(self.libdir, self.fn(simidx)), Tobs)

            if self.simulationdata.flavour != 'sky':
                # only save sky maps if data has not been provided by used. We only get here if sim_generator generates simulations.
                Tsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='map', field='temperature')
                np.save(opj(self.libdir_sky, self.fn_sky(simidx)), Tsky)


class QE_lr(Basejob):
    """Quadratic estimate lensing reconstruction Job. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """
    @check_MPI
    def __init__(self, dlensalot_model, caller=None):
        if caller is not None:
            dlensalot_model.qe_tasks = dlensalot_model.it_tasks
            ## TODO. Current solution to fake an iteration handler for QE to calc blt is to initialize one MAP_job here.
            ## In the future, I want to remove get_template_blm from the iteration_handler for QE.
            if 'calc_blt' in dlensalot_model.qe_tasks:
                self.MAP_job = caller

        super().__init__(dlensalot_model)
        self.dlensalot_model = dlensalot_model
        
        self.simgen = Sim_generator(dlensalot_model)
        # self.filter_ = transform(self.configfile.dlensalot_model, opfilt_handler_QE())

        if self.qe_filter_directional == 'isotropic':
            self.ivfs = filt_simple.library_fullsky_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
            if self.qlm_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        else:
            ## Wait for instantiation, as plancklens triggers cinv_calc...
            pass

        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.R_unl = lambda: qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.lm_max_qlm[0])[0]

        ## Faking here sims_MAP for calc_blt as it needs iteration_handler
        if 'calc_blt' in self.qe_tasks:
            if not os.path.exists(opj(self.libdir_QE, 'BLT')):
                os.makedirs(opj(self.libdir_QE, 'BLT'))
            if self.it_filter_directional == 'anisotropic':
                # TODO reimplement ztrunc
                self.sims_MAP = utils_sims.ztrunc_sims(self.simulationdata, self.nivjob_geominfo[1]['nside'], [self.zbounds])
            elif self.it_filter_directional == 'isotropic':
                self.sims_MAP = self.simulationdata


        if self.cl_analysis == True:
            # TODO fix numbers for mc ocrrection and total nsims
            self.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
            self.ds_dict = { k : -1 for k in range(300)}

            self.ivfs_d = filt_util.library_shuffle(self.ivfs, self.ds_dict)
            self.ivfs_s = filt_util.library_shuffle(self.ivfs, self.ss_dict)

            self.qlms_ds = qest.library_sepTP(opj(self.libdir_QE, 'qlms_ds'), self.ivfs, self.ivfs_d, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
            self.qlms_ss = qest.library_sepTP(opj(self.libdir_QE, 'qlms_ss'), self.ivfs, self.ivfs_s, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])

            self.mc_sims_bias = np.arange(60, dtype=int)
            self.mc_sims_var  = np.arange(60, 300, dtype=int)

            self.qcls_ds = qecl.library(opj(self.libdir_QE, 'qcls_ds'), self.qlms_ds, self.qlms_ds, np.array([]))  # for QE RDN0 calculations
            self.qcls_ss = qecl.library(opj(self.libdir_QE, 'qcls_ss'), self.qlms_ss, self.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
            self.qcls_dd = qecl.library(opj(self.libdir_QE, 'qcls_dd'), self.qlms_dd, self.qlms_dd, self.mc_sims_bias)
        self.filter = self.get_filter()


    def init_cinv(self):
        self.cinv_t = filt_cinv.cinv_t(opj(self.libdir_QE, 'cinv_t'),
            self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
            self.ttebl['t'], self.nivt_desc,
            marge_monopole=True, marge_dipole=True, marge_maps=[])

        transf_elm_loc = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_ivf[0])
        if self.OBD:
            self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                transf_elm_loc[:self.lm_max_ivf[0]+1], self.nivp_desc, geom=self.ninvjob_qe_geometry,
                chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                zbounds=self.zbounds, _bmarg_lib_dir=self.obd_libdir_QE, _bmarg_rescal=self.obd_rescale,
                sht_threads=self.tr)
        else:
            self.cinv_p = filt_cinv.cinv_p(opj(self.TEMP, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                self.ttebl['e'], self.nivp_desc, chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                transf_blm=self.ttebl['b'], marge_qmaps=(), marge_umaps=())


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.collect_jobs(recalc={recalc}) started")
    #@log_on_end(logging.INFO, "QE.collect_jobs(recalc={recalc}) finished: jobs={self.jobs}")
    def collect_jobs(self, recalc=False):

        if not isinstance(self.simulationdata, parameter_sims):
            self.simgen.collect_jobs()
        # qe_tasks overwrites task-list and is needed if MAP lensrec calls QE lensrec
        jobs = list(range(len(self.qe_tasks)))
        for taski, task in enumerate(self.qe_tasks):
            ## task_dependence
            ## calc_mf -> calc_phi, calc_blt -> calc_phi, (calc_mf)
            _jobs = []

            if task == 'calc_meanfield':
                fn_mf = os.path.join(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(self.simidxs_mf)))
                if not os.path.isfile(fn_mf) or recalc:
                    for simidx in self.simidxs_mf:
                        fn_qlm = os.path.join(self.qlms_dd.lib_dir, 'sim_%s_%04d.fits'%(self.k, simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                        if not os.path.isfile(fn_qlm) or recalc:
                            _jobs.append(int(simidx))

            ## Calculate realization dependent phi, i.e. plm_it000.
            if task == 'calc_phi':
                ## TODO this filename must match plancklens filename.. refactor
                fn_mf = os.path.join(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(self.simidxs_mf)))
                ## Skip if meanfield already calculated
                if not os.path.isfile(fn_mf) or recalc:
                    for simidx in self.simidxs:
                        fn_qlm = os.path.join(self.qlms_dd.lib_dir, 'sim_%s_%04d.fits'%(self.k, simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                        if not os.path.isfile(fn_qlm) or recalc:
                            _jobs.append(simidx)

            ## Calculate B-lensing template
            if task == 'calc_blt':
                for simidx in self.simidxs:
                    ## TODO this filename must match the one created in get_template_blm()... refactor..
                    fn_blt = os.path.join(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
                    if not os.path.isfile(fn_blt) or recalc:
                        _jobs.append(simidx)

            jobs[taski] = _jobs
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.run(task={task}) started")
    #@log_on_end(logging.INFO, "QE.run(task={task}) finished")
    def run(self, task=None):
        ## task may be set from MAP lensrec, as MAP lensrec has prereqs to QE lensrec
        ## if None, then this is a normal QE lensrec call
        if not isinstance(self.simulationdata, parameter_sims):
            self.simgen.run()


        # Only now instantiate filters
        if self.qe_filter_directional == 'anisotropic':
            self.init_cinv()
            _filter_raw = filt_cinv.library_cinv_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.cinv_t, self.cinv_p, self.cls_len)
            _ftl_rs = np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[0])
            _fel_rs = np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[1])
            _fbl_rs = np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[2])
            self.ivfs = filt_util.library_ftl(_filter_raw, self.lm_max_qlm[0], _ftl_rs, _fel_rs, _fbl_rs)
            self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])

        _tasks = self.qe_tasks if task is None else [task]
        
        for taski, task in enumerate(_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_meanfield':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    # In principle it is enough to calculate qlms. 
                    self.get_sim_qlm(int(idx))
                    log.info('{}/{}, finished job {}'.format(mpi.rank,mpi.size,idx))
                if len(self.jobs[taski])>0:
                    log.info('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                    mpi.barrier()
                    # Tunneling the meanfield-calculation, so only rank 0 calculates it. Otherwise,
                    # some processes will try accessing it too fast, or calculate themselves, which results in
                    # an io error
                    log.info("Done waiting. Rank 0 going to calculate meanfield-file.. everyone else waiting.")
                    if mpi.rank == 0:
                        self.get_meanfield(int(idx))
                        log.info("rank finished calculating meanfield-file.. everyone else waiting.")
                    mpi.barrier()

            if task == 'calc_phi':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.get_plm(idx, self.QE_subtract_meanfield)

            if task == 'calc_blt':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    # ## Faking here MAP filters
                    self.itlib_iterator = transform(self.MAP_job, iterator_transformer(self.MAP_job, simidx, self.dlensalot_model))
                    self.get_blt(simidx)


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_sim_qlm(simidx={simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_sim_qlm(simidx={simidx}) finished")
    def get_sim_qlm(self, simidx):

        return self.qlms_dd.get_sim_qlm(self.k, int(simidx))


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_wflm(simidx={simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_wflm(simidx={simidx}) finished")    
    def get_wflm(self, simidx):

        return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_R_unl() started")
    #@log_on_end(logging.INFO, "QE.get_R_unl() finished")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_meanfield(simidx={simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_meanfield(simidx={simidx}) finished")
    def get_meanfield(self, simidx):
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, 0))
        if self.Nmf > 0:
            if self.mfvar == None:
                # TODO hack: plancklens needs to be less restrictive with type for simidx. hack for now
                ret = self.qlms_dd.get_sim_qlm_mf(self.k, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
                if simidx in self.simidxs_mf:    
                    ret = (ret - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
            else:
                ret = hp.read_alm(self.mfvar)
                if simidx in self.simidxs_mf:    
                    ret = (ret - self.qlms_dd_mfvar.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
            return ret
        
        return ret
        

    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) started")
    #@log_on_end(logging.INFO, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) finished")
    def get_plm(self, simidx, sub_mf=True):
        libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
        fn_plm = opj(libdir_MAPidx, 'phi_plm_it000.npy') # Note: careful, this one doesn't have a simidx, so make sure it ends up in a simidx_directory (like MAP)
        if not os.path.exists(fn_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            if sub_mf and self.version != 'noMF':
                plm -= self.mf(int(simidx))  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
            plm = alm_copy(plm,  None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(plm, utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(plm, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.lm_max_qlm[1], True)
            np.save(fn_plm, plm)

        return np.load(fn_plm)


    #@log_on_start(logging.INFO, "QE.get_response_meanfield() started")
    #@log_on_end(logging.INFO, "QE.get_response_meanfield() finished")
    def get_response_meanfield(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.lm_max_qlm[0])[0]
        else:
            log.info('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lm_max_qlm[0] + 1, dtype=float)

        return mf_resp

    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_meanfield_normalized(simidx={simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_meanfield_normalized(simidx={simidx}) finished")
    def get_meanfield_normalized(self, simidx):
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = qresp.get_response(self.k, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
        WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
        almxfl(mf_QE, utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.cpp > 0, self.lm_max_qlm[1], True)

        return mf_QE


    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_blt({simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_blt({simidx}) finished")
    def get_blt(self, simidx):
        fn_blt = os.path.join(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            ## For QE, dlm_mod by construction doesn't do anything, because mean-field had already been subtracted from plm and we don't want to repeat that.
            dlm_mod = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, int(simidx)))
            blt = self.itlib_iterator.get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, dlm_mod=dlm_mod, perturbative=self.blt_pert)
            np.save(fn_blt, blt)
        return np.load(fn_blt)
    

    # @base_exception_handler
    #@log_on_start(logging.INFO, "QE.get_blt({simidx}) started")
    #@log_on_end(logging.INFO, "QE.get_blt({simidx}) finished")
    def get_blt_new(self, simidx):

        def get_template_blm(it, it_e, lmaxb=1024, lmin_plm=1, perturbative=False):
            fn_blt = 'blt_p%03d_e%03d_lmax%s'%(it, it_e, lmaxb)
            fn_blt += 'perturbative' * perturbative      

            elm_wf = self.filter.transf
            assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt
            mmaxb = lmaxb
            dlm = self.get_hlm(it, 'p')
            self.hlm2dlm(dlm, inplace=True)
            almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
            if perturbative: # Applies perturbative remapping
                get_alm = lambda a: elm_wf if a == 'e' else np.zeros_like(elm_wf)
                geom, sht_tr = self.filter.ffi.geom, self.filter.ffi.sht_tr
                d1 = geom.alm2map_spin([dlm, np.zeros_like(dlm)], 1, self.lmax_qlm, self.mmax_qlm, sht_tr, [-1., 1.])
                dp = utils_qe.qeleg_multi([2], +3, [utils_qe.get_spin_raise(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dm = utils_qe.qeleg_multi([2], +1, [utils_qe.get_spin_lower(2, self.lmax_filt)])(get_alm, geom, sht_tr)
                dlens = -0.5 * ((d1[0] - 1j * d1[1]) * dp + (d1[0] + 1j * d1[1]) * dm)
                del dp, dm, d1
                elm, blm = geom.map2alm_spin([dlens.real, dlens.imag], 2, lmaxb, mmaxb, sht_tr, [-1., 1.])
            else: # Applies full remapping (this will re-calculate the angles)
                ffi = self.filter.ffi.change_dlm([dlm, None], self.mmax_qlm)
                elm, blm = ffi.lensgclm(np.array([elm_wf, np.zeros_like(elm_wf)]), self.mmax_filt, 2, lmaxb, mmaxb)

                self.blt_cacher.cache(fn_blt, blm)

            return blm
        
        fn_blt = os.path.join(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            blt = get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, perturbative=self.blt_pert)
            np.save(fn_blt, blt)

        return np.load(fn_blt)


    #@log_on_start(logging.INFO, "get_filter() started")
    #@log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self): 
        QE_filters = transform(self, QE_transformer())
        filter = transform(self, QE_filters())
        return filter
    

class MAP_lr(Basejob):
    """Iterative lensing reconstruction Job. Depends on class QE_lr, and class Sim_generator.  Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """

    @check_MPI
    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)
        # TODO Only needed to hand over to ith(). in c2d(), prepare an ith model for it
        self.dlensalot_model = dlensalot_model
        
        # TODO This is not the prettiest way to provide MAP_lr with QE and Simgen dependency.. probably better to just put it as a separate job into the job-list.. so do this in config_handler... same with Sim_generator?
        self.simgen = Sim_generator(dlensalot_model)
        self.qe = QE_lr(dlensalot_model, caller=self)

        ## tasks -> mf_dirname
        if "calc_meanfield" in self.it_tasks or 'calc_blt' in self.it_tasks:
            if not os.path.isdir(self.mf_dirname) and mpi.rank == 0:
                os.makedirs(self.mf_dirname)

        # sims -> sims_MAP
        if self.it_filter_directional == 'anisotropic':
            self.sims_MAP = utils_sims.ztrunc_sims(self.simulationdata, self.nivjob_geominfo[1]['nside'], [self.zbounds])
            if self.k in ['ptt']:
                self.ninv = self.sims_MAP.ztruncify(read_map(self.nivt_desc)) # inverse pixel noise map on consistent geometry
            else:
                self.ninv = [self.sims_MAP.ztruncify(read_map(ni)) for ni in self.nivp_desc] # inverse pixel noise map on consistent geometry
        elif self.it_filter_directional == 'isotropic':
            self.sims_MAP = self.simulationdata
        self.filter = self.get_filter()

    # # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.map.collect_jobs() started")
    #@log_on_end(logging.INFO, "MAP.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        self.simgen.collect_jobs()
        self.qe.collect_jobs(recalc=False)
        jobs = list(range(len(self.it_tasks)))
        # TODO order of task list matters, but shouldn't
        for taski, task in enumerate(self.it_tasks):
            _jobs = []

            if task == 'calc_phi':
                ## Here I only want to calculate files not calculated before, and only for the it job tasks.
                ## i.e. if no blt task in iterator job, then no blt task in QE job 
                for simidx in self.simidxs:
                    libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    if rec.maxiterdone(libdir_MAPidx) < self.itmax:
                        _jobs.append(simidx)

            ## Calculate realization independent meanfields up to iteration itmax
            ## prereq: plms exist for itmax. maxiterdone won't work if calc_phi in task list
            elif task == 'calc_meanfield':
                # TODO need to make sure that all iterator wflms are calculated
                # either mpi.barrier(), or check all simindices TD(1)
                for simidx in self.simidxs_mf:
                    libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    if "calc_phi" in self.it_tasks:
                        _jobs.append(0)
                    elif rec.maxiterdone(libdir_MAPidx) < self.itmax:
                        _jobs.append(0)

            elif task == 'calc_blt':
                for simidx in self.simidxs:
                    fns_blt = np.array([os.path.join(self.libdir_MAP_blt, 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it, self.lm_max_blt[0]) + '.npy') for it in np.arange(1,self.itmax+1)])
                    if not np.all([os.path.exists(fn_blt) for fn_blt in fns_blt]):
                        _jobs.append(simidx)

            jobs[taski] = _jobs
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.run() started")
    #@log_on_end(logging.INFO, "MAP.run() finished")
    def run(self):
        for taski, task in enumerate(self.it_tasks):
            log.info('{}, task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))

            if task == 'calc_phi':
                self.simgen.run()
                self.qe.run(task=task)
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    if self.itmax >= 0 and rec.maxiterdone(libdir_MAPidx) < self.itmax:
                        itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
                        for it in range(self.itmax + 1):
                            itlib_iterator.chain_descr = self.it_chain_descr(self.lm_max_unl[0], self.it_cg_tol(it))
                            itlib_iterator.soltn_cond = self.soltn_cond(it)
                            itlib_iterator.iterate(it, 'p')
                            log.info('{}, simidx {} done with it {}'.format(mpi.rank, simidx, it))

            if task == 'calc_meanfield':
                self.qe.run(task=task)
                # TODO if TD(1) solved, replace np.arange() accordingly
                # TODO I don't like barriers and not sure if they are still needed
                mpi.barrier()
                self.get_meanfields_it(np.arange(self.itmax+1), calc=True)
                mpi.barrier()

            if task == 'calc_blt':
                self.qe.run(task=task)
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    self.itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
                    for it in range(self.itmax + 1):
                        self.get_blt_it(simidx, it)


    # # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.get_plm_it(simidx={simidx}, its={its}) started")
    #@log_on_end(logging.INFO, "MAP.get_plm_it(simidx={simidx}, its={its}) finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), its)

        return plms
    

    # # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.get_meanfield_it(it={it}, calc={calc}) started")
    #@log_on_end(logging.INFO, "MAP.get_meanfield_it(it={it}, calc={calc}) finished")
    def get_meanfield_it(self, it, calc=False):
        fn = opj(self.mf_dirname, 'mf%03d_it%03d.npy'%(self.Nmf, it))
        if not calc:
            if os.path.isfile(fn):
                mf = np.load(fn)
            else:
                mf = self.get_meanfield_it(self, it, calc=True)
        else:
            plm = rec.load_plms(self.libdir_MAP(self.k, self.simidxs[0], self.version), [0])[-1]
            mf = np.zeros_like(plm)
            for simidx in self.simidxs_mf:
                log.info("it {:02d}: adding sim {:03d}/{}".format(it, simidx, self.Nmf-1))
                mf += rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), [it])[-1]
            np.save(fn, mf/self.Nmf)

        return mf


    # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.get_meanfields_it(its={its}, calc={calc}) started")
    #@log_on_end(logging.INFO, "MAP.get_meanfields_it(its={its}, calc={calc}) finished")
    def get_meanfields_it(self, its, calc=False):
        plm = rec.load_plms(self.libdir_MAP(self.k, self.simidxs[0], self.version), [0])[-1]
        mfs = np.zeros(shape=(len(its),*plm.shape), dtype=np.complex128)
        if calc==True:
            for iti, it in enumerate(its[mpi.rank::mpi.size]):
                mfs[iti] = self.get_meanfield_it(it, calc=calc)
            mpi.barrier()
        for iti, it in enumerate(its[mpi.rank::mpi.size]):
            mfs[iti] = self.get_meanfield_it(it, calc=False)

        return mfs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "MAP.get_blt_it(simidx={simidx}, it={it}) started")
    #@log_on_end(logging.INFO, "MAP.get_blt_it(simidx={simidx}, it={it}) finished")
    def get_blt_it(self, simidx, it):
        if it == 0:
            self.qe.itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
            return self.qe.get_blt(simidx)
        fn_blt = os.path.join(self.libdir_MAP_blt, 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it, self.lm_max_blt[0]) + '.npy')
        if not os.path.exists(fn_blt):     
            self.libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            dlm_mod = np.zeros_like(rec.load_plms(self.libdir_MAPidx, [0])[0])
            if self.dlm_mod_bool and it>0 and it<=rec.maxiterdone(self.libdir_MAPidx):
                dlm_mod = self.get_meanfields_it([it], calc=False)
                if simidx in self.simidxs_mf:
                    dlm_mod = (dlm_mod - np.array(rec.load_plms(self.libdir_MAPidx, [it]))/self.Nmf) * self.Nmf/(self.Nmf - 1)
            if it<=rec.maxiterdone(self.libdir_MAPidx):
                blt = self.itlib_iterator.get_template_blm(it, it, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, dlm_mod=dlm_mod, perturbative=False)
                np.save(fn_blt, blt)
        return np.load(fn_blt)


    #@log_on_start(logging.INFO, "get_filter() started")
    #@log_on_end(logging.INFO, "get_filter() finished")
    def get_filter(self): 
        MAP_filters = transform(self, MAP_transformer())
        filter = transform(self, MAP_filters())
        return filter


class Map_delenser(Basejob):
    """Map delenser Job for calculating delensed ILC and Blens spectra using precaulculated Btemplates as input.
    This is a combination of,
     * delensing with Btemplates (QE, MAP),
     * choosing power spectrum calculation as in binning, masking, and templating
    """

    @check_MPI
    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)

        self.lib = dict()
        if 'nlevel' in self.binmasks:
            self.lib.update({'nlevel': {}})
        if 'mask' in self.binmasks:
            self.lib.update({'mask': {}})
        self.simgen = Sim_generator(dlensalot_model)
        self.libdir_delenser = opj(self.TEMP, 'delensing/{}'.format(self.dirid))
        if mpi.rank == 0:
            if not(os.path.isdir(self.libdir_delenser)):
                os.makedirs(self.libdir_delenser)
        self.fns = lambda simidx: opj(self.libdir_delenser, 'ClBB_sim%04d.npy'%(simidx))


    # @base_exception_handler
    #@log_on_start(logging.INFO, "collect_jobs() started")
    #@log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO a valid job is any requested job?, as BLTs may also be on CFS
        jobs = []
        for idx in self.simidxs:
            jobs.append(idx)
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    #@log_on_start(logging.INFO, "run() started")
    #@log_on_end(logging.INFO, "run() finished")
    def run(self):
        outputdata = self._prepare_job()
        if self.jobs != []:
            for simidx in self.jobs[mpi.rank::mpi.size]:
                log.debug('will store file at: {}'.format(self.fns(simidx)))
                self.delens(simidx, outputdata)


    def _prepare_job(self):
        if self.binning == 'binned':
            outputdata = np.zeros(shape=(2, 2+len(self.its), len(self.nlevels)+len(self.masks_fromfn), len(self.edges)-1))
            for maskflavour, masks in self.binmasks.items():
                for maskid, mask in masks.items():
                    ## for a future me: ell-max of clc_templ must be edges[-1], lmax_mask can be anything...
                    self.lib[maskflavour].update({maskid: self.cl_calc.map2cl_binned(mask, self.clc_templ, self.edges, self.lmax_mask)})
        elif self.binning == 'unbinned':
            for maskflavour, masks in self.binmasks.items():
                for maskid, mask in masks.items():
                    a = overwrite_anafast() if self.cl_calc == hp else masked_lib(mask, self.cl_calc, self.lmax, self.lmax_mask)
                    outputdata = np.zeros(shape=(2, 2+len(self.its), len(self.nlevels)+len(self.masks_fromfn), self.lmax+1))
                    self.lib[maskflavour].update({maskid: a})

        return outputdata
    

    # #@log_on_start(logging.INFO, "get_basemap() started")
    # #@log_on_end(logging.INFO, "get_basemap() finished")  
    def get_basemap(self, simidx):
        if self.basemap == 'lens':
            return almxfl(alm_copy(self.simulationdata.get_sim_sky(simidx, space='alm', spin=0, field='polarization')[1], self.simulationdata.lmax, *self.lm_max_blt), self.ttebl['e'], self.lm_max_blt[0], inplace=False) 
        else:
            return alm_copy(self.simulationdata.get_sim_obs(simidx, space='alm', spin=0, field='polarization')[1], self.simulationdata.lmax, *self.lm_max_blt)
    

    #@log_on_start(logging.INFO, "_delens() started")
    #@log_on_end(logging.INFO, "_delens() finished")
    def delens(self, simidx, outputdata):
        blm_L = self.get_basemap(simidx)
        blt_QE = self.get_blt_it(simidx, 0)
        
        bdel_QE = self.nivjob_geomlib.alm2map(blm_L-blt_QE, *self.lm_max_blt, nthreads=4)
        maskcounter = 0
        for maskflavour, masks in self.binmasks.items():
            for maskid, mask in masks.items():
                log.debug("starting mask {} {}".format(maskflavour, maskid))
                
                bcl_L = self.lib[maskflavour][maskid].map2cl(self.nivjob_geomlib.alm2map(blm_L, *self.lm_max_blt, nthreads=4))
                outputdata[0][0][maskcounter] = bcl_L

                blt_L_QE = self.lib[maskflavour][maskid].map2cl(bdel_QE)
                outputdata[0][1][maskcounter] = blt_L_QE

                for iti, it in enumerate(self.its):
                    blt_MAP = self.get_blt_it(simidx, it)
                    bdel_MAP = self.nivjob_geomlib.alm2map(blm_L-blt_MAP, *self.lm_max_blt, nthreads=4)
                    log.info("starting MAP delensing for iteration {}".format(it))
                    blt_L_MAP = self.lib[maskflavour][maskid].map2cl(bdel_MAP)    
                    outputdata[0][2+iti][maskcounter] = blt_L_MAP

                maskcounter+=1

        np.save(self.fns(simidx), outputdata)
            

    #@log_on_start(logging.INFO, "get_residualblens() started")
    #@log_on_end(logging.INFO, "get_residualblens() finished")
    def get_residualblens(self, simidx, it):
        basemap = self.get_basemap(simidx)
        
        return basemap - self.get_blt_it(simidx, it)
    

    # @base_exception_handler
    #@log_on_start(logging.INFO, "read_data() started")
    #@log_on_end(logging.INFO, "read_data() finished")
    def read_data(self):
        bcl_L = np.zeros(shape=(len(self.its)+2, len(self.nlevels)+len(self.masks_fromfn), len(self.simidxs), len(self.edges)-1))
        for simidxi, simidx in enumerate(self.simidxs):
            data = np.load(self.fns(simidx))
            bcl_L[0,:,simidxi] = data[0][0]
            bcl_L[1,:,simidxi] = data[0][1]
            for iti, it in enumerate(self.its):
                bcl_L[2+iti,:,simidxi] = data[0][2+iti]

        return bcl_L


    def hlm2dlm(self, hlm, inplace):
        if self.h == 'd':
            return hlm if inplace else hlm.copy()
        if self.h == 'p':
            h2d = np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float))
        elif self.h == 'k':
            h2d = cli(0.5 * np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float)))
        else:
            assert 0, self.h + ' not implemented'
        if inplace:
            almxfl(hlm, h2d, self.mmax_qlm, True)
        else:
            return  almxfl(hlm, h2d, self.mmax_qlm, False)


class overwrite_anafast():
    """Convenience class for overwriting method name
    """    

    def map2cl(self, *args, **kwargs):
        return hp.anafast(*args, **kwargs)


class masked_lib:
    """Convenience class for handling method names
    """   
    def __init__(self, mask, cl_calc, lmax, lmax_mask):
        self.mask = mask
        self.cl_calc = cl_calc
        self.lmax = lmax
        self.lmax_mask = lmax_mask

    def map2cl(self, map):
        return self.cl_calc.map2cl(map, self.mask, self.lmax, self.lmax_mask)
