#!/usr/bin/env python

"""handler.py: This module collects the delensalot jobs. It receives the delensalot model build for the respective job. They all initialize needed modules and directories, collect the computing-jobs, and run the computing-jobs, with MPI support, if available.
    
"""
import os
from os.path import join as opj
import hashlib
import datetime, getpass, copy

import numpy as np
import healpy as hp

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end


from plancklens import qresp, qest, utils as pl_utils
from plancklens.sims import planck2018_sims

from lenspyx.lensing import get_geom 

from delensalot.utils import read_map, ztruncify
from delensalot.utility import utils_qe, utils_sims
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam, alm2cl

from delensalot.config.visitor import transform, transform3d
from delensalot.config.metamodel import DEFAULT_NotAValue

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI
from delensalot.core.ivf import filt_util, filt_cinv, filt_simple

from delensalot.core.iterator.iteration_handler import iterator_transformer
from delensalot.core.iterator.statics import rec as rec
from delensalot.core.decorator.exception_handler import base as base_exception_handler
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD
from delensalot.core.opfilt.opfilt_handler import QE_transformer, MAP_transformer
from delensalot.core.opfilt.bmodes_ninv import template_dense, template_bfilt

def get_dirname(s):
    return s.replace('(', '').replace(')', '').replace('{', '').replace('}', '').replace(' ', '').replace('\'', '').replace('\"', '').replace(':', '_').replace(',', '_').replace('[', '').replace(']', '')

def dict2roundeddict(d):
    s = ''
    for k,v in d.items():
        d[k] = np.around(v,3)
    return d

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
        self.libdir_blt = lambda simidx: opj(self.TEMP, 'MAP/%s'%(self.k), 'sim%04d'%(simidx) + self.version, 'BLT/')
        for simidx in np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int):
            ## calculates all plms even for mf indices. This is not necessarily requested due to potentially simidxs =/= simidxs_mf, but otherwise collect and run must be adapted and its ok like this.
            libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            if not os.path.exists(libdir_MAPidx):
                os.makedirs(libdir_MAPidx)
            if not os.path.exists(self.libdir_blt(simidx)):
                os.makedirs(self.libdir_blt(simidx))
         
        self.config_model = model
        self.jobs = []


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):

        assert 0, "Overwrite"


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def run(self):

        assert 0, "Implement if needed"


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_qlm_it(self, simidx, it):

        assert 0, "Implement if needed"


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), its)

        return plms


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_mf_it(self, simidx, it, normalized=True):

        assert 0, "Implement if needed"


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_blt_it(self, simidx, it):
        if self.data_from_CFS:
            # TODO probably enough to just check if libdir_blt_MAP_CFS is empty
            assert 0, 'implement if needed'
            fn_blt = self.libdir_blt_MAP_CFS(self.k, simidx, self.version)
        else:
            if it == 0:
                fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
            elif it >0:
                fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it, self.lm_max_blt[0]) + '.npy')
        return np.load(fn_blt)
    

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_ivf(self, simidx, it, field):

        assert 0, "Implement if needed"


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_wf(self, simidx, it, field):

        assert 0, "Implement if needed"
    

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def get_fiducial_sim(self, simidx, field):
        """_summary_
        """   
        assert 0, "Implement if needed"


    @log_on_start(logging.DEBUG, "get_filter() started")
    @log_on_end(logging.DEBUG, "get_filter() finished")
    def get_filter(self): 
        assert 0, 'overwrite'


class OBD_builder(Basejob):
    """OBD matrix builder Job. Calculates the OBD matrix, used to correctly deproject the B-modes at a masked sky.
    """
    @check_MPI
    def __init__(self, OBD_model):
        self.__dict__.update(OBD_model.__dict__)
        nivp = self._load_niv(self.nivp_desc)
        self.nivp = ztruncify(nivp, self.zbounds)


    def _load_niv(self, niv_desc):
        n_inv = []
        for i, tn in enumerate(niv_desc):
            if isinstance(tn, list):
                n_inv_prod = read_map(tn[0])
                if len(tn) > 1:
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * read_map(n)
                n_inv.append(n_inv_prod)
            else:
                n_inv.append(read_map(self._n_inv[i]))
        assert len(n_inv) in [1, 3], len(n_inv)
        return np.array(n_inv)

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def collect_jobs(self):
        jobs = []
        if not os.path.isfile(opj(self.libdir,'tniti.npy')):
            # This fakes the collect/run structure, as bpl takes care of MPI 
            jobs = [0]  
        self.jobs = jobs
        return jobs


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "run() started")
    @log_on_end(logging.DEBUG, "run() finished")
    def run(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        for job in self.jobs:
            bpl = template_bfilt(self.lmin_b, self.nivjob_geomlib, self.tr, _lib_dir=self.libdir)
            if not os.path.exists(self.libdir+ '/tnit.npy'):
                bpl._get_rows_mpi(self.nivp, prefix='')
            mpi.barrier()
            if mpi.rank == 0:
                if not os.path.exists(self.libdir+ '/tnit.npy'):
                    tnit = bpl._build_tnit()
                    np.save(self.libdir+ '/tnit.npy', tnit)
                else:
                    tnit = np.load(self.libdir+ '/tnit.npy')
                if not os.path.exists(self.libdir+ '/tniti.npy'):
                    log.info(tnit.shape)
                    log.debug('inverting')
                    tniti = np.linalg.inv(tnit + np.diag((1. / (self.nlev_dep / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
                    np.save(self.libdir+ '/tniti.npy', tniti)
                    readme = '{}: tniti.npy. created from user {} using lerepi/delensalot with the following settings: {}'.format(getpass.getuser(), datetime.date.today(), self.__dict__)
                    with open(self.libdir+ '/README.txt', 'w') as f:
                        f.write(readme)
                else:
                    log.debug('Matrix already created')
        mpi.barrier()


class Sim_generator(Basejob):
    """Simulation generation Job. Generates simulations for the requested configuration.
        * If any libdir exists, then a flavour of data is provided. Therefore, can only check by making sure flavour == obs, and fns exist.
    """
    def __init__(self, dlensalot_model):
        """ In this init we make the following decision:
         * (1) Does user provide obs data? Then Sim_generator can be fully skipped
         * (2) Otherwise, check if files already generated (of course, delensalot model may not know this),
           * If so, update the simhandler
           * If not,
             * generate the simulations
             * update the simhandler
        """        
        super().__init__(dlensalot_model)
        if self.simulationdata.flavour == 'obs' or np.all(self.simulationdata.obs_lib.maps != DEFAULT_NotAValue): # (1)
            # Here, obs data is provided and nothing needs to be generated
            if np.all(self.simulationdata.obs_lib.maps != DEFAULT_NotAValue):
                log.info('Will use data provided in memory')
            else:
                log.info('Will use obs data stored at {} with filenames {}'.format(self.simulationdata.libdir, str(self.simulationdata.fns)))
        else:
            if self.simulationdata.flavour == 'sky':
                # Here, sky data is provided and obs needs to be generated
                self.libdir_sky = self.simulationdata.libdir
                self.fns_sky = self.simulationdata.fns
                lenjob_geomstr = 'unknown_lensinggeometry'
            else:
                hlib = hashlib.sha256()
                hlib.update(str(self.simulationdata.transfunction).encode())
                transfunctioncode = hlib.hexdigest()[:4]
                # some flavour provided, and we need to generate the sky and obs maps from this.
                lenjob_geomstr = get_dirname(str(self.simulationdata.len_lib.lenjob_geominfo))
                self.libdir_suffix = 'generic' if self.libdir_suffix == '' else self.libdir_suffix
                self.libdir_sky = opj(os.environ['SCRATCH'], 'simulation/', self.libdir_suffix, get_dirname(str(self.simulationdata.geominfo)), lenjob_geomstr)
                self.fns_sky = self.set_basename_sky()
                self.fnsP = 'philm_{}.npy'
            self.libdir_suffix = 'generic' if self.libdir_suffix == '' else self.libdir_suffix
            nlev_round = dict2roundeddict(self.simulationdata.nlev)
            self.libdir = opj(os.environ['SCRATCH'], 'simulation/', self.libdir_suffix, get_dirname(str(self.simulationdata.geominfo)), get_dirname(lenjob_geomstr), get_dirname(str(sorted(nlev_round.items()))), '{}'.format(transfunctioncode)) # 
            self.fns = self.set_basename_obs()
            
            # in init, only rank 0 enters in first round to set dirs etc.. so cannot use bcast
            if mpi.rank == 0:
                if mpi.size>1:
                    if not os.path.exists(self.libdir):
                        os.makedirs(self.libdir)
                    [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
                else:
                    if not os.path.exists(self.libdir):
                        os.makedirs(self.libdir)
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)
            
            simidxs_ = np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int)
            check_ = True  
            for simidx in simidxs_: # (2)
                if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']: 
                    if not (os.path.exists(opj(self.libdir, self.fns['E'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['B'].format(simidx)))):
                        check_ = False
                        break
                elif self.k in ['ptt']:
                    if not os.path.exists(opj(self.libdir, self.fns['T'].format(simidx))):
                        check_ = False
                        break
                elif self.k in ['p']:
                    if not (os.path.exists(opj(self.libdir, self.fns['T'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['E'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['B'].format(simidx)))):
                        check_ = False
                        break
            if check_: # (3)
                self.postrun_obs()
                log.info('Will use obs data at {} with filenames {}'.format(self.libdir, str(self.fns)))
            else:
                log.info('obs data will be stored at {} with filenames {}'.format(self.libdir, str(self.fns)))

            if self.simulationdata.flavour != 'sky':
                # if data is below sky, I need to check. otherwise we know they exist
                check_ = True  
                for simidx in simidxs_: # (2)
                    if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']: 
                        if not (os.path.exists(opj(self.libdir_sky, self.fns_sky['E'].format(simidx))) and os.path.exists(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)))):
                            check_ = False
                            break
                    elif self.k in ['ptt']:
                        if not os.path.exists(opj(self.libdir_sky, self.fns_sky['T'].format(simidx))):
                            check_ = False
                            break
                    elif self.k in ['p']:
                        if not (os.path.exists(opj(self.libdir_sky, self.fns_sky['T'].format(simidx))) and os.path.exists(opj(self.libdir_sky, self.fns_sky['E'].format(simidx))) and os.path.exists(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)))):
                            check_ = False
                            break
                if check_: # (3)
                    self.postrun_sky()
                    log.info('Will use sky data at {} with filenames {}'.format(self.libdir_sky, str(self.fns_sky)))
                else:
                    log.info('sky data will be stored at {} with filenames {}'.format(self.libdir_sky, str(self.fns_sky)))


    def set_basename_sky(self):
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']: 
            fns_sky = {'E': 'Ealmsky_{}.npy', 'B': 'Balmsky_{}.npy'}
        elif self.k in ['ptt']:
            fns_sky = {'T': 'Talmsky_{}.npy'}
        elif self.k in ['p']:
            fns_sky = {'T': 'Talmsky_{}.npy', 'E': 'Ealmsky_{}.npy', 'B': 'Balmsky_{}.npy'}
        return fns_sky

    def set_basename_obs(self):
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']: 
            fns = {'E': 'Ealmobs_{}.npy', 'B': 'Balmobs_{}.npy'}
        elif self.k in ['ptt']:
            fns = {'T': 'Talmobs_{}.npy'}
        elif self.k in ['p']:
            fns = {'T': 'Talmobs_{}.npy', 'E': 'Ealmobs_{}.npy', 'B': 'Balmobs_{}.npy'}
        return fns

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "Sim.collect_jobs() started")
    @log_on_end(logging.DEBUG, "Sim.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        jobs = list(range(len(['generate_sky', 'generate_obs'])))
        if np.all(self.simulationdata.maps == DEFAULT_NotAValue) and self.simulationdata.flavour != 'obs':
            for taski, task in enumerate(['generate_sky', 'generate_obs']):
                _jobs = []
                simidxs_ = np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int)
                if task == 'generate_sky':
                    for simidx in simidxs_:
                        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                            fnQ = opj(self.libdir_sky, self.fns_sky['E'].format(simidx))
                            fnU = opj(self.libdir_sky, self.fns_sky['B'].format(simidx))
                            if not os.path.isfile(fnQ) or not os.path.isfile(fnU) or not os.path.exists(opj(self.libdir_sky, self.fnsP.format(simidx))):
                                _jobs.append(simidx)
                        elif self.k in ['ptt']:
                            fnT = opj(self.libdir_sky, self.fns_sky['T'].format(simidx))
                            if not os.path.isfile(fnT) or not os.path.exists(opj(self.libdir_sky, self.fnsP.format(simidx))):
                                _jobs.append(simidx)
                        elif self.k in ['p']:
                            fnT = opj(self.libdir_sky, self.fns_sky['T'].format(simidx))
                            fnQ = opj(self.libdir_sky, self.fns_sky['E'].format(simidx))
                            fnU = opj(self.libdir_sky, self.fns_sky['B'].format(simidx))
                            if not os.path.isfile(fnT) or not os.path.isfile(fnQ) or not os.path.isfile(fnU) or not os.path.exists(opj(self.libdir_sky, self.fnsP.format(simidx))):
                                _jobs.append(simidx)

                if task == 'generate_obs':
                    for simidx in simidxs_:
                        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                            fnQ = opj(self.libdir, self.fns['E'].format(simidx))
                            fnU = opj(self.libdir, self.fns['B'].format(simidx))
                            if not os.path.isfile(fnQ) or not os.path.isfile(fnU):
                                _jobs.append(simidx)
                        elif self.k in ['ptt']:
                            fnT = opj(self.libdir, self.fns['T'].format(simidx))
                            if not os.path.isfile(fnT):
                                _jobs.append(simidx)
                        elif self.k in ['p']:
                            fnT = opj(self.libdir, self.fns['T'].format(simidx))
                            fnQ = opj(self.libdir, self.fns['E'].format(simidx))
                            fnU = opj(self.libdir, self.fns['B'].format(simidx))
                            if not os.path.isfile(fnT) or not os.path.isfile(fnQ) or not os.path.isfile(fnU):
                                _jobs.append(simidx)          
                jobs[taski] = _jobs
            self.jobs = jobs
        else:
            self.jobs = [[],[]]
        return self.jobs


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "Sim.run() started")
    @log_on_end(logging.DEBUG, "Sim.run() finished")
    def run(self):
        for taski, task in enumerate(['generate_sky', 'generate_obs']):
            for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                log.info("rank {} (size {}) generating sim {}".format(mpi.rank, mpi.size, simidx))
                if task == 'generate_sky':
                    self.generate_sky(simidx)
                if task == 'generate_obs':
                    self.generate_obs(simidx)
                if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                    self.simulationdata.purgecache()
        if np.all(self.simulationdata.maps == DEFAULT_NotAValue):
            self.postrun_sky()
            self.postrun_obs()


    @log_on_start(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) finished")
    def generate_sky(self, simidx):
        if not os.path.exists(opj(self.libdir_sky, self.fnsP.format(simidx))):
            phi = self.simulationdata.get_sim_phi(simidx, space='alm')
            np.save(opj(self.libdir_sky, self.fnsP.format(simidx)), phi)
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            if not (os.path.exists(opj(self.libdir_sky, self.fns_sky['E'].format(simidx))) and os.path.exists(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)))):
                EBsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='polarization')
                np.save(opj(self.libdir_sky, self.fns_sky['E'].format(simidx)), EBsky[0])
                np.save(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)), EBsky[1])

        elif self.k in ['ptt']:
            if not os.path.exists(opj(self.libdir_sky, self.fns_sky['T'].format(simidx))):
                Tsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='temperature')
                np.save(opj(self.libdir_sky, self.fns_sky['T'].format(simidx)), Tsky)

        elif self.k in ['p']:
            if not (os.path.exists(opj(self.libdir_sky, self.fns_sky['E'].format(simidx))) and os.path.exists(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)))):
                EBsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='polarization')
                np.save(opj(self.libdir_sky, self.fns_sky['E'].format(simidx)), EBsky[0])
                np.save(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)), EBsky[1])
            if not os.path.exists(opj(self.libdir_sky, self.fns_sky['T'].format(simidx))):
                Tsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='temperature')
                np.save(opj(self.libdir_sky, self.fns_sky['T'].format(simidx)), Tsky)


    @log_on_start(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) finished")
    def generate_obs(self, simidx):
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            if not (os.path.exists(opj(self.libdir, self.fns['E'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['B'].format(simidx)))):
                EBobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='polarization')
                np.save(opj(self.libdir, self.fns['E'].format(simidx)), EBobs[0])
                np.save(opj(self.libdir, self.fns['B'].format(simidx)), EBobs[1])

        elif self.k in ['ptt']:
            if not os.path.exists(opj(self.libdir, self.fns['T'].format(simidx))):
                Tobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='temperature')
                np.save(opj(self.libdir, self.fns['T'].format(simidx)), Tobs)

        elif self.k in ['p']:
            if not (os.path.exists(opj(self.libdir, self.fns['E'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['B'].format(simidx)))):
                EBobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='polarization')
                np.save(opj(self.libdir, self.fns['E'].format(simidx)), EBobs[0])
                np.save(opj(self.libdir, self.fns['B'].format(simidx)), EBobs[1])
            if not os.path.exists(opj(self.libdir, self.fns['T'].format(simidx))):
                Tobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='temperature')
                np.save(opj(self.libdir, self.fns['T'].format(simidx)), Tobs)

    def postrun_obs(self):
        # we always enter postrun, even from other jobs (like QE_lensrec). So making sure we are not accidently overwriting libdirs and fns
        if self.simulationdata.flavour != 'sky' and self.simulationdata.flavour != 'obs' and np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
            self.simulationdata.libdir = self.libdir
            self.simulationdata.fns = self.fns
            if self.simulationdata.flavour != 'obs':
                self.simulationdata.obs_lib.fns = self.fns
                self.simulationdata.obs_lib.libdir = self.libdir
                self.simulationdata.obs_lib.space = 'alm'
                self.simulationdata.obs_lib.spin = 0

    def postrun_sky(self):
        # we always enter postrun, even from other jobs (like QE_lensrec). So making sure we are not accidently overwriting libdirs and fns
        if self.simulationdata.flavour != 'sky' and self.simulationdata.flavour != 'obs' and np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
            self.simulationdata.len_lib.fns = self.fns_sky
            self.simulationdata.len_lib.libdir = self.libdir_sky
            self.simulationdata.len_lib.space = 'alm'
            self.simulationdata.len_lib.spin = 0

            self.simulationdata.unl_lib.libdir_phi = self.libdir_sky
            self.simulationdata.unl_lib.fnsP = self.fnsP
            self.simulationdata.unl_lib.phi_field = 'potential'
            self.simulationdata.unl_lib.phi_space = 'alm' # we always safe phi as lm's


class Noise_modeller(Basejob):

    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)

        """
        niv
        delensalot either 
         * nivt/p_desc (QE, iso and aniso) - TODO how does it truncify inside filter?
         * uses nlev (MAP, iso),
         * uses niv (MAP, aniso), - truncified
        """

        ## QE ##
        if self.qe_filter_directional == 'isotropic':
            nlev = self.nlev
            # no masking or bmarg needed here
        else:
            nivt_desc = self.nivt_desc
            nivp_desc = self.nivp_desc
            ## mask info is in here,  TODO but no bmarg, truncification missing
        ## MAP ##
        if self.it_filter_directional == 'isotropic':
            nlev = self.nlev
        else:
            if self.k in ['ptt']:
                self.niv = ztruncify(read_map(self.nivt_desc), self.zbounds)
            else:
                assert self.k not in ['p'], 'implement if needed, niv needs t map'
                self.niv = ztruncify(read_map(self.nivp_desc), self.zbounds)

        # for QE, what we pass to plancklens is only the descriptor, no truncification, no bmarg. This is passed separately
        # for MAP, what we pass is truncified noise map, but bmarg still missing

        """
        cinv
        filters
         * cinv_t and cinv_p (QE, iso and aniso). These are the opfilt filters, and eventually access alm_filter_n*
         * alm_nlev_*
        """
        ## QE ##
        QE_filters = transform(self, QE_transformer())
        QE_filter = transform(self, QE_filters())

        self.cinv_t = filt_cinv.cinv_t(opj(self.libdir_QE, 'cinv_t'),
            self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
            self.ttebl['t'], self.nivt_desc,
            marge_monopole=True, marge_dipole=True, marge_maps=[])
        transf_elm_loc = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_ivf[0])
        if self.OBD == 'OBD':
            nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
            self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                transf_elm_loc[:self.lm_max_ivf[0]+1], self.nivp_desc, geom=nivjob_geomlib_, #self.nivjob_geomlib,
                chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                zbounds=(-1,1), _bmarg_lib_dir=self.obd_libdir, _bmarg_rescal=self.obd_rescale,
                sht_threads=self.tr)
        else:
            self.cinv_p = filt_cinv.cinv_p(opj(self.TEMP, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                self.ttebl['e'], self.nivp_desc, chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                transf_blm=self.ttebl['b'], marge_qmaps=(), marge_umaps=())


        ## MAP ##
        MAP_filters = transform(self, MAP_transformer())
        MAP_filter = transform(self, MAP_filters())

        # for QE 

        """
        inverse variance filtering
        names are,
         * self.ivfs (QE, aniso, iso) - takes transfer function,
         * ??? (MAP)
        """
        ## QE ##
        # 
        if self.qe_filter_directional == 'isotropic':
            self.ivfs = filt_simple.library_fullsky_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
        elif self.qe_filter_directional == 'anisotropic':
            _filter_raw = filt_cinv.library_cinv_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.cinv_t, self.cinv_p, self.cls_len)
            _ftebl_rs = lambda x: np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[x])
            self.ivfs = filt_util.library_ftl(_filter_raw, self.lm_max_qlm[0], _ftebl_rs(0), _ftebl_rs(1), _ftebl_rs(2))
        
        ### MAP ###
        # For MAP, this happens inside cs_iterator
        

        """
        lenspot estimator lib
        names are,
         * qlms_dd(QE, aniso, iso) - takes filter, as input
         * cs_iterator (MAP, aniso and iso) - takes filter and qe starting point as input
        """       
        ## QE ##
        if self.qlm_type == 'sepTP':
            self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])

        ## MAP ##
            # cs_iterator
        
    
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
        self.simulationdata = self.simgen.simulationdata

        if self.qe_filter_directional == 'isotropic':
            self.ivfs = filt_simple.library_fullsky_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
            if self.qlm_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        elif self.qe_filter_directional == 'anisotropic':
            ## Wait for finished run(), as plancklens triggers cinv_calc...
            if len(self.collect_jobs()[0]) == 0:
                self.init_aniso_filter()

        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.R_unl = lambda: qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.lm_max_qlm[0])[0]

        ## Faking here sims_MAP for calc_blt as iteration_handler needs it
        if 'calc_blt' in self.qe_tasks:
            if self.it_filter_directional == 'anisotropic':
                # TODO reimplement ztrunc
                self.sims_MAP = utils_sims.ztrunc_sims(self.simulationdata, self.nivjob_geominfo[1]['nside'], [self.zbounds])
            elif self.it_filter_directional == 'isotropic':
                self.sims_MAP = self.simulationdata

        # FIXME currently only used for testing filter integration. These QE filter are not used for QE reoconstruction, but will be in the near future when Plancklens dependency is dropped. 
        if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee', 'ptt']:
            self.filter = self.get_filter()


    def init_cinv(self):
        self.cinv_t = filt_cinv.cinv_t(opj(self.libdir_QE, 'cinv_t'),
            self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
            self.ttebl['t'], self.nivt_desc,
            marge_monopole=True, marge_dipole=True, marge_maps=[])

        # FIXME is this right? what if analysis includes pixelwindow function?
        transf_elm_loc = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_ivf[0])
        if self.OBD == 'OBD':
            nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
            self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                transf_elm_loc[:self.lm_max_ivf[0]+1], self.nivp_desc, geom=nivjob_geomlib_, #self.nivjob_geomlib,
                chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                zbounds=(-1,1), _bmarg_lib_dir=self.obd_libdir, _bmarg_rescal=self.obd_rescale,
                sht_threads=self.tr)
        else:
            self.cinv_p = filt_cinv.cinv_p(opj(self.TEMP, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                self.ttebl['e'], self.nivp_desc, chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol),
                transf_blm=self.ttebl['b'], marge_qmaps=(), marge_umaps=())


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.collect_jobs(recalc={recalc}) started")
    @log_on_end(logging.DEBUG, "QE.collect_jobs(recalc={recalc}) finished: jobs={self.jobs}")
    def collect_jobs(self, recalc=False):

        # qe_tasks overwrites task-list and is needed if MAP lensrec calls QE lensrec
        jobs = list(range(len(self.qe_tasks)))
        for taski, task in enumerate(self.qe_tasks):
            ## task_dependence
            ## calc_mf -> calc_phi, calc_blt -> calc_phi, (calc_mf)
            _jobs = []

            ## Calculate realization dependent phi, i.e. plm_it000.
            if task == 'calc_phi':
                ## this filename must match plancklens filename
                fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.simidxs_mf)))
                ## Skip if meanfield already calculated
                if not os.path.isfile(fn_mf) or recalc:
                    for simidx in np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int):
                        fn_qlm = opj(opj(self.libdir_QE, 'qlms_dd'), 'sim_%s_%04d.fits'%(self.k, simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                        if not os.path.isfile(fn_qlm) or recalc:
                            _jobs.append(simidx)

            if task == 'calc_meanfield':
                fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.simidxs_mf)))
                if not os.path.isfile(fn_mf) or recalc:
                    for simidx in self.simidxs_mf:
                        fn_qlm = opj(opj(self.libdir_QE, 'qlms_dd'), 'sim_%s_%04d.fits'%(self.k, simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                        if not os.path.isfile(fn_qlm) or recalc:
                            _jobs.append(int(simidx))

            ## Calculate B-lensing template
            if task == 'calc_blt':
                for simidx in self.simidxs:
                    ## this filename must match the one created in get_template_blm()
                    fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
                    if not os.path.isfile(fn_blt) or recalc:
                        _jobs.append(simidx)

            jobs[taski] = _jobs
        self.jobs = jobs

        return jobs


    def init_aniso_filter(self):
        self.init_cinv()
        # self.sims_MAP = utils_sims.ztrunc_sims(self.simulationdata, self.nivjob_geominfo[1]['nside'], [self.zbounds])
        _filter_raw = filt_cinv.library_cinv_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.cinv_t, self.cinv_p, self.cls_len)
        _ftebl_rs = lambda x: np.ones(self.lm_max_qlm[0] + 1, dtype=float) * (np.arange(self.lm_max_qlm[0] + 1) >= self.lmin_teb[x])
        self.ivfs = filt_util.library_ftl(_filter_raw, self.lm_max_qlm[0], _ftebl_rs(0), _ftebl_rs(1), _ftebl_rs(2))
        self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.run(task={task}) started")
    @log_on_end(logging.DEBUG, "QE.run(task={task}) finished")
    def run(self, task=None):
        ## TODO following comment can be removed now?
        ## task may be set from MAP lensrec, as MAP lensrec has prereqs to QE lensrec
        ## if None, then this is a normal QE lensrec call

        # blueprint for new task: calc_cinv
        # Only now instantiate aniso filter as it triggers an expensive computation
        if True: # 'calc_cinv'
            if self.qe_filter_directional == 'anisotropic':
                first_rank = mpi.bcast(mpi.rank)
                if first_rank == mpi.rank:
                    mpi.disable()
                    self.init_aniso_filter()
                    mpi.enable()
                    [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
                else:
                    mpi.receive(None, source=mpi.ANY_SOURCE)
                self.init_aniso_filter()
                        
        _tasks = self.qe_tasks if task is None else [task]
        for taski, task in enumerate(_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_phi':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.qlms_dd.get_sim_qlm(self.k, int(idx))
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()
                mpi.barrier()
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.get_plm(idx, self.QE_subtract_meanfield)
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()
                

            if task == 'calc_meanfield':
                if len(self.jobs[taski])>0:
                    log.debug('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                    mpi.barrier()
                    log.debug("Done waiting. Rank 0 going to calculate meanfield-file.. everyone else waiting.")
                    if mpi.rank == 0:
                        self.get_meanfield(int(idx))
                        log.debug("rank 0 finished calculating meanfield-file.. everyone else waiting.")
                    mpi.barrier()

            if task == 'calc_blt':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    # ## Faking here MAP filters
                    self.itlib_iterator = transform(self.MAP_job, iterator_transformer(self.MAP_job, simidx, self.dlensalot_model))
                    self.get_blt(simidx)
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_sim_qlm(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_sim_qlm(simidx={simidx}) finished")
    def get_sim_qlm(self, simidx):

        return self.qlms_dd.get_sim_qlm(self.k, int(simidx))


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_wflm(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_wflm(simidx={simidx}) finished")    
    def get_wflm(self, simidx):
        if self.k in ['ptt']:
            return lambda: alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.k in ['p']:
            return lambda: np.array([alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_R_unl() started")
    @log_on_end(logging.DEBUG, "QE.get_R_unl() finished")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_meanfield(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_meanfield(simidx={simidx}) finished")
    def get_meanfield(self, simidx):
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, 0))
        if self.Nmf > 1:
            if self.mfvar == None:
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
    @log_on_start(logging.DEBUG, "QE.get_plm_n1(simidx={simidx}, sub_mf={sub_mf}) started")
    @log_on_end(logging.DEBUG, "QE.get_plm_n1(simidx={simidx}, sub_mf={sub_mf}) finished")
    def get_plm_n1(self, simidx, sub_mf=True, N1=np.array([])):
        libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
        if N1.size == 0:
            N1 = 0
            fn_plm = opj(libdir_MAPidx, 'phi_plm_it000.npy') # Note: careful, this one doesn't have a simidx, so make sure it ends up in a simidx_directory (like MAP)
        else:
            fn_plm = opj(libdir_MAPidx, 'phi_plm_it000{}.npy'.format('_wN1'))
        if not os.path.exists(fn_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            if sub_mf and self.version != 'noMF':
                plm -= self.mf(int(simidx))  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R) + N1)
            plm = alm_copy(plm, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(plm, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(plm, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.lm_max_qlm[1], True)
            np.save(fn_plm, plm)

        return np.load(fn_plm)


    @log_on_start(logging.DEBUG, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) started")
    @log_on_end(logging.DEBUG, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) finished")
    def get_plm(self, simidx, sub_mf=True):
        libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
        fn_plm = opj(libdir_MAPidx, 'phi_plm_it000.npy') # Note: careful, this one doesn't have a simidx, so make sure it ends up in a simidx_directory (like MAP)
        if not os.path.exists(fn_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            if sub_mf and self.version != 'noMF':
                plm -= self.mf(int(simidx))  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R))
            plm = alm_copy(plm, None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            almxfl(plm, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
            almxfl(plm, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.lm_max_qlm[1], True)
            np.save(fn_plm, plm)

        return np.load(fn_plm)


    @log_on_start(logging.DEBUG, "QE.get_response_meanfield() started")
    @log_on_end(logging.DEBUG, "QE.get_response_meanfield() finished")
    def get_response_meanfield(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.lm_max_qlm[0])[0]
        else:
            log.info('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lm_max_qlm[0] + 1, dtype=float)

        return mf_resp

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_meanfield_normalized(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_meanfield_normalized(simidx={simidx}) finished")
    def get_meanfield_normalized(self, simidx):
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = qresp.get_response(self.k, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
        WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R))
        almxfl(mf_QE, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.cpp > 0, self.lm_max_qlm[1], True)

        return mf_QE


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_blt({simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_blt({simidx}) finished")
    def get_blt(self, simidx):
        fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            ## For QE, dlm_mod by construction doesn't do anything, because mean-field had already been subtracted from plm and we don't want to repeat that.
            dlm_mod = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, int(simidx)))
            blt = self.itlib_iterator.get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, dlm_mod=dlm_mod, perturbative=self.blt_pert, k=self.k)
            np.save(fn_blt, blt)
        return np.load(fn_blt)
    

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "QE.get_blt({simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_blt({simidx}) finished")
    def get_blt_new(self, simidx):

        def get_template_blm(it, it_e, lmaxb=1024, lmin_plm=1, perturbative=False):
            fn_blt = 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0])
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
        
        fn_blt = opj(self.libdir_QE, 'BLT/blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            blt = get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, perturbative=self.blt_pert)
            np.save(fn_blt, blt)

        return np.load(fn_blt)


    @log_on_start(logging.DEBUG, "get_filter() started")
    @log_on_end(logging.DEBUG, "get_filter() finished")
    def get_filter(self): 
        QE_filters = transform(self, QE_transformer())
        filter = transform(self, QE_filters())
        return filter
    

class MAP_lr(Basejob):
    """Iterative lensing reconstruction Job. Depends on class QE_lr, and class Sim_generator. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """

    @check_MPI
    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)
        # TODO Only needed to hand over to ith()
        self.dlensalot_model = dlensalot_model
        
        # FIXME remnant of previous version when jobs were dependent on each other. This can perhaps be simplified now.
        self.simgen = Sim_generator(dlensalot_model)
        self.simulationdata = self.simgen.simulationdata
        self.qe = QE_lr(dlensalot_model, caller=self)
        self.qe.simulationdata = self.simgen.simulationdata # just to be sure, so we have a single truth in MAP_lr. 


        if self.OBD == 'OBD':
            nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
            self.tpl = template_dense(self.lmin_teb[2], nivjob_geomlib_, self.tr, _lib_dir=self.obd_libdir, rescal=self.obd_rescale)
        else:
            self.tpl = None
        
        ## tasks -> mf_dirname
        if "calc_meanfield" in self.it_tasks or 'calc_blt' in self.it_tasks:
            if not os.path.isdir(self.mf_dirname) and mpi.rank == 0:
                os.makedirs(self.mf_dirname)

        # sims -> sims_MAP
        if self.it_filter_directional == 'anisotropic':
            self.sims_MAP = utils_sims.ztrunc_sims(self.simulationdata, self.nivjob_geominfo[1]['nside'], [self.zbounds])
            if self.k in ['ptt']:
                self.niv = self.sims_MAP.ztruncify(read_map(self.nivt_desc)) # inverse pixel noise map on consistent geometry
            else:
                assert self.k not in ['p'], 'implement if needed, niv needs t map'
                self.niv = np.array([self.sims_MAP.ztruncify(read_map(ni)) for ni in self.nivp_desc]) # inverse pixel noise map on consistent geometry
        elif self.it_filter_directional == 'isotropic':
            self.sims_MAP = self.simulationdata
        self.filter = self.get_filter()

    # # @base_exception_handler
    @log_on_start(logging.DEBUG, "MAP.map.collect_jobs() started")
    @log_on_end(logging.DEBUG, "MAP.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
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
                for simidx in self.simidxs_mf:
                    libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    if "calc_phi" in self.it_tasks:
                        _jobs.append(0)
                    elif rec.maxiterdone(libdir_MAPidx) < self.itmax:
                        _jobs.append(0)

            elif task == 'calc_blt':
                for simidx in self.simidxs:
                    fns_blt = np.array([opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it-1, self.lm_max_blt[0]) + '.npy') for it in np.arange(1,self.itmax+1)])
                    if not np.all([os.path.exists(fn_blt) for fn_blt in fns_blt]):
                        _jobs.append(simidx)

            jobs[taski] = _jobs
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "MAP.run() started")
    @log_on_end(logging.DEBUG, "MAP.run() finished")
    def run(self):
        for taski, task in enumerate(self.it_tasks):
            log.info('{}, task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
            if task == 'calc_phi':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    if self.itmax >= 0 and rec.maxiterdone(libdir_MAPidx) < self.itmax:
                        itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
                        for it in range(self.itmax + 1):
                            itlib_iterator.chain_descr = self.it_chain_descr(self.lm_max_unl[0], self.it_cg_tol(it))
                            itlib_iterator.soltn_cond = self.soltn_cond(it)
                            itlib_iterator.iterate(it, 'p')
                            log.info('{}, simidx {} done with it {}'.format(mpi.rank, simidx, it))
                    # If data is in memory only, don't purge simslib
                    if type(self.simulationdata.obs_lib.maps) == np.array:
                        pass
                    else:
                        if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                            self.simulationdata.purgecache()

            if task == 'calc_meanfield':
                # TODO I don't like barriers and not sure if they are still needed
                mpi.barrier()
                self.get_meanfields_it(np.arange(self.itmax+1), calc=True)
                mpi.barrier()

            if task == 'calc_blt':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
                    self.itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
                    for it in range(self.itmax + 1):
                        self.get_blt_it(simidx, it)
                    # If data is in memory only, don't purge simslib
                    if type(self.simulationdata.obs_lib.maps) == np.array:
                        pass
                    else:
                        if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                            self.simulationdata.purgecache()


    # # @base_exception_handler
    @log_on_start(logging.DEBUG, "MAP.get_plm_it(simidx={simidx}, its={its}) started")
    @log_on_end(logging.DEBUG, "MAP.get_plm_it(simidx={simidx}, its={its}) finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), its)

        return plms
    

    # # @base_exception_handler
    @log_on_start(logging.DEBUG, "MAP.get_meanfield_it(it={it}, calc={calc}) started")
    @log_on_end(logging.DEBUG, "MAP.get_meanfield_it(it={it}, calc={calc}) finished")
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
    @log_on_start(logging.DEBUG, "MAP.get_meanfields_it(its={its}, calc={calc}) started")
    @log_on_end(logging.DEBUG, "MAP.get_meanfields_it(its={its}, calc={calc}) finished")
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
    @log_on_start(logging.DEBUG, "MAP.get_blt_it(simidx={simidx}, it={it}) started")
    @log_on_end(logging.DEBUG, "MAP.get_blt_it(simidx={simidx}, it={it}) finished")
    def get_blt_it(self, simidx, it):
        if it == 0:
            self.qe.itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
            return self.qe.get_blt(simidx)
        fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, it, it, self.lm_max_blt[0]) + '.npy')
        if not os.path.exists(fn_blt):     
            self.libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            dlm_mod = np.zeros_like(rec.load_plms(self.libdir_MAPidx, [0])[0])
            if self.dlm_mod_bool and it>0 and it<=rec.maxiterdone(self.libdir_MAPidx):
                dlm_mod = self.get_meanfields_it([it], calc=False)
                if simidx in self.simidxs_mf:
                    dlm_mod = (dlm_mod - np.array(rec.load_plms(self.libdir_MAPidx, [it]))/self.Nmf) * self.Nmf/(self.Nmf - 1)
            if it<=rec.maxiterdone(self.libdir_MAPidx):
                blt = self.itlib_iterator.get_template_blm(it, it-1, lmaxb=self.lm_max_blt[0], lmin_plm=np.max([self.Lmin,5]), dlm_mod=dlm_mod, perturbative=False, k=self.k)
                np.save(fn_blt, blt)
        return np.load(fn_blt)


    @log_on_start(logging.DEBUG, "get_filter() started")
    @log_on_end(logging.DEBUG, "get_filter() finished")
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
        if not(os.path.isdir(self.libdir_delenser)):
            os.makedirs(self.libdir_delenser)
        self.fns = opj(self.libdir_delenser, 'ClBB_sim{:04d}.npy')


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "collect_jobs() started")
    @log_on_end(logging.DEBUG, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO a valid job is any requested job?, as BLTs may also be on CFS
        jobs = []
        for idx in self.simidxs:
            jobs.append(idx)
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    @log_on_start(logging.DEBUG, "run() started")
    @log_on_end(logging.DEBUG, "run() finished")
    def run(self):
        outputdata = self._prepare_job()
        if self.jobs != []:
            for simidx in self.jobs[mpi.rank::mpi.size]:
                log.debug('will store file at: {}'.format(self.fns.format(simidx)))
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
    

    # @log_on_start(logging.DEBUG, "get_basemap() started")
    # @log_on_end(logging.DEBUG, "get_basemap() finished")  
    def get_basemap(self, simidx):
        # TODO depends if data comes from delensalot simulations or from external.. needs cleaner implementation
        if self.basemap == 'lens':  
            return almxfl(alm_copy(self.simulationdata.get_sim_sky(simidx, space='alm', spin=0, field='polarization')[1], self.simulationdata.lmax, *self.lm_max_blt), self.ttebl['e'], self.lm_max_blt[0], inplace=False) 
        elif self.basemap == 'lens_ffp10':
            return almxfl(alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(simidx), None, lmaxout=self.lm_max_blt[0], mmaxout=self.lm_max_blt[1]), gauss_beam(2.3 / 180 / 60 * np.pi, lmax=self.lm_max_blt[1]))  
        else:
            # only checking for map to save some memory..
            if np.all(self.simulationdata.maps == DEFAULT_NotAValue):
                return alm_copy(self.simulationdata.get_sim_obs(simidx, space='alm', spin=0, field='polarization')[1], self.simulationdata.lmax, *self.lm_max_blt)
            else:
                return hp.map2alm_spin(self.simulationdata.get_sim_obs(simidx, space='map', spin=2, field='polarization'), spin=2, lmax=self.lm_max_blt[0], mmax=self.lm_max_blt[1])[1]

    

    @log_on_start(logging.DEBUG, "_delens() started")
    @log_on_end(logging.DEBUG, "_delens() finished")
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

        np.save(self.fns.format(simidx), outputdata)
            

    @log_on_start(logging.DEBUG, "get_residualblens() started")
    @log_on_end(logging.DEBUG, "get_residualblens() finished")
    def get_residualblens(self, simidx, it):
        basemap = self.get_basemap(simidx)
        
        return basemap - self.get_blt_it(simidx, it)
    

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "read_data() started")
    @log_on_end(logging.DEBUG, "read_data() finished")
    def read_data(self):
        bcl_L = np.zeros(shape=(len(self.its)+2, len(self.nlevels)+len(self.masks_fromfn), len(self.simidxs), len(self.edges)-1))
        for simidxi, simidx in enumerate(self.simidxs):
            data = np.load(self.fns.format(simidx))
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


class Phi_analyser(Basejob):
    """ This only works on Full sky.
    Phi analyser Job for calculating,
        * cross correlation,
        * cross correlation coefficient,
        * reconstruction bias,
        * empiric Wiener-filter.
    Data is stored in CLpp/
    """

    def __init__(self, dlensalot_model):
        super().__init__(dlensalot_model)
        self.its = np.arange(self.itmax)
        self.libdir_phianalayser = opj(self.TEMP, 'CL/{}'.format(self.k))
        if self.custom_WF_TEMP == self.libdir_phianalayser:
            # custom WF in fact is the standard WF
            self.custom_WF_TEMP = [None for n in np.arange(len(self.its))]
        else:
            self.WFemps = np.load(opj(self.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) if self.custom_WF_TEMP else [None for n in np.arange(len(self.its))]
        self.tasks = ['calc_WFemp', 'calc_crosscorr', 'calc_reconbias', 'calc_crosscorrcoeff']
        

        
        if not(os.path.isdir(self.libdir_phianalayser)):
            os.makedirs(self.libdir_phianalayser)
        
        self.TEMP_WF = opj(self.libdir_phianalayser, 'WF')
        if not os.path.isdir(self.TEMP_WF):
            os.makedirs(self.TEMP_WF)
        self.TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        if not os.path.isdir(self.TEMP_Cx):
            os.makedirs(self.TEMP_Cx)
        self.TEMP_Cxbias = opj(self.libdir_phianalayser, 'Cxb')
        if not os.path.isdir(self.TEMP_Cxbias):
            os.makedirs(self.TEMP_Cxbias)
        self.TEMP_Cccc = opj(self.libdir_phianalayser, 'Cccc')
        if not os.path.isdir(self.TEMP_Cccc):
            os.makedirs(self.TEMP_Cccc)

    def collect_jobs(self):
        _jobs, jobs = [], []
        for taski, task in enumerate(self.tasks):

            if task == 'calc_WFemp':
                fns = opj(self.TEMP_WF,'WFemp_%s_sim%s_it%s.npy')
                for simidx in self.simidxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, simidx, it)):
                            _jobs.append(simidx)
                            break
       
            if task == 'calc_crosscorr':
                fns = opj(self.TEMP_Cx,'CLx_%s_sim%s_it%s.npy')
                for simidx in self.simidxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, simidx, it)):
                            _jobs.append(simidx)
                            break
            
            if task == 'calc_reconbias':
                fns = opj(self.TEMP_Cxbias,'CLxb_%s_sim%s_it%s.npy')
                for simidx in self.simidxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, simidx, it)):
                            _jobs.append(simidx)
                            break

            if task == 'calc_crosscorrcoeff':
                fns = opj(self.TEMP_Cccc,'CLccc_%s_sim%s_it%s.npy')
                for simidx in self.simidxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, simidx, it)):
                            _jobs.append(simidx)
                            break

            # if task == 'calc_PB':
            #     """If self.other_analysis_TEMP, then calculate specific differences between the two analyses. This is,
            #     ## PB and LSS
            #         * $PB = \left(C_L^{\hat{\phi}\phi}(nG)-C_L^{\hat{\phi}\phi}(nGf)\right)$
            #     """
            #     TEMP_PB = opj(self.libdir_phianalayser, 'PB')
            #     if not os.path.isdir(TEMP_PB):
            #         os.makedirs(TEMP_PB)
            #     fns = opj(TEMP_PB,'PB_%s_sim%s_it%s.npy')
            #     for simidx in self.simidxs:
            #         for it in self.its:
            #             if not os.path.isfile(fns%(self.k, simidx, it)):
            #                 _jobs.append(simidx)
            #                 break

            # if task == 'calc_LSS':
            #     """If self.other_analysis_TEMP, then calculate specific differences between the two analyses. This is,
            #         * $LSS = \left(C_L^{\hat{\phi}\phi}(nGf)-C_L^{\phi\phi}(G)\right)$
            #         * $LSS_n = \left(C_L^{\hat{\phi}\phi}(nGf)-C_L^{\phi\phi}(nG)\right)$
            #     """
            #     TEMP_LSS = opj(self.libdir_phianalayser, 'LSS')
            #     if not os.path.isdir(TEMP_LSS):
            #         os.makedirs(TEMP_LSS)
            #     fns = opj(TEMP_LSS,'LSS_%s_sim%s_it%s.npy')
            #     for simidx in self.simidxs:
            #         for it in self.its:
            #             if not os.path.isfile(fns%(self.k, simidx, it)):
            #                 _jobs.append(simidx)
            #                 break

            # if task == 'calc_TOT':
            #     """If self.other_analysis_TEMP, then calculate specific differences between the two analyses. This is,
            #         * $TOT = \left(C_L^{\hat{\phi}\phi}(nG)-C_L^{\phi\phi}(G)\right)$
            #         * $TOT_n = \left(C_L^{\hat{\phi}\phi}(nG)-C_L^{\phi\phi}(nG)\right)$
            #     """
            #     TEMP_TOT = opj(self.libdir_phianalayser, 'TOT')
            #     if not os.path.isdir(TEMP_TOT):
            #         os.makedirs(TEMP_TOT)
            #     fns = opj(TEMP_TOT,'TOT_%s_sim%s_it%s.npy')
            #     for simidx in self.simidxs:
            #         for it in self.its:
            #             if not os.path.isfile(fns%(self.k, simidx, it)):
            #                 _jobs.append(simidx)
            #                 break

            jobs.append(_jobs)
        self.jobs = jobs


    def run(self):
        # Wait for everyone to finish previous job
        mpi.barrier()
        for taski, task in enumerate(self.tasks):
            if task == 'calc_WFemp':
                # First, calc for each simindex individually
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    val = self._get_wienerfilter_empiric(simidx, self.its)
                # Second, calc average WF, only let only one rank do this
                first_rank = mpi.bcast(mpi.rank)
                if first_rank == mpi.rank:
                    self.get_wienerfilter_empiric()
                    [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
                else:
                    mpi.receive(None, source=mpi.ANY_SOURCE)

            if task == 'calc_crosscorr':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_crosscorrelation(simidx, it, WFemps=self.WFemps)
                        self.get_autocorrelation(simidx, it, WFemps=self.WFemps)
           
            if task == 'calc_reconbias':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_reconstructionbias(simidx, it, WFemps=self.WFemps)

            if task == 'calc_crosscorrcoeff':  
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_crosscorrelationcoefficient(simidx, it, WFemps=self.WFemps)
            
            # if task == 'calc_PB':
            #     TEMP_PB = opj(self.libdir_phianalayser, 'PB')
            #     WFemps = np.load(opj(self.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) if self.custom_WF_TEMP else [None for n in np.arange(len(self.its))]
            #     ## cross, est_1 = kappa, est_2 = kappa_first
                
            #     (cross(est_1, in_1) - cross(est_2, in_2))/WFx_emp_norm[it]

            #     ## auto, est_1 = kappa, est_2 = kappa_first
            #     (auto(est_1) - auto(est_2))/WFx_emp_norm[it]**2

            # if task == 'calc_LSS':
            #     TEMP_LSS = opj(self.libdir_phianalayser, 'LSS')
            #     WFemps = np.load(opj(self.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) if self.custom_WF_TEMP else [None for n in np.arange(len(self.its))]
            #     ## cross, est_1 = kappa_first, est_2 = kappa
            #     (cross(est_1, in_1)/WFx_emp_norm[it] - auto(in_2))

            #     ## auto, est_1 = kappa_first, est_2 = kappa
            #     (auto(est_1) - auto(est_2))/WFx_emp_norm[it]**2

            # if task == 'calc_TOT':
            #     TEMP_TOT = opj(self.libdir_phianalayser, 'TOT')
            #     WFemps = np.load(opj(self.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) if self.custom_WF_TEMP else [None for n in np.arange(len(self.its))]
            #     ## cross, est_1 = kappa, est_2 = kappa
            #     (cross(est_1, in_1)/WFx_emp_norm[it] - auto(in_2))

            #     ## auto, est_1 = kappa_first, est_2 = gauss
            #     (auto(est_1) - auto(est_2))/WFx_emp_norm[it]**2


    def get_crosscorrelation(self, simidx, it, WFemps=None):
        TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        fns = opj(TEMP_Cx,'CLx_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cx,'CLx_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, simidx, it)):
            plm_est = self.get_plm_it(simidx, [it])[0]
            plm_in = alm_copy(self.simulationdata.get_sim_phi(simidx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None)/WFemps[it]
            np.save(fns%(self.k, simidx, it), val)
        return np.load(fns%(self.k, simidx, it))


    def get_autocorrelation(self, simidx, it, WFemps=None):
        # Note: this calculates auto of the estimate
        TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        fns = opj(TEMP_Cx,'CLa_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cx,'CLa_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, simidx, it)):
            plm_est = self.get_plm_it(simidx, [it])[0]
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) 
            val = alm2cl(plm_est, plm_est, None, None, None)/WFemps[it]**2
            np.save(fns%(self.k, simidx, it), val)
        return np.load(fns%(self.k, simidx, it))


    def get_reconstructionbias(self, simidx, it, WFemps=None):
        TEMP_Cxbias = opj(self.libdir_phianalayser, 'Cxb')
        fns = opj(TEMP_Cxbias,'CLxb_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cxbias,'CLxb_%s_sim%s_it%s.npy') 
        if not os.path.isfile(fns%(self.k, simidx, it)):
            plm_est = self.get_plm_it(simidx, [it])[0]
            plm_in = alm_copy(self.simulationdata.get_sim_phi(simidx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None) / alm2cl(plm_in, plm_in, None, None, None)/WFemps[it]
            np.save(fns%(self.k, simidx, it), val)
        return np.load(fns%(self.k, simidx, it))


    def get_crosscorrelationcoefficient(self, simidx, it, WFemps=None):
        TEMP_Cccc = opj(self.libdir_phianalayser, 'Cccc')
        fns = opj(TEMP_Cccc,'CLccc_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cccc,'CLccc_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, simidx, it)):
            # plm_QE = almxfl(self.qe.get_sim_qlm(simidx), utils.cli(R))
            plm_est = self.get_plm_it(simidx, [it])[0]
            plm_in = alm_copy(self.simulationdata.get_sim_phi(simidx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.simidxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None)**2/(alm2cl(plm_est, plm_est, None, None, None)*alm2cl(plm_in, plm_in, None, None, None))
            np.save(fns%(self.k, simidx, it), val)
        return np.load(fns%(self.k, simidx, it))



    def get_wienerfilter_analytic(self, simidx, it):
        assert 0, 'implement if needed'
        return None   
    

    def _get_wienerfilter_empiric(self, simidx, its):
        ## per sim calculation, no need to expose this. Only return averaged result across all sims, which is the function without the pre underline: get_wienerfilter_empiric()
        fns = opj(self.TEMP_WF, 'WFemp_%s_sim%s_it%s.npy')
        WFemps = np.zeros(shape=(len(its), self.lm_max_qlm[0]+1))
        if not np.array([os.path.isfile(fns%(self.k, simidx, it)) for it in its]).all():   
            plm_in = alm_copy(self.simulationdata.get_sim_phi(simidx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            plm_est = self.get_plm_it(simidx, its)
            for it in its:       
                WFemps[it] = alm2cl(plm_in, plm_est[it], None, None, None)/alm2cl(plm_in, plm_in, None, None, None)
                np.save(fns%(self.k, simidx, it), WFemps[it])
        for it in its:
            WFemps[it] = np.load(fns%(self.k, simidx, it))
        return WFemps
    
    def get_wienerfilter_empiric(self):
        fn = opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')
        if not os.path.isfile(fn%(self.k, len(self.simidxs), len(self.its))):   
            WFemps = np.array([self._get_wienerfilter_empiric(simidx, self.its) for simidx in self.simidxs])
            np.save(fn%(self.k, len(self.simidxs), len(self.its)), np.mean(WFemps, axis=0))
        return np.load(fn%(self.k, len(self.simidxs), len(self.its)))
        




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
