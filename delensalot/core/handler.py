#!/usr/bin/env python

"""handler.py: This module collects the delensalot jobs. It receives the delensalot model build for the respective job. They all initialize needed modules and directories, collect the computing-jobs, and run the computing-jobs, with MPI support, if available.
    
"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from typing import List, Type, Union
import os
from os.path import join as opj
import hashlib
import datetime, getpass, copy

import numpy as np
import healpy as hp

from plancklens.sims import planck2018_sims

from delensalot.core.MAP import functionforwardlist
from delensalot.utils import cli
from delensalot.utils import read_map, ztruncify
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam, alm2cl, alm_copy_nd

from delensalot.config.visitor import transform, transform3d
from delensalot.config.metamodel import DEFAULT_NotAValue

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI

from delensalot.core.iterator.statics import rec as rec
from delensalot.core.opfilt.bmodes_ninv import template_dense, template_bfilt

from delensalot.core.MAP import handler as MAP_handler
from delensalot.core.QE import handler as QE_handler

from delensalot.core.MAP.context import get_computation_context
from delensalot.sims.data_source import dirname_generator

from collections import UserDict

class ConstantDict(UserDict):
    def __init__(self, value):
        super().__init__()
        self._value = value

    def __getitem__(self, key):
        return self._value

    def get(self, key, default=None):
        return self._value


required_files_map = {
    'p_p': ['E', 'B'], 'p_eb': ['E', 'B'], 'peb': ['E', 'B'], 'p_be': ['E', 'B'], 'pee': ['E', 'B'],
    'ptt': ['T'],
    'p': ['T', 'E', 'B']}
# NOTE This is to generate all maps no matter the CMB estimator request
required_files_map = ConstantDict(['T', 'E', 'B'])

def dict2roundeddict(d):
    return {k: np.around(v, 3) for k, v in d.items()}

def get_hashcode(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:4]

def get_dirname(s):
    return str(s).translate(str.maketrans({"(": "", ")": "", "{": "", "}": "", "[": "", "]": "", 
                                            " ": "", "'": "", '"': "", ":": "_", ",": "_"}))

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}


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
        self.libdir_MAP = lambda qe_key, idx, version: opj(self.TEMP, 'MAP/%s'%(qe_key), 'sim%04d'%(idx) + version)
        self.libdir_blt = lambda idx: opj(self.TEMP, 'MAP/%s'%(self.k), 'sim%04d'%(idx) + self.version, 'BLT/')
        for idx in np.array(list(set(np.concatenate([self.idxs, self.idxs_mf]))), dtype=int):
            ## calculates all plms even for mf indices. This is not necessarily requested due to potentially idxs =/= idxs_mf, but otherwise collect and run must be adapted and its ok like this.
            libdir_MAPidx = self.libdir_MAP(self.k, idx, self.version)
            if not os.path.exists(libdir_MAPidx):
                os.makedirs(libdir_MAPidx)
            if not os.path.exists(self.libdir_blt(idx)):
                os.makedirs(self.libdir_blt(idx))
         
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


class OBDBuilder(Basejob):
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


    def collect_jobs(self):
        jobs = []
        if not os.path.isfile(opj(self.libdir,'tniti.npy')):
            # This fakes the collect/run structure, as bpl takes care of MPI 
            jobs = [0]  
        self.jobs = jobs
        return jobs


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


class DataContainer:
    """Simulation generation Job. Generates simulations for the requested configuration.
        * If any libdir exists, then a flavour of data is provided. Therefore, can only check by making sure flavour == obs, and fns exist.
    """
    def __init__(self, data_source, k, idxs, idxs_mf, TEMP, mask, sky_coverage, data_key, lm_max_sky):
        """ In this init we make the following checks:
         * (1) Does user provide obs data? Then DataContainer can be fully skipped
         * (2) Otherwise, check if files already generated (delensalot model may not know this, so need to search),
           * If so, update the simhandler with the respective libdirs and fns
           * If not,
             * generate the simulations
             * update the simhandler
        """
        self.data_source = data_source
        self.k = k
        self.idxs = idxs
        self.idxs_mf = idxs_mf
        self.TEMP = TEMP
        self.mask = mask
        self.sky_coverage = sky_coverage
        self.data_key = data_key
        self.lm_max_sky = lm_max_sky
        if self.data_source.flavour == 'obs' or np.all(self.data_source.obs_lib.maps != DEFAULT_NotAValue): # (1)
            # Here, obs data is provided and nothing needs to be generated
            if np.all(self.data_source.obs_lib.maps != DEFAULT_NotAValue):
                log.info('Will use data provided in memory')
            else:
                log.info('Will use obs data stored at {} with filenames {}'.format(self.data_source.libdir, str(self.data_source.fns)))
        else:
            if self.data_source.flavour == 'sky':
                # Here, sky data is provided and obs needs to be generated
                self.libdir_sky = self.data_source.libdir
                self.fns_sky = self.data_source.fns
                geomstr = 'unknown_skygeometry'
            else:
                # some flavour provided, and we need to generate the sky and obs maps from this.
                hashc = get_hashcode([val['component'] for val in self.data_source.sec_info.values()])
                geominfo = self.data_source.sky_lib.operator_info['lensing']['geominfo'] if 'lensing' in self.data_source.sky_lib.operator_info else self.data_source.sky_lib.operator_info['birefringence']['geominfo']
                geomstr = get_dirname(geominfo)+"_"+hashc
                
                self.libdir_sky = opj(dirname_generator(self.data_source.libdir_suffix, self.data_source.geominfo), geomstr)
                self.fns_sky = self.set_basename_sky()
                # NOTE for each operator, I need sec fns
                self.fns_sec = {}
                for sec, operator_info in self.data_source.operator_info.items():
                    self.fns_sec.update({sec:{}})
                    for comp in operator_info['component']:
                        self.fns_sec[sec][comp] = f'{sec}_{comp}lm_{{}}.npy'

            hashc = get_hashcode(str([val['component'] for val in self.data_source.sec_info.values()])+str([val['component'] for val in self.data_source.sec_info.values()]))
            nlev_round = dict2roundeddict(self.data_source.nlev)
            self.libdir = opj(dirname_generator(self.data_source.libdir_suffix, self.data_source.geominfo), geomstr, get_dirname(sorted(nlev_round.items())), f'{hashc}')
            self.fns = self.set_basename_obs()
            
            # in init, only rank 0 enters in first round to set dirs etc.. so cannot use bcast
            if mpi.rank == 0:
                if not os.path.exists(self.libdir):
                    os.makedirs(self.libdir)
                if mpi.size > 1:
                    for dest in range(mpi.size):
                        if dest != mpi.rank:
                            mpi.send(1, dest=dest)
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)

            required_files = required_files_map.get(self.k, [])
            idxs_ = np.unique(np.concatenate([self.idxs, self.idxs_mf])).astype(int)
            def check_and_log(libdir, fns, _postrun_method, data_type):
                """function to check file existence """
                if all(os.path.exists(opj(libdir, fns[f].format(idx))) for f in required_files for idx in idxs_):
                    _postrun_method()
                    log.info(f'will use {data_type} data at {libdir} with filenames {fns}')
                else:
                    log.info(f'{data_type} data will be stored at {libdir} with filenames {fns}')

            check_and_log(self.libdir, self.fns, self._postrun_obs, "obs")
            if self.data_source.flavour != 'sky':
                if all(os.path.exists(opj(self.libdir_sky, self.fns_sec[sec][component].format(idx))) for sec in self.fns_sec.keys() for component in self.fns_sec[sec] for idx in idxs_):
                    check_and_log(self.libdir_sky, self.fns_sky, self._postrun_sky, "sky")
                else:
                    log.info(f'sky data will be stored at {self.libdir_sky} with filenames {self.fns_sky}. All secondaries will be generated along the way')

        self.cls_lib = self.data_source.cls_lib
        self.obs_lib = self.data_source.obs_lib

    def set_basename_sky(self):
        return {'T': 'Talmsky_{}.npy', 'E': 'Ealmsky_{}.npy', 'B': 'Balmsky_{}.npy'}


    def set_basename_obs(self):
        return {'T': 'Talmobs_{}.npy', 'E': 'Ealmobs_{}.npy', 'B': 'Balmobs_{}.npy'}

    # @base_exception_handler
    @log_on_start(logging.INFO, "Sim.collect_jobs() started")
    @log_on_end(logging.INFO, "Sim.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        jobs = list(range(len(['generate_sky', 'generate_obs'])))
        required_files = required_files_map.get(self.k, [])
        if np.all(self.data_source.maps == DEFAULT_NotAValue) and self.data_source.flavour != 'obs':
            idxs_ = np.unique(np.concatenate([self.idxs, self.idxs_mf])).astype(int)
            for taski, task in enumerate(['generate_sky', 'generate_obs']):
                _jobs = []

                if task == 'generate_sky':
                    for idx in idxs_:
                        missing_files = any(not os.path.isfile(opj(self.libdir_sky, self.fns_sky[f].format(idx))) for f in required_files) or any(not os.path.exists(opj(self.libdir_sky, fnsec[comp].format(idx))) for fnsec in self.fns_sec.values() for comp in fnsec.keys())
                        if missing_files:
                            _jobs.append(idx)
                
                if task == 'generate_obs':
                    for idx in idxs_:
                        missing_files = any(not os.path.isfile(opj(self.libdir, self.fns[f].format(idx))) for f in required_files)
                        if missing_files:
                            _jobs.append(idx)  

                jobs[taski] = _jobs
            self.jobs = jobs
        else:
            self.jobs = [[],[]]
        return self.jobs


    # @log_on_start(logging.DEBUG, "Sim.run() started")
    # @log_on_end(logging.DEBUG, "Sim.run() finished")
    def run(self):
        for taski, task in enumerate(['generate_sky', 'generate_obs']):
            for idx in self.jobs[taski][mpi.rank::mpi.size]:
                log.info(f"rank {mpi.rank} (size {mpi.size}) {task} sim {idx}")
                if task == 'generate_sky':
                    self.generate_sky(idx)
                if task == 'generate_obs':
                    self.generate_obs(idx)
                if np.all(self.data_source.obs_lib.maps == DEFAULT_NotAValue):
                    self.data_source.purgecache()
        if np.all(self.data_source.maps == DEFAULT_NotAValue):
            self._postrun_sky()
            self._postrun_obs()


    def purgecache(self):
        self.data_source.purgecache()


    # @log_on_start(logging.DEBUG, "Sim.generate_sim(idx={idx}) started")
    # @log_on_end(logging.DEBUG, "Sim.generate_sim(idx={idx}) finished")
    def generate_sky(self, idx):
        for sec, secinfo in self.data_source.sec_info.items():
            # FIXME if there is cross-correlation between the components, we need to generate them together
            for comp in secinfo['component']:
                if not os.path.exists(opj(self.libdir_sky, self.fns_sec[sec][comp].format(idx))):
                    s = self.data_source.get_sim_sec(idx, space='alm', secondary=sec, component=comp)
                    np.save(opj(self.libdir_sky, self.fns_sec[sec][comp].format(idx)), s)

        for field in required_files_map.get(self.k, []):
            filepath = opj(self.libdir_sky, self.fns_sky[field].format(idx))
            if not os.path.exists(filepath):
                if field in ['E', 'B']:
                    EBsky = self.data_source.get_sim_sky(idx, spin=0, space='alm', field='polarization')
                    np.save(opj(self.libdir_sky, self.fns_sky['E'].format(idx)), EBsky[0])
                    np.save(opj(self.libdir_sky, self.fns_sky['B'].format(idx)), EBsky[1])
                    break
                if field == 'T':
                    Tsky = self.data_source.get_sim_sky(idx, spin=0, space='alm', field='temperature')
                    np.save(filepath, Tsky)


    # @log_on_start(logging.DEBUG, "Sim.generate_sim(idx={idx}) started")
    # @log_on_end(logging.DEBUG, "Sim.generate_sim(idx={idx}) finished")
    def generate_obs(self, idx):
        for field in required_files_map.get(self.k, []):
            filepath = opj(self.libdir, self.fns[field].format(idx))  
            if not os.path.exists(filepath):
                if field in ['E', 'B']:
                    EBobs = self.data_source.get_sim_obs(idx, spin=0, space='alm', field='polarization')
                    np.save(opj(self.libdir, self.fns['E'].format(idx)), EBobs[0])
                    np.save(opj(self.libdir, self.fns['B'].format(idx)), EBobs[1])
                    break

                if field == 'T':
                    Tobs = self.data_source.get_sim_obs(idx, spin=0, space='alm', field='temperature')
                    np.save(filepath, Tobs)


    def _postrun_obs(self):
        # NOTE if this class here decides to generate data, we need to update some parameters in the data_source object
        # NOTE if later reconstruction is run with the same config file, these updates also make sure they find the data without having to update the config file
        if self.data_source.flavour != 'sky' and self.data_source.flavour != 'obs' and np.all(self.data_source.obs_lib.maps == DEFAULT_NotAValue):
            self.data_source.libdir = self.libdir
            self.data_source.fns = self.fns
            if self.data_source.flavour != 'obs':
                self.data_source.obs_lib.CMB_info['fns'] = self.fns
                self.data_source.obs_lib.CMB_info['libdir'] = self.libdir
                self.data_source.obs_lib.CMB_info['space'] = 'alm'
                self.data_source.obs_lib.CMB_info['spin'] = 0

                self.obs_lib = self.data_source.obs_lib


    def _postrun_sky(self):
        # NOTE if this class here decides to generate data, we need to update some parameters in the data_source object
        # NOTE if later reconstruction is run with the same config file, these updates also make sure they find the data without having to update the config file
        

        if not self.data_source.flavour in ['sky', 'obs'] and np.all(self.data_source.obs_lib.maps == DEFAULT_NotAValue):
            self.data_source.sky_lib.CMB_info['fns'] = self.fns_sky
            self.data_source.sky_lib.CMB_info['libdir'] = self.libdir_sky
            self.data_source.sky_lib.CMB_info['space'] = 'alm'
            self.data_source.sky_lib.CMB_info['spin'] = 0


        # NOTE for pri_lib we set the paths to the generated secondaries
        for sec, secinfo in self.data_source.operator_info.items():
            self.data_source.pri_lib.sec_info[sec]['fn'] = self.fns_sec[sec]
            self.data_source.pri_lib.sec_info[sec]['libdir'] = self.libdir_sky
            self.data_source.pri_lib.sec_info[sec]['space'] = 'alm'
            self.data_source.pri_lib.sec_info[sec]['spin'] = 0
            self.data_source.pri_lib.sec_info[sec]['lm_max'] = secinfo['lm_max']
            self.data_source.pri_lib.sec_info[sec]['component'] = secinfo['component']


    def get_sim_sky(self, idx, space, field, spin):
        return self.data_source.get_sim_sky(idx=idx, space=space, field=field, spin=spin)

    def get_sim_pri(self, idx, space, field, spin):
        return self.data_source.get_sim_pri(idx=idx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, idx, space, field, spin, lm_max=None):
        if self.sky_coverage == 'full':
            assert space == 'alm', "'full' sky_coverage only works for space = alm"
            return  alm_copy_nd(self.data_source.get_sim_obs(idx=idx, space=space, field=field, spin=spin), None, lm_max)
        elif self.sky_coverage == 'masked':
            assert self.mask, "mask must be provided for sky_coverage = 'masked'"
            # FIXME if data is already masked (e.g. provided from disk), this will doubly mask the data.. not sure we want this
            assert space == 'map', "'masked' sky_coverage only works for space = map"
            assert field == 'polarization'
            obs = alm_copy_nd(self.data_source.get_sim_obs(idx=idx, space='alm', field=field, spin=0), None, lm_max)
            obs = hp.alm2map_spin(obs, nside=2048, spin=2, lmax=lm_max[0], mmax=lm_max[1])
            return np.array([dat*self.mask for dat in obs])

    def get_sim_noise(self, idx, space, field, spin=2):
        return self.data_source.get_sim_noise(idx, spin=spin, space=space, field=field)
    
    def get_sim_sec(self, idx, space, secondary=None, component=None, return_nonrec=False):
        return self.data_source.get_sim_sec(idx=idx, space=space, secondary=secondary, component=component, return_nonrec=return_nonrec)
    
    def get_fidCMB(self, idx, component):
        return self.data_source.get_fidCMB(idx=idx, component=component)

    def get_fidsec(self, idx, secondary=None, component=None, return_nonrec=False):
        return self.data_source.get_fidsec(idx=idx, secondary=secondary, component=component, return_nonrec=return_nonrec)
    
    # compatibility with Plancklens
    def hashdict(self):
        return {}

    def get_sim_pmap(self, idx):
        if self.sky_coverage == 'full':
            return self.data_source.get_sim_obs(idx=idx, space='map', field='polarization', spin=2)
        elif self.sky_coverage == 'masked':
            assert self.mask, "mask must be provided for sky_coverage = 'masked'"
            # FIXME if data is already masked (e.g. provided from disk), this will doubly mask the data.. not sure we want this
            obs = self.data_source.get_sim_obs(idx=idx, space='map', field='polarization', spin=2)
            return np.array([dat*self.mask for dat in obs])
        

    def get_data(self, idx):
        # TODO this could be provided by the data_container directly
        space = 'alm' if self.sky_coverage == 'full' else 'map'
        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            # NOTE dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
                
            if self.data_key in ['p', 'eb', 'be']:
                ret = alm_copy_nd(
                    self.data_source.get_sim_obs(idx, space='alm', spin=0, field='polarization'),
                    None, self.lm_max_sky)                
            elif self.data_key in ['ee']:
                ret = alm_copy_nd(
                    self.data_source.get_sim_obs(idx, space='alm', spin=0, field='polarization'),
                    None, self.lm_max_sky)[0]
            elif self.data_key in ['tt']:
                ret = alm_copy_nd(
                    self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'),
                    None, self.lm_max_sky)
            elif self.data_key in ['p']:
                EBobs = alm_copy_nd(
                    self.data_source.get_sim_obs(idx, space='alm', spin=0, field='polarization'),
                    None, self.lm_max_sky)
                Tobs = alm_copy_nd(
                    self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'),
                    None, self.lm_max_sky)         
                ret = np.array([Tobs, *EBobs])
            else:
                assert 0, 'implement if needed'
            if space == 'alm':
                return ret
            elif space == 'map':
                assert self.data_key == 'p', 'implement if needed'
                # TODO this should move to the data_container
                ret = np.atleast_2d(ret)
                import healpy as hp
                return [hp.alm2map_spin(r, nside=2048, spin=2) for r in ret]
            else:
                assert 0, 'implement if needed'
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.idx), dtype=float)
            else:
                assert 0, 'implement if needed'


class QEScheduler:
    """Quadratic estimate lensing reconstruction Job. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """
    @check_MPI
    def __init__(self, QE_job_desc, QE_searchs_desc, data_container, TEMP):
        # NOTE plancklens uses get_sim_pmap() from data_container.
        # DataContainer updates the data_container object with the libdirs and fns if it generated simulations, so need to update this
        self.data_container = data_container

        self.tasks = QE_job_desc['tasks']
        self.idxs = QE_job_desc['idxs']
        self.idxs_mf = QE_job_desc['idxs_mf']
        self.TEMP = TEMP

        # I want to have a QE search for each field
        for QE_search_desc in QE_searchs_desc.values():
            QE_search_desc['idxs_mf'] = self.idxs_mf
            QE_search_desc['QE_filterqest_desc']['data_container'] = self.data_container
        self.QE_searchs = [QE_handler.base(**QE_search_desc) for name, QE_search_desc in QE_searchs_desc.items()]

        self.secondary2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self.idx2secondary = {i: QE_search.secondary.ID for i, QE_search in enumerate(self.QE_searchs)}

        self.template_operator = QE_job_desc['template_operator'] # FIXME deal with this later
        
        # if there is no job, we can already init the filterqest
        if len(np.array([x for x in self.collect_jobs().ravel() if x is not None]))==0:
            for QE_search in self.QE_searchs:
                QE_search.init_filterqest()


    def collect_jobs(self, recalc=False):
        jobs = list(range(len(self.tasks)))
        for taski, task in enumerate(self.tasks):
            _jobs = []
            if task == 'calc_fields':
                _nomfcheck = True if self.idxs_mf == [] else False
                for idx in self.idxs:
                    __jobs = []
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        _add = False
                        for ci, component in enumerate(QE_search.secondary.component):
                            if _nomfcheck or not QE_search.secondary.cacher.is_cached(QE_search.secondary.qmflm_fns[component].format(idx=idx)) or recalc:
                                if not QE_search.secondary.cacher.is_cached(QE_search.secondary.klm_fns[component].format(idx=idx)) or recalc:
                                   _add = True
                        __jobs.append(idx) if _add else __jobs.append(None)
                    _jobs.append(__jobs)
            jobs.append(_jobs)
             
            if task == 'calc_meanfields':
                for idx in self.idxs_mf:
                    __jobs = []
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        _add = False
                        for ci, component in enumerate(QE_search.secondary.component): # each field has n components # fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.idxs_mf)))
                            mf_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qmflm_fns[component])
                            if not os.path.isfile(mf_fn) or recalc:
                                field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=idx) if idx != -1 else 'dat_%s.fits'%self.k)
                                if not os.path.isfile(field_fn) or recalc:
                                    _add = True
                         # checking for each component, but only adding the complete field as task index
                        __jobs.append(idx) if _add else __jobs.append(None)
                    _jobs.append(__jobs)
                jobs.append(_jobs)

            # TODO later. If i add combinatorics here across all operators, could add this to the collect list.
            if task == 'calc_templates':
                for idx in self.idxs:
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        __jobs = []
                        for ci, component in enumerate(QE_search.te.components): # each field has n components # fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.idxs_mf)))
                            tepmplate_fn = opj(QE_search.libdir, 'templates', QE_search.template.qmflm_fns[component])
                            if not os.path.isfile(tepmplate_fn) or recalc:
                                field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=idx) if idx != -1 else 'dat_%s.fits'%self.k)
                                if not os.path.isfile(field_fn) or recalc:
                                    _jobs.append(int(idx))
            jobs[taski] = _jobs
        self.jobs = jobs
        if not np.all(np.array(jobs)==None):
            log.info(f"QE jobs: {jobs}")
        return np.array(jobs)


    def run(self, task=None):
        if not np.all(np.array(self.jobs)==None):
            log.info(f"Running QE jobs: {self.jobs}")
        if True: # 'triggers calc_cinv'
            self.init_QEsearchs()
                   
        tasks = self.tasks if task is None else [task]
        for taski, task in enumerate(tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_fields':
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for seci, secidx in enumerate(idxs):
                        if secidx is not None: #these Nones come from the field already being done.
                            self.QE_searchs[seci].get_qlm(int(secidx))
                            self.QE_searchs[seci].get_est(int(secidx)) # this is here for convenience
                    if np.all(self.data_container.obs_lib.maps == DEFAULT_NotAValue):
                        self.data_container.data_source.purgecache()
                mpi.barrier()

            if task == 'calc_meanfields':
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for QE_search in self.QE_searchs:
                        for seci, secidx in enumerate(idxs):
                            self.QE_searchs[seci].get_qlm(int(secidx))
                            self.QE_searchs[seci].get_est(int(secidx)) # this is here for convenience
                for QE_search in self.QE_searchs:
                    QE_search.get_qmflm(QE_search.estimator_key, self.idxs_mf)
                mpi.barrier()

            # TODO later
            if task == 'calc_templates':
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    # For each combination of operators, I want to build templates
                    # jobs list could come as [idx-delta,idx-beta,idx-deltabeta] for each idx
                    idx, operator_indexs = idxs.split('-')
                    self.get_template(idx, operator_indexs)
                    if np.all(self.data_container.obs_lib.maps == DEFAULT_NotAValue):
                        self.data_container.purgecache()


    def get_qlm(self, idx, it=0, secondary=None, component=None):
        assert it == 0, 'QE does not have iterations, leave blank or set it=0'
        if secondary not in self.secondary2idx:
            print(f'secondary {secondary} not found. Available secondaries are: ', self.secondary2idx.keys())
            return np.array([[]])
        return self.QE_searchs[self.secondary2idx[secondary]].get_qlm(idx, component)
    

    def get_est(self, idx, it=0, secondary=None, component=None, subtract_meanfield=None, scale='k'):
        self.init_QEsearchs()
        if isinstance(it, (int,np.int64)):
            assert it == 0, 'QE does not have iterations, leave blank or set it=0'
        else:
            assert 0 in it, 'QE does not have iterations, leave blank or set it=0, not {}'
            return [self.get_est(idx, 0, secondary, component, subtract_meanfield, scale)]
        if secondary is None:
            return [self.QE_searchs[secidx].get_est(idx, component, subtract_meanfield, scale=scale) for secidx in self.secondary2idx.values()]
        if isinstance(secondary, list):
            return [self.QE_searchs[self.secondary2idx[sec]].get_est(idx, component, subtract_meanfield, scale=scale) for sec in secondary]
        if secondary not in self.secondary2idx:
            print('secondary not found. Available secondaries are: ', self.secondary2idx.keys())
            return np.array([[]])
        return self.QE_searchs[self.secondary2idx[secondary]].get_est(idx, component, subtract_meanfield, scale=scale)


    def get_template(self, idx, it=0, secondary=None, component=None, calc=False):
        assert it==0, 'QE does not have iterations, leave blank or set it=0'
        path = opj(self.QE_searchs[0].fq.libdir, 'template', f"template_sim{idx}_it{it}")
        if not os.path.isfile(path):
            if not self.QE_searchs[self.secondary2idx[secondary]].is_cached(self, idx, component, type='qlm'):
                if not calc:
                    print(f'cannot generate template as estimate of secondary {secondary} with idx {idx} not found, set calc=True to calculate')
                    return np.array([[]])
                self.get_est(idx, it, secondary, component)
            self.template_operator.set_field(idx, it)
            estCMB = self.get_wflm(idx, it)
            np.save(path, self.template_operator.act(estCMB))
        return self.template_operator.act(estCMB)


    def get_wflm(self, idx, it, lm_max):
        if it!=0:
            print('QE does not have iterations, leave blank or set it=0')
            return np.array([[]])
        return self.QE_searchs[0].get_wflm(idx, lm_max)


    def get_ivflm(self, idx, it, lm_max):
        if it!=0:
            print('QE does not have iterations, leave blank or set it=0')
            return np.array([[]])
        return self.QE_searchs[0].get_ivflm(idx, lm_max)
    

    def init_QEsearchs(self):
        __init = False
        first_rank = mpi.bcast(mpi.rank)
        for QE_search in self.QE_searchs:
            if 'qlms' not in QE_search.__dict__:
                __init = True
                break
        if __init:
            if first_rank == mpi.rank:
                mpi.disable()
                for QE_search in self.QE_searchs:
                    QE_search.init_filterqest()
                mpi.enable()
                [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)
            for QE_search in self.QE_searchs:
                QE_search.init_filterqest()


    def maxiterdone(self, idx):
        return self.QE_searchs[0].isdone(idx)


class MAPScheduler:
    MAP_minimizers: List[MAP_handler.Minimizer]
    def __init__(self, idxs, idxs_mf, data_container, QE_searchs, tasks, MAP_minimizers):
        self.data_container = data_container

        self.idxs = idxs
        self.idxs_mf = idxs_mf
        self.QE_searchs: QEScheduler = QE_searchs

        self._sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self._seclist_sorted = sorted(list(self._sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))

        self.MAP_minimizers: MAP_handler.Minimizer = MAP_minimizers
        self.tasks = tasks
        for idx in self.idxs:
            if np.all([self.QE_searchs[0].isdone(idx, comp) for comp in self.QE_searchs[0].secondary.component]):
                if mpi.rank == 0:
                    self.MAP_minimizers[idx].likelihood.copyQEtoDirectory(QE_searchs)


    def collect_jobs(self):
        jobs = list(range(len(self.tasks)))
        for taski, task in enumerate(self.tasks):
            _jobs = []
            if task == 'calc_fields':
                for idxi, idx in enumerate(self.idxs):
                    if self.MAP_minimizers[idxi].maxiterdone() < self.MAP_minimizers[idxi].itmax:
                        _jobs.append(idx)
                jobs[taski] = _jobs
        self.jobs = jobs
        return jobs


    def run(self):
        for idx in self.idxs:
            if np.all([self.QE_searchs[0].isdone(idx, comp) for comp in self.QE_searchs[0].secondary.component]):
                if mpi.rank == 0:
                    self.MAP_minimizers[idx].likelihood.copyQEtoDirectory(self.QE_searchs)

        for taski, task in enumerate(self.tasks):
            log.info('{}, MAP task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
            if task == 'calc_fields':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.MAP_minimizers[idx].get_est(self.MAP_minimizers[idx].itmax)


    def get_est(self, idx, it=None, secondary=None, component=None, scale='k', subtract_QE_meanfield=True, calc_flag=False, idx2=None):
        if isinstance(secondary, str) and secondary not in self.seclist_sorted:
            print('Secondary not found. Available secondaries are:', self.seclist_sorted)
            return np.array([[]])
        if it is None:
            it = self.MAP_minimizers[idx].maxiterdone()

        for idx in self.idxs:
            self.MAP_minimizers[idx].likelihood.copyQEtoDirectory(self.QE_searchs)
        def get_map_est(it_):
            return self.MAP_minimizers[idx].get_est(it_, secondary, component, scale, calc_flag)

        if isinstance(it, (list, np.ndarray)):
            it = np.array(it)
        return get_map_est(it)


    def get_qlm(self, idx, it, secondary=None, component=None, idx2=None):
        if secondary is None:
            return [self.QE_searchs[self.sec2idx[QE_search.ID]].get_qlm(idx, component) for QE_search in self.QE_searchs]
        if it==0:
            return self.QE_searchs[self.sec2idx[secondary]].get_qlm(idx, component)
        print('only available for QE, set it=0')


    def get_template(self, idx, it, secondary=None, component=None, idx2=None, lm_max_in=None, lm_max_out=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        stash = ctx.idx, ctx.idx2, ctx.component
        ctx.set(idx=idx, secondary=secondary, component=component)
        assert it>0, 'Need to correctly implement QE template generation first'
        assert it <= self.maxiterdone(), 'Requested iteration is not available'
        res = self.MAP_minimizers[idx].likelihood.gradient_lib.wf_filter.get_template(it, secondary, component, lm_max_in, lm_max_out)
        ctx.set(idx=stash[0], idx2=stash[1], component=stash[2])
        secondary = secondary or self.MAP_minimizers[idx].likelihood.secondaries.keys()
        if isinstance(it, (list, np.ndarray)):
            ret = [alm_copy_nd(res[it_], None, lm_max_out) for it_ in it]
        else:
            ret = alm_copy_nd(res, None, lm_max_out)
        return ret


    def get_wflm(self, idx, it=None, lm_max=None, idx2=None):
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            return self.QE_searchs[0].get_wflm(idx, lm_max=lm_max)
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        stash = ctx.idx, ctx.idx2, ctx.component
        ctx.set(idx=idx)
        ret = self.MAP_minimizers[idx].get_wflm(it)
        ctx.set(idx=stash[0], idx2=stash[1], component=stash[2])
        return ret


    def get_ivflm(self, idx, it=0, idx2=None):
        # NOTE currently no support for list of secondary or it
        if it==0:
            return self.QE_searchs[0].get_ivflm(idx)
        print('only available for QE, set it=0')


    def get_ivfreslm(self, idx, it=None, idx2=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        stash = ctx.idx, ctx.idx2, ctx.component
        ctx.set(idx=idx)
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            print('only available for MAP, set it>0')
        ret = self.MAP_minimizers[idx].get_ivfreslm(it)
        ctx.set(idx=stash[0], idx2=stash[1], component=stash[2])
        return ret


    def maxiterdone(self):
        return min([MAP_search.maxiterdone() for MAP_search in self.MAP_minimizers])
    

    def get_gradient_quad(self, idx, it, secondary=None, component=None, idx2=None):
        self.MAP_minimizers[idx].ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizers[idx].get_gradient_quad(it=it)
    
    def get_gradient_total(self, idx, it, secondary=None, component=None, idx2=None):
        self.MAP_minimizers[idx].ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizers[idx].get_gradient_total(it=it)
    
    def get_gradient_prior(self, idx, it, secondary=None, component=None, idx2=None):
        self.MAP_minimizers[idx].ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizers[idx].get_gradient_prior(it=it)
    
    def get_gradient_meanfield(self, idx, it, secondary=None, component=None, idx2=None):
        self.MAP_minimizers[idx].ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizers[idx].get_gradient_meanfield(it=it)
    

    def __getattr__(self, name):
        # Forward the method call to the minimizer specified by idx
        def method_forwarder(idx, *args, **kwargs):
            if idx < len(self.MAP_minimizers):
                minimizer = self.MAP_minimizers[idx]
                if name in functionforwardlist and hasattr(minimizer, name):
                    return getattr(minimizer, name)(idx, *args, **kwargs)  # Pass idx to the minimizer
                else:
                    raise AttributeError(f"method {name} not found in MAP_minimizer")
            raise IndexError(f"scheduler has no MAP_minimizer at index {idx}")

        return method_forwarder


class MapDelenser(Basejob):
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
        self.simgen = DataContainer(dlensalot_model)
        self.libdir_delenser = opj(self.TEMP, 'delensing/{}'.format(self.dirid))
        if not(os.path.isdir(self.libdir_delenser)):
            os.makedirs(self.libdir_delenser)
        self.fns = opj(self.libdir_delenser, 'ClBB_sim{:04d}.npy')


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "collect_jobs() started")
    # @log_on_end(logging.DEBUG, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO a valid job is any requested job?, as BLTs may also be on CFS
        jobs = []
        for idx in self.idxs:
            jobs.append(idx)
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "run() started")
    # @log_on_end(logging.DEBUG, "run() finished")
    def run(self):
        outputdata = self._prepare_job()
        if self.jobs != []:
            for idx in self.jobs[mpi.rank::mpi.size]:
                log.debug('will store file at: {}'.format(self.fns.format(idx)))
                self.delens(idx, outputdata)


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
                    a = OverwriteAnafast() if self.cl_calc == hp else MaskedLib(mask, self.cl_calc, self.lmax, self.lmax_mask)
                    outputdata = np.zeros(shape=(2, 2+len(self.its), len(self.nlevels)+len(self.masks_fromfn), self.lmax+1))
                    self.lib[maskflavour].update({maskid: a})

        return outputdata
    

    # # @log_on_start(logging.DEBUG, "get_basemap() started")
    # # @log_on_end(logging.DEBUG, "get_basemap() finished")  
    def get_basemap(self, idx):
        '''
        Return a B-map to be delensed. Can be the map handled in the sims_lib library (basemap='lens'), 'lens_ffp10' (these are the ffp10 relizations on NERSC),
        or the observed map itself, in which case the residual foregrounds and noise will still be in there.
            gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lm_max_blt[1])
        '''
        # TODO depends if data comes from delensalot simulations or from external.. needs cleaner implementation
        if self.basemap == 'pico_dc2_lens':
            from delensalot.utils import cli
            FWHMs = [38.4, 32.0, 28.3, 23.6, 22.2, 18.4, 12.8, 10.7, 9.5, 7.9, 7.4, 6.2, 4.3, 3.6, 3.2, 2.6, 2.5, 2.1, 1.5, 1.3, 1.1]
            freqs = [21, 25, 30, 36, 43, 52, 62, 75, 90, 108, 129, 155, 186, 223, 268, 321, 385, 462, 555, 666, 799]
            beams = np.array([hp.gauss_beam(FWHM / 180. / 60. * np.pi, lmax=2000) for FWHM in FWHMs])
            map = dict()
            map.update({freqs[-1]: hp.read_map('/pscratch/sd/e/erussie/PICO/data/maps/pysm_3.4.0_maps/c4_{freq:03d}_4096.fits'.format(freq=freqs[-1]), field=(1,2))})
            log.info('loaded input B map')
            temp_ = hp.map2alm_spin(map[freqs[-1]], spin=2, lmax=self.lm_max_blt[0])
            return hp.almxfl(temp_[1],cli(beams[-1]))*1e6
        if self.basemap == 'lens': 
            return alm_copy(
                    self.data_container.get_sim_sky(idx, space='alm', spin=0, field='polarization')[1],
                    self.data_container.lmax, *self.lm_max_blt
                )
        elif self.basemap == 'lens_ffp10':
                return alm_copy(
                    planck2018_sims.cmb_len_ffp10.get_sim_blm(idx),
                    None,
                    lmaxout=self.lm_max_blt[0],
                    mmaxout=self.lm_max_blt[1]
                )  
        else:
            # only checking for map to save some memory..
            if np.all(self.data_container.maps == DEFAULT_NotAValue):
                return alm_copy(self.data_container.get_sim_obs(idx, space='alm', spin=0, field='polarization')[1], self.data_container.lmax, *self.lm_max_blt)
            else:
                return hp.map2alm_spin(self.data_container.get_sim_obs(idx, space='map', spin=2, field='polarization'), spin=2, lmax=self.lm_max_blt[0], mmax=self.lm_max_blt[1])[1]

    
    @log_on_start(logging.DEBUG, "_delens() started")
    @log_on_end(logging.DEBUG, "_delens() finished")
    def delens(self, idx, outputdata):
        blm_L = self.get_basemap(idx)
        log.info('got inbut Blms')
        blt_QE = self.get_blt_it(idx, 0)
        
        bdel_QE = self.nivjob_geomlib.alm2map(blm_L-blt_QE, *self.lm_max_blt, nthreads=4)
        del blt_QE
        maskcounter = 0
        for maskflavour, masks in self.binmasks.items():
            for maskid, mask in masks.items():
                log.info("starting mask {} {}".format(maskflavour, maskid))
                
                bcl_L = self.lib[maskflavour][maskid].map2cl(self.nivjob_geomlib.alm2map(blm_L, *self.lm_max_blt, nthreads=4))
                outputdata[0][0][maskcounter] = bcl_L

                blt_L_QE = self.lib[maskflavour][maskid].map2cl(bdel_QE)
                outputdata[0][1][maskcounter] = blt_L_QE

                for iti, it in enumerate(self.its):
                    blt_MAP = self.get_blt_it(idx, it)
                    bdel_MAP = self.nivjob_geomlib.alm2map(blm_L-blt_MAP, *self.lm_max_blt, nthreads=4)
                    blt_L_MAP = self.lib[maskflavour][maskid].map2cl(bdel_MAP)    
                    outputdata[0][2+iti][maskcounter] = blt_L_MAP
                    log.info("Finished MAP delensing for idx {}, iteration {}".format(idx, it))

                maskcounter+=1

        np.save(self.fns.format(idx), outputdata)
            

    # @log_on_start(logging.DEBUG, "get_residualblens() started")
    # @log_on_end(logging.DEBUG, "get_residualblens() finished")
    def get_residualblens(self, idx, it):
        basemap = self.get_basemap(idx)
        
        return basemap - self.get_blt_it(idx, it)
    

    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "read_data() started")
    # @log_on_end(logging.DEBUG, "read_data() finished")
    def read_data(self):
        bcl_L = np.zeros(shape=(len(self.its)+2, len(self.nlevels)+len(self.masks_fromfn), len(self.idxs), len(self.edges)-1))
        for idxi, idx in enumerate(self.idxs):
            data = np.load(self.fns.format(idx))
            bcl_L[0,:,idxi] = data[0][0]
            bcl_L[1,:,idxi] = data[0][1]
            for iti, it in enumerate(self.its):
                bcl_L[2+iti,:,idxi] = data[0][2+iti]

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


class PhiAnalyser(Basejob):
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
            self.WFemps = np.load(opj(self.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.idxs), len(self.its))) if self.custom_WF_TEMP else [None for n in np.arange(len(self.its))]
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
                for idx in self.idxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, idx, it)):
                            _jobs.append(idx)
                            break
       
            if task == 'calc_crosscorr':
                fns = opj(self.TEMP_Cx,'CLx_%s_sim%s_it%s.npy')
                for idx in self.idxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, idx, it)):
                            _jobs.append(idx)
                            break
            
            if task == 'calc_reconbias':
                fns = opj(self.TEMP_Cxbias,'CLxb_%s_sim%s_it%s.npy')
                for idx in self.idxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, idx, it)):
                            _jobs.append(idx)
                            break

            if task == 'calc_crosscorrcoeff':
                fns = opj(self.TEMP_Cccc,'CLccc_%s_sim%s_it%s.npy')
                for idx in self.idxs:
                    for it in self.its:
                        if not os.path.isfile(fns%(self.k, idx, it)):
                            _jobs.append(idx)
                            break
            jobs.append(_jobs)
        self.jobs = jobs


    def run(self):
        # Wait for everyone to finish previous job
        mpi.barrier()
        for taski, task in enumerate(self.tasks):
            if task == 'calc_WFemp':
                # First, calc for each simindex individually
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    val = self._get_wienerfilter_empiric(idx, self.its)
                # Second, calc average WF, only let only one rank do this
                first_rank = mpi.bcast(mpi.rank)
                if first_rank == mpi.rank:
                    self.get_wienerfilter_empiric()
                    [mpi.send(1, dest=dest) for dest in range(0,mpi.size) if dest!=mpi.rank]
                else:
                    mpi.receive(None, source=mpi.ANY_SOURCE)

            if task == 'calc_crosscorr':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_crosscorrelation(idx, it, WFemps=self.WFemps)
                        self.get_autocorrelation(idx, it, WFemps=self.WFemps)
           
            if task == 'calc_reconbias':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_reconstructionbias(idx, it, WFemps=self.WFemps)

            if task == 'calc_crosscorrcoeff':  
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    for it in self.its:
                        self.get_crosscorrelationcoefficient(idx, it, WFemps=self.WFemps)


    def get_crosscorrelation(self, idx, it, WFemps=None):
        TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        fns = opj(TEMP_Cx,'CLx_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cx,'CLx_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, idx, it)):
            plm_est = self.get_plm_it(idx, [it])[0]
            plm_in = alm_copy(self.data_container.get_sim_phi(idx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.idxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None)/WFemps[it]
            np.save(fns%(self.k, idx, it), val)
        return np.load(fns%(self.k, idx, it))


    def get_autocorrelation(self, idx, it, WFemps=None):
        # Note: this calculates auto of the estimate
        TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        fns = opj(TEMP_Cx,'CLa_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cx,'CLa_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, idx, it)):
            plm_est = self.get_plm_it(idx, [it])[0]
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.idxs), len(self.its))) 
            val = alm2cl(plm_est, plm_est, None, None, None)/WFemps[it]**2
            np.save(fns%(self.k, idx, it), val)
        return np.load(fns%(self.k, idx, it))


    def get_reconstructionbias(self, idx, it, WFemps=None):
        TEMP_Cxbias = opj(self.libdir_phianalayser, 'Cxb')
        fns = opj(TEMP_Cxbias,'CLxb_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cxbias,'CLxb_%s_sim%s_it%s.npy') 
        if not os.path.isfile(fns%(self.k, idx, it)):
            plm_est = self.get_plm_it(idx, [it])[0]
            plm_in = alm_copy(self.data_container.get_sim_phi(idx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.idxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None) / alm2cl(plm_in, plm_in, None, None, None)/WFemps[it]
            np.save(fns%(self.k, idx, it), val)
        return np.load(fns%(self.k, idx, it))


    def get_crosscorrelationcoefficient(self, idx, it, WFemps=None):
        TEMP_Cccc = opj(self.libdir_phianalayser, 'Cccc')
        fns = opj(TEMP_Cccc,'CLccc_%s_sim%s_it%s_customWF.npy') if self.custom_WF_TEMP else opj(TEMP_Cccc,'CLccc_%s_sim%s_it%s.npy')
        if not os.path.isfile(fns%(self.k, idx, it)):
            # plm_QE = almxfl(self.qe.get_sim_qlm(idx), utils.cli(R))
            plm_est = self.get_plm_it(idx, [it])[0]
            plm_in = alm_copy(self.data_container.get_sim_phi(idx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            if type(WFemps) != np.ndarray:
                WFemps = np.load(opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')%(self.k, len(self.idxs), len(self.its))) 
            val = alm2cl(plm_est, plm_in, None, None, None)**2/(alm2cl(plm_est, plm_est, None, None, None)*alm2cl(plm_in, plm_in, None, None, None))
            np.save(fns%(self.k, idx, it), val)
        return np.load(fns%(self.k, idx, it))



    def get_wienerfilter_analytic(self, idx, it):
        assert 0, 'implement if needed'
        return None   
    

    def _get_wienerfilter_empiric(self, idx, its):
        ## per sim calculation, no need to expose this. Only return averaged result across all sims, which is the function without the pre underline: get_wienerfilter_empiric()
        fns = opj(self.TEMP_WF, 'WFemp_%s_sim%s_it%s.npy')
        WFemps = np.zeros(shape=(len(its), self.lm_max_qlm[0]+1))
        if not np.array([os.path.isfile(fns%(self.k, idx, it)) for it in its]).all():   
            plm_in = alm_copy(self.data_container.get_sim_phi(idx, space='alm'), None, self.lm_max_qlm[0], self.lm_max_qlm[1])
            plm_est = self.get_plm_it(idx, its)
            for it in its:       
                WFemps[it] = alm2cl(plm_in, plm_est[it], None, None, None)/alm2cl(plm_in, plm_in, None, None, None)
                np.save(fns%(self.k, idx, it), WFemps[it])
        for it in its:
            WFemps[it] = np.load(fns%(self.k, idx, it))
        return WFemps
    
    def get_wienerfilter_empiric(self):
        fn = opj(self.TEMP_WF,'WFemp_%s_simall%s_itall%s_avg.npy')
        if not os.path.isfile(fn%(self.k, len(self.idxs), len(self.its))):   
            WFemps = np.array([self._get_wienerfilter_empiric(idx, self.its) for idx in self.idxs])
            np.save(fn%(self.k, len(self.idxs), len(self.its)), np.mean(WFemps, axis=0))
        return np.load(fn%(self.k, len(self.idxs), len(self.its)))
        

class OverwriteAnafast():
    """Convenience class for overwriting method name
    """    

    def map2cl(self, *args, **kwargs):
        return hp.anafast(*args, **kwargs)


class MaskedLib:
    """Convenience class for handling method names
    """   
    def __init__(self, mask, cl_calc, lmax, lmax_mask):
        self.mask = mask
        self.cl_calc = cl_calc
        self.lmax = lmax
        self.lmax_mask = lmax_mask

    def map2cl(self, map):
        return self.cl_calc.map2cl(map, self.mask, self.lmax, self.lmax_mask)
