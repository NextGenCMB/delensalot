#!/usr/bin/env python

"""job_handler.py: This module collects the delensalot jobs. It receives the delensalot model build for the respective job. They all initialize needed modules and directories, collect the computing-jobs, and run the computing-jobs, with MPI support, if available.
"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
# from delensalot.config.etc import logger

from typing import List, Type, Union
import os
from os.path import join as opj
import hashlib
import datetime, getpass, copy

import numpy as np
import healpy as hp
from collections import UserDict

from plancklens.sims import planck2018_sims

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI
from delensalot.core.opfilt.bmodes_ninv import template_dense, template_bfilt
from delensalot.core.QE import handler as QE_handler
from delensalot.core.MAP import handler as MAP_handler, functionforwardlist
from delensalot.core.MAP.context import get_computation_context

from delensalot.sims.data_source import dirname_generator, dict2roundeddict

from delensalot.config.metamodel import DEFAULT_NotAValue
from delensalot.config.config_manager import get_config

from delensalot.utils import read_map, ztruncify, cli
from delensalot.utility.utils_hp import Alm, almxfl, alm_copy, gauss_beam, alm2cl, alm_copy_nd

import matplotlib.pyplot as plt


class ConstantDict(UserDict):
    def __init__(self, value):
        super().__init__()
        self._value = value

    def __getitem__(self, key):
        return self._value

    def get(self, key, default=None):
        return self._value


# NOTE This is to generate all maps no matter the CMB estimator request
required_files_map = {
    'p_p': ['E', 'B'], 'p_eb': ['E', 'B'], 'peb': ['E', 'B'], 'p_be': ['E', 'B'], 'pee': ['E', 'B'],
    'ptt': ['T'],
    'p': ['T', 'E', 'B']}
required_files_map = ConstantDict(['T', 'E', 'B'])

def get_hashcode(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:4]

def get_dirname(s):
    return str(s).translate(str.maketrans({"(": "", ")": "", "{": "", "}": "", "[": "", "]": "", 
                                            " ": "", "'": "", '"': "", ":": "_", ",": "_"}))


class OBDBuilder:
    """OBD matrix builder Job. Calculates the OBD matrix, used to correctly deproject the B-modes at a masked sky.
    """
    @check_MPI
    def __init__(self, OBD_model):
        self.__dict__.update(OBD_model.__dict__)
        nivp = self._load_niv(self.nivp_desc)
        # self.nivp = ztruncify(nivp, self.zbounds)


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
                    if mpi.rank==0: log.info(tnit.shape)
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
    def __init__(self, data_source, estimator_key, idxs, idxs_mf, mask_fn, sky_coverage, data_key, lm_max_sky):
        """ In this init we make the following checks:
         * (1) Does user provide obs data? Then DataContainer can be fully skipped
         * (2) Otherwise, check if files already generated (delensalot model may not know this, so need to search),
           * If so, update the simhandler with the respective libdirs and fns
           * If not,
             * generate the simulations
             * update the simhandler
        """
        self.data_source = data_source
        self.estimator_key = estimator_key
        self.idxs = idxs
        self.idxs_mf = idxs_mf
        self.mask_fn = mask_fn
        self.sky_coverage = sky_coverage
        if sky_coverage == 'masked':
            assert os.path.isfile(mask_fn), "mask must be provided for sky_coverage = 'masked'"
        self.data_key = data_key
        if self.estimator_key == 'p':
            self.data_key = 'tp'
            # FIXME 
        self.lm_max_sky = lm_max_sky

        if self.data_source.flavour == 'obs' or np.all(self.data_source.obs_lib.maps != DEFAULT_NotAValue): # (1)
            # Here, obs data is provided and nothing needs to be generated
            if np.all(self.data_source.obs_lib.maps != DEFAULT_NotAValue):
                if mpi.rank==0: log.info('Will use data provided in memory')
                pass
            else:
                if mpi.rank==0: log.info('Will use obs data stored at {} with filenames {}'.format(self.data_source.libdir, str(self.data_source.fns)))
                pass
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
                    os.makedirs(self.libdir, exist_ok=True)
                if mpi.size > 1:
                    for dest in range(mpi.size):
                        if dest != mpi.rank:
                            mpi.send(1, dest=dest)
            else:
                mpi.receive(None, source=mpi.ANY_SOURCE)

            required_files = required_files_map.get(self.estimator_key, [])
            idxs_ = np.unique(np.concatenate([self.idxs, self.idxs_mf])).astype(int)
            def check_and_log(libdir, fns, _postrun_method, data_type):
                """function to check file existence """
                if all(os.path.exists(opj(libdir, fns[f].format(idx))) for f in required_files for idx in idxs_):
                    _postrun_method()
                    if mpi.rank==0: log.info(f'will use {data_type} data at {libdir} with filenames {fns}')
                    pass
                else:
                    if mpi.rank==0: log.info(f'{data_type} data will be stored at {libdir} with filenames {fns}')
                    pass

            check_and_log(self.libdir, self.fns, self._postrun_obs, "obs")
            if self.data_source.flavour != 'sky':
                if all(os.path.exists(opj(self.libdir_sky, self.fns_sec[sec][component].format(idx))) for sec in self.fns_sec.keys() for component in self.fns_sec[sec] for idx in idxs_):
                    check_and_log(self.libdir_sky, self.fns_sky, self._postrun_sky, "sky")
                else:
                    if mpi.rank==0: log.info(f'sky data will be stored at {self.libdir_sky} with filenames {self.fns_sky}. All secondaries will be generated along the way')
                    pass

        self.cls_lib = self.data_source.cls_lib
        self.obs_lib = self.data_source.obs_lib

    
    def set_basename_sky(self):
        return {'T': 'Talmsky_{}.npy', 'E': 'Ealmsky_{}.npy', 'B': 'Balmsky_{}.npy'}


    def set_basename_obs(self):
        return {'T': 'Talmobs_{}.npy', 'E': 'Ealmobs_{}.npy', 'B': 'Balmobs_{}.npy'}

    # @base_exception_handler
    @log_on_start(logging.DEBUG, "DataContainer.collect_jobs() started")
    @log_on_end(logging.DEBUG, "DataContainer.collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        jobs = list(range(len(['generate_sky', 'generate_obs'])))
        required_files = required_files_map.get(self.estimator_key, [])
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
                if task == 'generate_sky':
                    self.generate_sky(idx)
                if task == 'generate_obs':
                    self.generate_obs(idx)
                if np.all(self.data_source.obs_lib.maps == DEFAULT_NotAValue):
                    self.data_source.purgecache()
        if np.all(self.data_source.maps == DEFAULT_NotAValue):
            if self.data_source.flavour != 'obs': self._postrun_sky()
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

        for field in required_files_map.get(self.estimator_key, []):
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
        for field in required_files_map.get(self.estimator_key, []):
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

                # NOTE dumping data_source info to readme
                config = get_config()
                np.savetxt(self.data_source.obs_lib.CMB_info['libdir'] + '/README_simulation_info.txt', np.array([str(config.data_source.__dict__).replace(" '", "\n'")]), fmt="%s")

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
            assert self.mask_fn, "mask must be provided for sky_coverage = 'masked'"
            # FIXME if data is already masked (e.g. provided from disk), this will doubly mask the data.. not sure we want this
            assert space == 'map', "'masked' sky_coverage only works for space = map"
            assert field == 'polarization'
            mask = hp.read_map(self.mask_fn)
            obs = alm_copy_nd(self.data_source.get_sim_obs(idx=idx, space='alm', field=field, spin=0), None, lm_max)
            obs = hp.alm2map_spin(obs, nside=2048, spin=2, lmax=lm_max[0], mmax=lm_max[1])
            ret = np.array([dat*mask for dat in obs])

            return ret

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

    def get_sim_tmap(self, idx):
        if self.sky_coverage == 'full':
            return self.data_source.get_sim_obs(idx=idx, space='map', field='temperature', spin=0)
        elif self.sky_coverage == 'masked':
            mask = np.load(self.mask_fn)
            # FIXME if data is already masked (e.g. provided from disk), this will doubly mask the data.. not sure we want this
            obs = self.data_source.get_sim_obs(idx=idx, space='map', field='temperature', spin=0)
            return np.array(obs*mask)

    
    def get_sim_pmap(self, idx):
        if self.sky_coverage == 'full':
            return self.data_source.get_sim_obs(idx=idx, space='map', field='polarization', spin=2)
        elif self.sky_coverage == 'masked':
            mask = np.load(self.mask_fn)
            ret = self.data_source.get_sim_obs(idx=idx, space='map', field='polarization', spin=2)
            return np.array([re*mask for re in ret])


    def get_data_(self, idx):
            # NOTE wrapper to access data that is both masked or unmasked, as data_source does not support masked data if generated.
            # If data is already masked, this will doubly mask the data.. not sure we want this 
            # FIXME remove hp and get nside from data_source
            import healpy as hp
            nside = 2048
            space = 'alm' if self.sky_coverage == 'full' else 'map'
            # lm_max_ = self.data_source.obs_lib.CMB_info['lm_max'] # NOTE using this gives an error in an operator action
            lm_max_ = self.lm_max_sky
            if space == 'alm':
                earr = np.zeros(shape=Alm.getsize(*lm_max_),dtype=complex)
            else:
                earr = np.zeros(hp.nside2npix(nside))
            if True: # NOTE trimmed data currently not supported
                pobs = self.data_source.get_sim_obs(idx, space='alm', spin=0, field='polarization')
                # pobs = alm_copy_nd(pobs, None, lm_max_)
                if self.data_key in ['p', 'eb', 'be']:
                    # ret = [earr, *alm_copy_nd(pobs, None, lm_max_)]
                    ret = [earr, pobs]
                    if space == 'map':
                        ret = [earr, *hp.alm2map_spin(ret[1:], nside=nside, spin=2, lmax=lm_max_[0], mmax=lm_max_[1])]
                elif self.data_key in ['ee']:
                    ret = [earr, alm_copy_nd(pobs, None, lm_max_)[0], earr]
                    if space == 'map':
                        assert 0, 'implement if needed'
                elif self.data_key in ['tt']:
                    ret = [alm_copy_nd(self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'), None, lm_max_), earr, earr]
                    if space == 'map':
                        ret = [*hp.alm2map(ret[0], nside=nside, spin=0), earr, earr]
                elif self.data_key in ['tp']:
                    Tobs = alm_copy_nd(self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'), None, lm_max_)   
                    QUobs = alm_copy_nd(pobs, None, lm_max_)
                    if space == 'map':
                        Tobs = hp.alm2map(Tobs, nside=nside)
                        QUobs = hp.alm2map_spin(QUobs, nside=nside, spin=2, lmax=lm_max_[0], mmax=lm_max_[1])
                    ret = [Tobs, *QUobs]
                else:
                    assert 0, 'implement if needed'
                return np.array(ret)
            else:
                if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                    return np.array(self.sims_MAP.get_sim_pmap(self.idx), dtype=float)
                else:
                    assert 0, 'implement if needed'


    def get_data(self, idx):
        # NOTE wrapper to access data that is both masked or unmasked, as data_source does not support masked data if generated.
        # If data is already masked, this will doubly mask the data.. not sure we want this 
        space = 'alm' if self.sky_coverage == 'full' else 'map'
        if space == 'alm':
            lm_max_ = self.lm_max_sky
            pobs = self.data_source.get_sim_obs(idx, space=space, spin=0, field='polarization')
            # earr = np.zeros(shape=pobs.shape[-1],dtype=complex)
            earr = np.zeros(shape=Alm.getsize(*lm_max_),dtype=complex)
            pobs = alm_copy_nd(pobs, None, lm_max_)
            if self.data_key in ['p', 'eb', 'be']:
                ret = [earr, *pobs]
            elif self.data_key in ['ee']:
                ret = [earr, alm_copy_nd(pobs, None, lm_max_)[0], earr]
                if space == 'map':
                    assert 0, 'implement if needed'
            elif self.data_key in ['tt']:
                ret = [alm_copy_nd(self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'), None, lm_max_), earr, earr]
            elif self.data_key in ['tp']:
                Tobs = alm_copy_nd(self.data_source.get_sim_obs(idx, space='alm', spin=0, field='temperature'), None, lm_max_)   
                pobs = alm_copy_nd(pobs, None, lm_max_)
                ret = [Tobs, *pobs]
            else:
                assert 0, 'implement if needed'
            return np.array(ret)
        
        elif space == 'map':
            # ret = [*hp.alm2map(ret[0], nside=nside, spin=0), earr, earr]
            # Tobs = hp.alm2map(Tobs, nside=nside)
            # QUobs = hp.alm2map_spin(QUobs, nside=nside, spin=2, lmax=lm_max_[0], mmax=lm_max_[1])
            # if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            if self.data_key in ['p', 'eb', 'be']:
                buff = np.array(self.data_source.get_sim_pmap(idx), dtype=float)
                ret = np.array([np.zeros_like(buff[0]), *buff])
                return ret
            elif self.data_key in ['ee']:
                # FIXME running on ee only means I need to get only E, but get_sim_pmap returns both Q and U, so "truncation" should actually happen somewhere else
                assert 0, "implement if needed"
                buff = np.array(self.data_source.get_sim_pmap(idx), dtype=float)
                ret = np.array([np.zeros_like(buff[0]), buff[0], np.zeros_like(buff[0])])
                return ret
            elif self.data_key in ['tt']:
                buff = np.array(self.data_source.get_sim_tmap(idx), dtype=float)
                ret = np.array([buff, np.zeros_like(buff), np.zeros_like(buff)])
                return ret 
            elif self.data_key in ['tp']:
                buff_p = np.array(self.data_source.get_sim_pmap(idx), dtype=float)
                buff_t = np.array(self.data_source.get_sim_tmap(idx), dtype=float)
                ret = np.array([buff_t, *buff_p])
                return ret


class QEScheduler:
    """Quadratic estimate lensing reconstruction Job. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """
    @check_MPI
    def __init__(self, QE_job_desc, QE_searchs_desc, data_container):
        # NOTE plancklens uses get_sim_pmap() from data_container.
        # DataContainer updates the data_container object with the libdirs and fns if it generated simulations, so need to update this
        self.data_container = data_container

        self.tasks = QE_job_desc['tasks']
        self.idxs = QE_job_desc['idxs']
        self.idxs_mf = QE_job_desc['idxs_mf']

        # NOTE I want to have a QE search for each field
        for QE_search_desc in QE_searchs_desc.values():
            QE_search_desc['idxs_mf'] = self.idxs_mf
            QE_search_desc['QE_filterqest_desc']['data_container'] = self.data_container
        self.QE_searchs = [QE_handler.Base(**QE_search_desc) for name, QE_search_desc in QE_searchs_desc.items()]

        self.secondary2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self.idx2secondary = {i: QE_search.secondary.ID for i, QE_search in enumerate(self.QE_searchs)}

        self.template_operator = QE_job_desc['template_operator'] # FIXME deal with this later
        
        # NOTE if there is no job in task "calc_fields", we can already init the filterqest
        if len(np.array([x for x in self.collect_jobs()[0].ravel() if x is not None]))==0:
            if len(self.data_container.collect_jobs()) == 0:
                for QE_search in self.QE_searchs:
                    QE_search.init_filterqest()


    def collect_jobs(self, recalc=False):
        jobs = list(range(len(self.tasks)))
        idxs_ = np.unique(np.concatenate([self.idxs, self.idxs_mf])).astype(int)
        for taski, task in enumerate(self.tasks):
            _jobs = []
            if task == 'calc_fields':
                _nomfcheck = self.idxs_mf.size == 0
                for idx in idxs_: # data indices
                    __jobs = []
                    _addindex = False
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search. # secondary indices
                        _addsecondary = False
                        for ci, component in enumerate(QE_search.secondary.component): # component indices
                            if _nomfcheck or not QE_search.secondary.cacher.is_cached(QE_search.secondary.qmflm_fns[component].format(idx=idx)) or recalc:
                                if not QE_search.secondary.is_cached(idx, component, 'qlm') or recalc:
                                #    print(idx, component, QE_search.secondary.klm_fns[component].format(idx=idx), QE_search.secondary.cacher.is_cached(QE_search.secondary.klm_fns[component].format(idx=idx)))
                                   _addsecondary = True
                                   _addindex = True
                        if _addsecondary: __jobs.append(idx)
                    if _addindex: _jobs.append(__jobs)
             
            if task == 'calc_meanfields':
                for idx in self.idxs:
                    _addindex = False
                    __jobs = []
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        _addsecondary = False
                        for ci, component in enumerate(QE_search.secondary.component): # each field has n components #
                            if not QE_search.secondary.is_cached(idx, component, 'kmflm') or recalc:
                                _addsecondary = True
                                _addindex = True
                                # for idxqlms in self.idxs_mf:
                                #     field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=idx) if idx != -1 else 'dat_%s.fits'%self.k)
                        if _addsecondary: __jobs.append(idx)
                    if _addindex: _jobs.append(__jobs)

            # TODO later. If i add combinatorics here across all operators, could add this to the collect list.
            if task == 'calc_templates':
                assert 0, "not yet implemented"
                for idx in self.idxs:
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        __jobs = []
                        for ci, component in enumerate(QE_search.secondary.components): # each field has n components # fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.idxs_mf)))
                            tepmplate_fn = opj(QE_search.libdir, 'templates', QE_search.template.qmflm_fns[component])
                            if not os.path.isfile(tepmplate_fn) or recalc:
                                field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=idx) if idx != -1 else 'dat_%s.fits'%self.k)
                                if not os.path.isfile(field_fn) or recalc:
                                    pass
                                    # jobs.append(np.array(_jobs,dtype=float))
            jobs[taski] = np.array(_jobs, dtype=int)
        self.jobs = jobs
        if mpi.rank==0: log.info(f"QE jobs: {jobs}")
        return jobs


    def run(self, task=None):
        ctx, isnew = get_computation_context()
        if True: # 'triggers calc_cinv'
            self.init_QEsearchs()
                   
        tasks = self.tasks if task is None else [task]
        # NOTE step 0 is making sure I run get_qlm() for all indices needed, before calculating mean-field or similar
        for taski, task in enumerate(tasks):
            if mpi.rank==0: log.info(f"Starting QE task {task}")
            if task == 'calc_fields':
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for seci, secidx in enumerate(idxs):
                        ctx.set(idx=secidx, idx2=secidx)
                        self.QE_searchs[seci].get_qlm(int(secidx))
                    if np.all(self.data_container.obs_lib.maps == DEFAULT_NotAValue):
                        self.data_container.data_source.purgecache()
                mpi.barrier()
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for seci, secidx in enumerate(idxs):
                        ctx.set(idx=secidx, idx2=secidx)
                        self.QE_searchs[seci].get_est(int(secidx))


            if task == 'calc_meanfields':
                if mpi.rank==0: log.info(f"Starting QE task {task}")
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    # for QE_search in self.QE_searchs:
                    #     for seci, secidx in enumerate(idxs):
                    #         if secidx is not None: #these Nones come from the field already being done.
                    #             ctx.set(idx=secidx, idx2=secidx)
                    #             self.QE_searchs[seci].get_qlm(int(secidx))
                    #             if secidx in self.idxs: # NOTE this should only run across the simidxs, not the union with mf idxs
                    #                 self.QE_searchs[seci].get_est(int(secidx)) # this is here for convenience
                    for QE_search in self.QE_searchs:
                        for ci, component in enumerate(QE_search.secondary.component):
                            ctx.set(idx=idxs[ci], idx2=idxs[ci])
                            qmf_lm = QE_search.get_qmflm(int(idxs[ci]), self.idxs_mf, component)
                            QE_search.secondary.cache_qmflm(qmf_lm, int(idxs[ci]), component=component)
                            kmf_lm = QE_search.get_kmflm(int(idxs[ci]), self.idxs_mf, component)
                            QE_search.secondary.cache_kmflm(kmf_lm, int(idxs[ci]), component=component)
                mpi.barrier()


            # TODO later
            if task == 'calc_templates':
                if mpi.rank==0: log.info(f"Starting QE task {task}")
                assert 0, 'implement if needed'
                for idxs in self.jobs[taski][mpi.rank::mpi.size]:
                    # For each combination of operators, I want to build templates
                    # jobs list could come as [idx-delta,idx-beta,idx-deltabeta] for each idx
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


    def get_template(self, idx, it=0, QE_perturbative=True, secondary=None, component=None, calc=False):
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
    MAP_minimizer: MAP_handler.Minimizer
    def __init__(self, idxs, idxs_mf, data_container, QE_searchs, tasks, MAP_minimizer):
        self.data_container = data_container

        self.idxs = idxs
        self.idxs_mf = idxs_mf
        self.QE_searchs: QEScheduler = QE_searchs

        self._sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self._seclist = list(self._sec2idx.keys())

        self.MAP_minimizer: MAP_handler.Minimizer = MAP_minimizer
        self.tasks = tasks

        # NOTE this conflicts with setting idx via ctx later during run. At init, context is not set yet, so copyQEtoDirectory will not work
        # for idx in self.idxs:
        #     if np.all([self.QE_searchs[0].isdone(idx, comp)==0 for comp in self.QE_searchs[0].secondary.component]):
        #         if mpi.rank == 0:
        #             self.MAP_minimizer.copyQEtoDirectory(QE_searchs)


    def collect_jobs(self):
        ctx, isnew = get_computation_context()
        jobs = list(range(len(self.tasks)))
        for taski, task in enumerate(self.tasks):
            _jobs = []
            if task == 'calc_fields':
                for idxi, idx in enumerate(self.idxs):
                    ctx.set(idx=idx, idx2=idx)
                    if self.MAP_minimizer.maxiterdone() < self.MAP_minimizer.itmax:
                        _jobs.append(idx)
                jobs[taski] = _jobs
        self.jobs = jobs
        return np.array(jobs, dtype=int)


    def run(self):
        ctx, isnew = get_computation_context()
        for taski, task in enumerate(self.tasks):
            log.info('MAPScheduler {}, MAP task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski][mpi.rank::mpi.size]))
            if task == 'calc_fields':
                for idx in self.jobs[taski][mpi.rank::mpi.size]: # NOTE every rank takes care of its own indices
                    if np.all([self.QE_searchs[0].isdone(idx, comp)==0 for comp in self.QE_searchs[0].secondary.component]):
                        ctx.set(idx=idx, idx2=idx)
                        self.MAP_minimizer.copyQEtoDirectory(self.QE_searchs)
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    ctx.set(idx=idx, idx2=idx)
                    self.MAP_minimizer.get_est(self.MAP_minimizer.itmax)

        #NOTE resetting context to first idx - for application level
        ctx.set(idx=min(self.idxs), idx2=min(self.idxs))


    def get_est(self, idx, it=None, secondary=None, component=None, scale='k', subtract_QE_meanfield=True, calc_flag=False, idx2=None):
        ctx, isnew = get_computation_context()
        ctx.set(idx=idx, idx2=idx)
        if isinstance(secondary, str) and secondary not in self._seclist:
            print('Secondary not found. Available secondaries are:', self._seclist)
            return np.array([[]])
        if it is None:
            it = self.MAP_minimizer.maxiterdone()

        for idx_ in self.idxs:
            ctx.set(idx=idx_, idx2=idx_)
            self.MAP_minimizer.copyQEtoDirectory(self.QE_searchs)
        ctx.set(idx=idx, idx2=idx)
        def get_map_est(it_):
            return self.MAP_minimizer.get_est(it_, secondary, component, scale, calc_flag)

        if isinstance(it, (list, np.ndarray)):
            it = np.array(it)
        return get_map_est(it)


    def get_qlm(self, idx, it, secondary=None, component=None, idx2=None):
        assert it==0, 'QLM only available for QE, set it=0'
        ctx, isnew = get_computation_context()
        ctx.set(idx=idx, idx2=idx)
        if secondary is None:
            return [self.QE_searchs[self._sec2idx[QE_search.ID]].get_qlm(idx, component) for QE_search in self.QE_searchs]
        return self.QE_searchs[self._sec2idx[secondary]].get_qlm(idx, component)


    def get_meanfield(self, idx, it=None, secondary=None, component=None, idx2=None):
        ctx, isnew = get_computation_context()
        ctx.set(idx=idx, idx2=idx)
        return self.get_gradient_meanfield(idx, it, secondary=None, component=None, idx2=None)


    def get_template(self, idx, it, QE_perturbative=True, secondary=None, component=None, idx2=None, order='reversed'):
        ctx, isnew = get_computation_context()
        ctx.set(idx=idx, idx2=idx2 or idx)
        return self.MAP_minimizer.get_template(it, QE_perturbative, secondary, component, order=order)


    def get_wflm(self, idx, it=None, lm_max=None, idx2=None):
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            return self.QE_searchs[0].get_wflm(idx, lm_max=lm_max)
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        stash = ctx.idx, ctx.idx2, ctx.component
        ctx.set(idx=idx)
        ret = self.MAP_minimizer.get_wflm(it)
        ctx.set(idx=stash[0], idx2=stash[1], component=stash[2])
        return ret


    def get_ivflm(self, idx, it=0, idx2=None):
        # NOTE currently no support for list of secondary or it
        if it==0:
            return self.QE_searchs[0].get_ivflm(idx)
        print('only available for QE, set it=0')


    def get_ivfreslm(self, idx, it=None, idx2=None):
        ctx, _ = get_computation_context()
        stash = ctx.idx, ctx.idx2, ctx.component
        ctx.set(idx=idx)
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            print('only available for MAP, set it>0')
        ret = self.MAP_minimizer.get_ivfreslm(it)
        ctx.set(idx=stash[0], idx2=stash[1], component=stash[2])
        return ret

    def maxiterdone(self):
        ctx, _ = get_computation_context()
        buff_ = ctx.idx, ctx.idx2
        buff = []
        for idx in self.idxs:
            ctx.set(idx=idx, idx2=idx)
            buff.append(self.MAP_minimizer.maxiterdone())
        ctx.set(idx=buff_[0], idx2=buff_[1])
        return min(buff)
    
    def get_gradient_quad(self, idx, it, secondary=None, component=None, idx2=None):
        ctx, _ = get_computation_context()
        ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizer.get_gradient_quad(it=it)
    
    def get_gradient_total(self, idx, it, secondary=None, component=None, idx2=None):
        ctx, _ = get_computation_context()
        ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizer.get_gradient_total(it=it)
    
    def get_gradient_prior(self, idx, it, secondary=None, component=None, idx2=None):
        ctx, _ = get_computation_context()
        ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizer.get_gradient_prior(it=it)
    
    def get_gradient_meanfield(self, idx, it, secondary=None, component=None, idx2=None):
        ctx, _ = get_computation_context()
        ctx.set(idx=idx, secondary=secondary, component=component, idx2=idx2)
        return self.MAP_minimizer.get_gradient_meanfield(it=it)
    

    def __getattr__(self, name):
        # Forward the method call to the minimizer
        def method_forwarder(*args, **kwargs):
            if name in functionforwardlist and hasattr(self.MAP_minimizer, name):
                return getattr(self.MAP_minimizer, name)(*args, **kwargs)
            else:
                raise AttributeError(f"method {name} not found in MAP_minimizer")

        return method_forwarder


class PhiAnalyser:
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
            os.makedirs(self.libdir_phianalayser, exist_ok=True)
        
        self.TEMP_WF = opj(self.libdir_phianalayser, 'WF')
        if not os.path.isdir(self.TEMP_WF):
            os.makedirs(self.TEMP_WF, exist_ok=True)
        self.TEMP_Cx = opj(self.libdir_phianalayser, 'Cx')
        if not os.path.isdir(self.TEMP_Cx):
            os.makedirs(self.TEMP_Cx, exist_ok=True)
        self.TEMP_Cxbias = opj(self.libdir_phianalayser, 'Cxb')
        if not os.path.isdir(self.TEMP_Cxbias):
            os.makedirs(self.TEMP_Cxbias, exist_ok=True)
        self.TEMP_Cccc = opj(self.libdir_phianalayser, 'Cccc')
        if not os.path.isdir(self.TEMP_Cccc):
            os.makedirs(self.TEMP_Cccc, exist_ok=True)

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
        

class OverwriteAnafast:
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
