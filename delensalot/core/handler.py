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

from delensalot.utils import cli
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
from delensalot.core.opfilt import utils_cinv_p as cinv_p_OBD
from delensalot.core.opfilt.opfilt_handler import QE_transformer, MAP_transformer
from delensalot.core.opfilt.bmodes_ninv import template_dense, template_bfilt

from delensalot.core.MAP import handler as MAP_handler
from delensalot.core.QE import handler as QE_handler

from delensalot.sims.sims_lib import dirname_generator

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
    # @log_on_start(logging.DEBUG, "collect_jobs() started")
    # @log_on_end(logging.DEBUG, "collect_jobs() finished")
    def collect_jobs(self):
        jobs = []
        if not os.path.isfile(opj(self.libdir,'tniti.npy')):
            # This fakes the collect/run structure, as bpl takes care of MPI 
            jobs = [0]  
        self.jobs = jobs
        return jobs


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "run() started")
    # @log_on_end(logging.DEBUG, "run() finished")
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


class Sim_generator:
    """Simulation generation Job. Generates simulations for the requested configuration.
        * If any libdir exists, then a flavour of data is provided. Therefore, can only check by making sure flavour == obs, and fns exist.
    """
    def __init__(self, delensalot_model):
        """ In this init we make the following checks:
         * (1) Does user provide obs data? Then Sim_generator can be fully skipped
         * (2) Otherwise, check if files already generated (delensalot model may not know this, so need to search),
           * If so, update the simhandler with the respective libdirs and fns
           * If not,
             * generate the simulations
             * update the simhandler
        """        
        self.__dict__.update(delensalot_model.__dict__)
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
                lenjob_geomstr = 'unknown_skygeometry'
            else:
                # some flavour provided, and we need to generate the sky and obs maps from this.
                hashc = get_hashcode([val['component'] for val in self.simulationdata.sec_info.values()])
                lenjob_geomstr = get_dirname(self.simulationdata.sky_lib.operator_info['lensing']['geominfo'])+"_"+hashc
                
                self.libdir_sky = opj(dirname_generator(self.simulationdata.libdir_suffix, self.simulationdata.geominfo), lenjob_geomstr)
                self.fns_sky = self.set_basename_sky()
                # NOTE for each operator, I need sec fns
                self.fns_sec = {}
                for sec, operator_info in self.simulationdata.operator_info.items():
                    self.fns_sec.update({sec:{}})
                    for comp in operator_info['component']:
                        self.fns_sec[sec][comp] = f'{sec}_{comp}lm_{{}}.npy'

            hashc = get_hashcode(str([val['component'] for val in self.simulationdata.sec_info.values()])+str([val['component'] for val in self.simulationdata.sec_info.values()]))
            nlev_round = dict2roundeddict(self.simulationdata.nlev)
            self.libdir = opj(dirname_generator(self.simulationdata.libdir_suffix, self.simulationdata.geominfo), lenjob_geomstr, get_dirname(sorted(nlev_round.items())), f'{hashc}')
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
            simidxs_ = np.unique(np.concatenate([self.simidxs, self.simidxs_mf])).astype(int)
            def check_and_log(libdir, fns, postrun_method, data_type):
                """function to check file existence """
                if all(os.path.exists(opj(libdir, fns[f].format(simidx))) for f in required_files for simidx in simidxs_):
                    postrun_method()
                    log.info(f'will use {data_type} data at {libdir} with filenames {fns}')
                else:
                    log.info(f'{data_type} data will be stored at {libdir} with filenames {fns}')

            check_and_log(self.libdir, self.fns, self.postrun_obs, "obs")
            if self.simulationdata.flavour != 'sky':
                if all(os.path.exists(opj(self.libdir_sky, self.fns_sec[sec][component].format(simidx))) for sec in self.fns_sec.keys() for component in self.fns_sec[sec] for simidx in simidxs_):
                    check_and_log(self.libdir_sky, self.fns_sky, self.postrun_sky, "sky")
                else:
                    log.info(f'sky data will be stored at {self.libdir_sky} with filenames {self.fns_sky}. All secondaries will be generated along the way')


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
        if np.all(self.simulationdata.maps == DEFAULT_NotAValue) and self.simulationdata.flavour != 'obs':
            simidxs_ = np.unique(np.concatenate([self.simidxs, self.simidxs_mf])).astype(int)
            for taski, task in enumerate(['generate_sky', 'generate_obs']):
                _jobs = []

                if task == 'generate_sky':
                    for simidx in simidxs_:
                        missing_files = any(not os.path.isfile(opj(self.libdir_sky, self.fns_sky[f].format(simidx))) for f in required_files) or any(not os.path.exists(opj(self.libdir_sky, fnsec[comp].format(simidx))) for fnsec in self.fns_sec.values() for comp in fnsec.keys())
                        if missing_files:
                            _jobs.append(simidx)
                
                if task == 'generate_obs':
                    for simidx in simidxs_:
                        missing_files = any(not os.path.isfile(opj(self.libdir, self.fns[f].format(simidx))) for f in required_files)
                        if missing_files:
                            _jobs.append(simidx)  

                jobs[taski] = _jobs
            self.jobs = jobs
        else:
            self.jobs = [[],[]]
        return self.jobs

    # @log_on_start(logging.DEBUG, "Sim.run() started")
    # @log_on_end(logging.DEBUG, "Sim.run() finished")
    def run(self):
        for taski, task in enumerate(['generate_sky', 'generate_obs']):
            for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                log.info(f"rank {mpi.rank} (size {mpi.size}) {task} sim {simidx}")
                if task == 'generate_sky':
                    self.generate_sky(simidx)
                if task == 'generate_obs':
                    self.generate_obs(simidx)
                if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                    self.simulationdata.purgecache()
        if np.all(self.simulationdata.maps == DEFAULT_NotAValue):
            self.postrun_sky()
            self.postrun_obs()


    # @log_on_start(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) started")
    # @log_on_end(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) finished")
    def generate_sky(self, simidx):
        for sec, secinfo in self.simulationdata.sec_info.items():
            # FIXME if there is cross-correlation between the components, we need to generate them together
            for comp in secinfo['component']:
                if not os.path.exists(opj(self.libdir_sky, self.fns_sec[sec][comp].format(simidx))):
                    s = self.simulationdata.get_sim_sec(simidx, space='alm', secondary=sec, component=comp)
                    np.save(opj(self.libdir_sky, self.fns_sec[sec][comp].format(simidx)), s)

        for field in required_files_map.get(self.k, []):
            filepath = opj(self.libdir_sky, self.fns_sky[field].format(simidx))
            if not os.path.exists(filepath):
                if field in ['E', 'B']:
                    EBsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='polarization')
                    np.save(opj(self.libdir_sky, self.fns_sky['E'].format(simidx)), EBsky[0])
                    np.save(opj(self.libdir_sky, self.fns_sky['B'].format(simidx)), EBsky[1])
                    break
                if field == 'T':
                    Tsky = self.simulationdata.get_sim_sky(simidx, spin=0, space='alm', field='temperature')
                    np.save(filepath, Tsky)


    # @log_on_start(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) started")
    # @log_on_end(logging.DEBUG, "Sim.generate_sim(simidx={simidx}) finished")
    def generate_obs(self, simidx):
        for field in required_files_map.get(self.k, []):
            filepath = opj(self.libdir, self.fns[field].format(simidx))  
            if not os.path.exists(filepath):
                if field in ['E', 'B']:
                    EBobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='polarization')
                    np.save(opj(self.libdir, self.fns['E'].format(simidx)), EBobs[0])
                    np.save(opj(self.libdir, self.fns['B'].format(simidx)), EBobs[1])
                    break

                if field == 'T':
                    Tobs = self.simulationdata.get_sim_obs(simidx, spin=0, space='alm', field='temperature')
                    np.save(filepath, Tobs)


    def postrun_obs(self):
        # NOTE if this class here decides to generate data, we need to update some parameters in the simulationdata object
        # NOTE if later reconstruction is run with the same config file, these updates also make sure they find the data without having to update the config file
        if self.simulationdata.flavour != 'sky' and self.simulationdata.flavour != 'obs' and np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
            self.simulationdata.libdir = self.libdir
            self.simulationdata.fns = self.fns
            if self.simulationdata.flavour != 'obs':
                self.simulationdata.obs_lib.CMB_info['fns'] = self.fns
                self.simulationdata.obs_lib.CMB_info['libdir'] = self.libdir
                self.simulationdata.obs_lib.CMB_info['space'] = 'alm'
                self.simulationdata.obs_lib.CMB_info['spin'] = 0


    def postrun_sky(self):
        # NOTE if this class here decides to generate data, we need to update some parameters in the simulationdata object
        # NOTE if later reconstruction is run with the same config file, these updates also make sure they find the data without having to update the config file
        

        if not self.simulationdata.flavour in ['sky', 'obs'] and np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
            self.simulationdata.sky_lib.CMB_info['fns'] = self.fns_sky
            self.simulationdata.sky_lib.CMB_info['libdir'] = self.libdir_sky
            self.simulationdata.sky_lib.CMB_info['space'] = 'alm'
            self.simulationdata.sky_lib.CMB_info['spin'] = 0


        # NOTE for pri_lib we set the paths to the generated secondaries
        for sec, secinfo in self.simulationdata.operator_info.items():
            self.simulationdata.pri_lib.sec_info[sec]['fn'] = self.fns_sec[sec]
            self.simulationdata.pri_lib.sec_info[sec]['libdir'] = self.libdir_sky
            self.simulationdata.pri_lib.sec_info[sec]['space'] = 'alm'
            self.simulationdata.pri_lib.sec_info[sec]['spin'] = 0
            self.simulationdata.pri_lib.sec_info[sec]['lm_max'] = secinfo['lm_max']
            self.simulationdata.pri_lib.sec_info[sec]['component'] = secinfo['component']


class Noise_modeller(Basejob):
    '''
    CURRENTLY NOT USED
    '''
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
            # FIXME not sure which nivjob_geomlib to pass here, restricted or not?
            # nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
            self.cinv_p = cinv_p_OBD.cinv_p(opj(self.libdir_QE, 'cinv_p'),
                self.lm_max_ivf[0], self.nivjob_geominfo[1]['nside'], self.cls_len,
                transf_elm_loc[:self.lm_max_ivf[0]+1], self.nivp_desc, geom=self.nivjob_geomlib, #self.nivjob_geomlib,
                chain_descr=self.chain_descr(self.lm_max_ivf[0], self.cg_tol), bmarg_lmax=self.lmin_teb[2],
                zbounds=self.zbounds, _bmarg_lib_dir=self.obd_libdir, _bmarg_rescal=self.obd_rescale,
                sht_threads=self.tr)
            # (-1,1)
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
        

class QE_lr_v2:
    """Quadratic estimate lensing reconstruction Job. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """
    @check_MPI
    def __init__(self, dm):
        self.__dict__.update(dm.__dict__)
        # NOTE plancklens uses get_sim_pmap() from simulationdata.
        # Sim_generator updates the simulationdata object with the libdirs and fns if it generated simulations, so need to update this
        self.simgen = Sim_generator(dm)
        self.simulationdata = self.simgen.simulationdata

        self.QE_tasks = dm.QE_job_desc['QE_tasks']
        self.simidxs = dm.QE_job_desc['simidxs']
        self.simidxs_mf = dm.QE_job_desc['simidxs_mf']

        # I want to have a QE search for each field
        for QE_search_desc in dm.QE_searchs_desc.values():
            QE_search_desc['simidxs_mf'] = self.simidxs_mf
            QE_search_desc['QE_filterqest_desc']['simulationdata'] = self.simulationdata
        self.QE_searchs = [QE_handler.base(**QE_search_desc) for name, QE_search_desc in dm.QE_searchs_desc.items()]

        self.secondary2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self.idx2secondary = {i: QE_search.secondary.ID for i, QE_search in enumerate(self.QE_searchs)}

        self.template_operator = dm.QE_job_desc['template_operator'] # FIXME deal with this later
        
        # if there is no job, we can already init the filterqest
        if len(np.array([x for x in self.collect_jobs().ravel() if x is not None]))==0:
            for QE_search in self.QE_searchs:
                QE_search.init_filterqest()


    def collect_jobs(self, recalc=False):
        jobs = list(range(len(self.QE_tasks)))
        for taski, task in enumerate(self.QE_tasks):
            _jobs = []
            if task == 'calc_fields':
                _nomfcheck = True if self.simidxs_mf == [] else False
                for simidx in self.simidxs:
                    __jobs = []
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        _add = False
                        for ci, component in enumerate(QE_search.secondary.component):
                            if _nomfcheck or not QE_search.secondary.cacher.is_cached(QE_search.secondary.qmflm_fns[component].format(idx=simidx)) or recalc:
                                if not QE_search.secondary.cacher.is_cached(QE_search.secondary.klm_fns[component].format(idx=simidx)) or recalc:
                                   _add = True
                        __jobs.append(simidx) if _add else __jobs.append(None)
                    _jobs.append(__jobs)
            jobs.append(_jobs)
             
            if task == 'calc_meanfields':
                for simidx in self.simidxs_mf:
                    __jobs = []
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        _add = False
                        for ci, component in enumerate(QE_search.secondary.component): # each field has n components # fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.simidxs_mf)))
                            mf_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qmflm_fns[component])
                            if not os.path.isfile(mf_fn) or recalc:
                                field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                                if not os.path.isfile(field_fn) or recalc:
                                    _add = True
                         # checking for each component, but only adding the complete field as task index
                        __jobs.append(simidx) if _add else __jobs.append(None)
                    _jobs.append(__jobs)
                jobs.append(_jobs)

            # TODO later. If i add combinatorics here across all operators, could add this to the collect list.
            if task == 'calc_templates':
                for simidx in self.simidxs:
                    for Qi, QE_search in enumerate(self.QE_searchs): # each field has its own QE_search
                        __jobs = []
                        for ci, component in enumerate(QE_search.te.components): # each field has n components # fn_mf = opj(self.libdir_QE, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, pl_utils.mchash(self.simidxs_mf)))
                            tepmplate_fn = opj(QE_search.libdir, 'templates', QE_search.template.qmflm_fns[component])
                            if not os.path.isfile(tepmplate_fn) or recalc:
                                field_fn = opj(QE_search.libdir, 'qlms_dd', QE_search.secondary.qlm_fns[component].format(idx=simidx) if simidx != -1 else 'dat_%s.fits'%self.k)
                                if not os.path.isfile(field_fn) or recalc:
                                    _jobs.append(int(simidx))
            jobs[taski] = _jobs
        self.jobs = jobs
        if not np.all(np.array(jobs)==None):
            print("QE jobs: ", jobs)
        return np.array(jobs)


    def run(self, task=None):
        if not np.all(np.array(self.jobs)==None):
            print("Running QE jobs: ", self.jobs)
        if True: # 'triggers calc_cinv'
            self.init_QEsearchs()
                   
        _tasks = self.QE_tasks if task is None else [task]
        for taski, task in enumerate(_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_fields':
                for simidxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for seci, secidx in enumerate(simidxs):
                        if secidx is not None: #these Nones come from the field already being done.
                            self.QE_searchs[seci].get_qlm(int(secidx))
                            self.QE_searchs[seci].get_est(int(secidx)) # this is here for convenience
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()
                mpi.barrier()

            if task == 'calc_meanfields':
                for simidxs in self.jobs[taski][mpi.rank::mpi.size]:
                    for QE_search in self.QE_searchs:
                        for seci, secidx in enumerate(simidxs):
                            self.QE_searchs[seci].get_qlm(int(secidx))
                            self.QE_searchs[seci].get_est(int(secidx)) # this is here for convenience
                for QE_search in self.QE_searchs:
                    QE_search.get_qmflm(QE_search.estimator_key, self.simidxs_mf)
                mpi.barrier()

            # TODO later
            if task == 'calc_templates':
                for simidxs in self.jobs[taski][mpi.rank::mpi.size]:
                    # For each combination of operators, I want to build templates
                    # jobs list could come as [simidx-delta,simidx-beta,simidx-deltabeta] for each simidx
                    simidx, operator_indexs = simidxs.split('-')
                    self.get_template(simidx, operator_indexs)
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()


    def get_qlm(self, simidx, it=0, secondary=None, component=None):
        assert it == 0, 'QE does not have iterations, leave blank or set it=0'
        if secondary not in self.secondary2idx:
            print(f'secondary {secondary} not found. Available secondaries are: ', self.secondary2idx.keys())
            return np.array([[]])
        return self.QE_searchs[self.secondary2idx[secondary]].get_qlm(simidx, component)
    

    def get_est(self, simidx, it=0, secondary=None, component=None, subtract_meanfield=None, scale='k'):
        self.init_QEsearchs()
        if isinstance(it, (int,np.int64)):
            assert it == 0, 'QE does not have iterations, leave blank or set it=0'
        else:
            assert 0 in it, 'QE does not have iterations, leave blank or set it=0, not {}'
            return [self.get_est(simidx, 0, secondary, component, subtract_meanfield, scale)]
        if secondary is None:
            return [self.QE_searchs[secidx].get_est(simidx, component, subtract_meanfield, scale=scale) for secidx in self.secondary2idx.values()]
        if isinstance(secondary, list):
            return [self.QE_searchs[self.secondary2idx[sec]].get_est(simidx, component, subtract_meanfield, scale=scale) for sec in secondary]
        if secondary not in self.secondary2idx:
            print('secondary not found. Available secondaries are: ', self.secondary2idx.keys())
            return np.array([[]])
        return self.QE_searchs[self.secondary2idx[secondary]].get_est(simidx, component, subtract_meanfield, scale=scale)


    def get_template(self, simidx, it=0, secondary=None, component=None, calc=False):
        assert it==0, 'QE does not have iterations, leave blank or set it=0'
        path = opj(self.QE_searchs[0].fq.libdir, 'template', f"template_sim{simidx}_it{it}")
        if not os.path.isfile(path):
            if not self.QE_searchs[self.secondary2idx[secondary]].is_cached(self, simidx, component, type='qlm'):
                if not calc:
                    print(f'cannot generate template as estimate of secondary {secondary} with simidx {simidx} not found, set calc=True to calculate')
                    return np.array([[]])
                self.get_est(simidx, it, secondary, component)
            self.template_operator.set_field(simidx, it)
            estCMB = self.get_wflm(simidx, it)
            np.save(path, self.template_operator.act(estCMB))
        return self.template_operator.act(estCMB)


    def get_wflm(self, simidx, it, lm_max):
        if it!=0:
            print('QE does not have iterations, leave blank or set it=0')
            return np.array([[]])
        return self.QE_searchs[0].get_wflm(simidx, lm_max)


    def get_ivflm(self, simidx, it, lm_max):
        if it!=0:
            print('QE does not have iterations, leave blank or set it=0')
            return np.array([[]])
        return self.QE_searchs[0].get_ivflm(simidx, lm_max)
    

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


    def maxiterdone(self, simidx):
        return self.QE_searchs[0].isdone(simidx)


class MAP_lr_v2:
    def __init__(self, dm):
        self.__dict__.update(dm.__dict__)
        self.simgen = Sim_generator(dm)
        self.simulationdata = self.simgen.simulationdata

        self.simidxs = dm.MAP_job_desc["simidxs"]
        self.simidxs_mf = dm.MAP_job_desc["simidxs_mf"]
        self.QE_searchs = dm.MAP_job_desc["QE_searchs"]

        self.sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(self.QE_searchs)}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))
        
        # self.simulationdata = self.MAP_job_desc["simulationdata"]

        self.MAP_searchs_desc = dm.MAP_searchs_desc
        self.MAP_searchs_desc["curvature_desc"].update({"Runl0": {}})
        for i, QE_search in enumerate(self.QE_searchs):
            scale = 'k' if QE_search.secondary.ID in ['lensing'] else 'p' #NOTE Plancklens by default returns p scale (for lensing). Delensalot works with convergence
            self.MAP_searchs_desc["curvature_desc"]["Runl0"].update({QE_search.secondary.ID: {component: self.__rescale(QE_search.get_response_unl(component), scale=scale) for component in QE_search.secondary.component}})
        self.MAP_searchs = [MAP_handler.base(self.simulationdata, **self.MAP_searchs_desc, simidx=simidx) for simidx in self.simidxs]

        self.it_tasks = self.MAP_job_desc["it_tasks"]
        for simidx in self.simidxs:
            if self.QE_searchs[0].isdone(simidx, 'p') == 0:
                if mpi.rank == 0:
                    self.__copyQEtoDirectory(simidx)


    def collect_jobs(self):
        jobs = list(range(len(self.it_tasks)))
        for taski, task in enumerate(self.it_tasks):
            _jobs = []
            if task == 'calc_fields':
                for simidxi, simidx in enumerate(self.simidxs):
                    if self.MAP_searchs[simidxi].maxiterdone() < self.MAP_searchs[simidxi].itmax:
                        _jobs.append(simidx)
                jobs[taski] = _jobs
        self.jobs = jobs
        return jobs


    def run(self):
        for simidx in self.simidxs:
            if self.QE_searchs[0].isdone(simidx, 'p') == 0:
                if mpi.rank == 0:
                    self.__copyQEtoDirectory(simidx)

        for taski, task in enumerate(self.it_tasks):
            log.info('{}, MAP task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
            if task == 'calc_fields':
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.MAP_searchs[simidx].get_est(simidx, self.MAP_searchs[simidx].itmax)


    def get_est(self, simidx, it=None, secondary=None, component=None, scale='k', subtract_QE_meanfield=True, calc_flag=False):
        for simidx in self.simidxs:
            self.__copyQEtoDirectory(simidx)
        if it is None:
            it = self.MAP_searchs[simidx].maxiterdone()
        if isinstance(secondary, str) and secondary not in self.seclist_sorted:
            print('Secondary not found. Available secondaries are:', self.seclist_sorted)
            return np.array([[]])
        secondary_ = self.seclist_sorted if secondary is None else secondary

        def get_qe_est():
            if isinstance(secondary_, (list, np.ndarray)):
                return [self.QE_searchs[self.sec2idx[sec]].get_est(simidx, scale, subtract_QE_meanfield, component) for sec in secondary_]
            return self.QE_searchs[self.sec2idx[secondary_]].get_est(simidx, scale, subtract_QE_meanfield, component)

        def get_map_est(it_):
            return self.MAP_searchs[simidx].get_est(simidx, it_, secondary, component, scale, calc_flag)

        if isinstance(it, (list, np.ndarray)):
            # if 0 in it:
            #     return [get_qe_est()] + (get_map_est(sorted(it)[1:]) if len(it) > 1 else [])
            return get_map_est(np.array(it))
        return get_map_est(it)
        # return get_qe_est() if it == 0 else get_map_est(it)


    def get_qlm(self, simidx, it, secondary=None, component=None):
        if secondary is None:
            return [self.QE_searchs[self.sec2idx[QE_search.ID]].get_qlm(simidx, component) for QE_search in self.QE_searchs]
        if it==0:
            return self.QE_searchs[self.sec2idx[secondary]].get_qlm(simidx, component)
        print('only available for QE, set it=0')


    def get_template(self, simidx, it, secondary=None, component=None):
        assert it>0, 'Need to correctly implement QE template generation first'
        assert it >= self.maxiterdone(), 'Requested iteration is not available'
        return self.MAP_searchs[simidx].get_template(it, secondary, component)


    def get_gradient_quad(self, simidx, it=None, secondary=None, component=None):
        if isinstance(it, (list, np.ndarray)) and any(np.array(it)>self.maxiterdone()):
            it = it[it<=self.maxiterdone()]
            print('items in param "it" too big. maxiterdone() = ', self.maxiterdone())
        elif isinstance(it, (int,np.int64)) and it>self.maxiterdone():
            it = self.maxiterdone()
            print(' param "it" too big. maxiterdone() = ', self.maxiterdone())
        if (isinstance(it, (list, np.ndarray)) and 0 not in it):
            return [self.MAP_searchs[simidx].get_gradient_quad(it_, secondary, component) for it_ in it]
        elif isinstance(it, (int,np.int64)) and it > 0:
            self.MAP_searchs[simidx].get_gradient_quad(it, secondary, component)
        print('only available for MAP, set it>0')


    def get_gradient_meanfield(self, simidx, it=None, secondary=None, component=None):
        if isinstance(it, (list, np.ndarray)) and any(np.array(it)>self.maxiterdone()):
            it = it[it<=self.maxiterdone()]
            print('items in param "it" too big. maxiterdone() = ', self.maxiterdone())
        elif isinstance(it, (int,np.int64)) and it>self.maxiterdone():
            it = self.maxiterdone()
        if (isinstance(it, (list, np.ndarray)) and 0 not in it):
            return [self.MAP_searchs[simidx].get_gradient_meanfield(it_, secondary, component) for it_ in it]
        elif (isinstance(it, (list, np.ndarray)) and 0 in it):
            return [self.QE_searchs[self.sec2idx[secondary]].get_kmflm(simidx, component)] + [self.MAP_searchs[simidx].get_gradient_meanfield(it_, secondary, component) for it_ in it[1:]]
        elif isinstance(it, (int,np.int64)) and it > 0:
            self.MAP_searchs[simidx].get_gradient_meanfield(it, secondary, component)


    def get_gradient_prior(self, simidx, it=None, secondary=None, component=None):
        if isinstance(it, (list, np.ndarray)) and any(np.array(it)>self.maxiterdone()):
            it = it[it<=self.maxiterdone()]
            print('items in param "it" too big. maxiterdone() = ', self.maxiterdone())
        elif isinstance(it, (int,np.int64)) and it>self.maxiterdone():
            it = self.maxiterdone()
            print(' param "it" too big. maxiterdone() = ', self.maxiterdone())
        if isinstance(it, (list, np.ndarray)):
            return [self.MAP_searchs[simidx].get_gradient_prior(it_, secondary, component) for it_ in it]
        elif isinstance(it, (int,np.int64)):
            self.MAP_searchs[simidx].get_gradient_prior(it, secondary, component)


    def get_gradient_total(self, simidx, it=None, secondary=None, component=None):
        if isinstance(it, (list, np.ndarray)) and any(np.array(it)>self.maxiterdone()):
            it = it[it<=self.maxiterdone()]
            print('items in param "it" too big. maxiterdone() = ', self.maxiterdone())
        elif isinstance(it, (int,np.int64)) and it>self.maxiterdone():
            it = self.maxiterdone()
            print(' param "it" too big. maxiterdone() = ', self.maxiterdone())
        if (isinstance(it, (list, np.ndarray)) and 0 not in it):
            return [self.MAP_searchs[simidx].get_gradient_total(it_, secondary, component) for it_ in it]
        elif isinstance(it, (int,np.int64)) and it > 0:
            self.MAP_searchs[simidx].get_gradient_total(it, secondary, component)
        print('only available for MAP, set it>0')


    def get_wflm(self, simidx, it=None, lm_max=None):
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            return self.QE_searchs[0].get_wflm(simidx, lm_max)
        return self.MAP_searchs[simidx].get_wflm(simidx, it)


    def get_ivflm(self, simidx, it=0, lm_max=None): 
        # NOTE currently no support for list of secondary or it
        if it==0:
            return self.QE_searchs[0].get_ivflm(simidx, lm_max)
        print('only available for QE, set it=0')


    def get_ivfreslm(self, simidx, it=None):
        # NOTE currently no support for list of secondary or it
        if it==None: it = self.maxiterdone()
        if it==0:
            print('only available for MAP, set it>0')
        return self.MAP_searchs[simidx].get_ivfreslm(simidx, it)
    

    def __copyQEtoDirectory(self, simidx):
        # copies fields and gradient starting points to MAP directory
        # NOTE this turns them into convergence fields
        for secname, secondary in self.MAP_searchs[simidx].secondaries.items():
            self.QE_searchs[self.sec2idx[secname]].init_filterqest()
            if not all(self.MAP_searchs[simidx].secondaries[secname].is_cached(simidx, it=0)):
                klm_QE = self.QE_searchs[self.sec2idx[secname]].get_est(simidx)
                self.MAP_searchs[simidx].secondaries[secname].cache_klm(klm_QE, simidx, it=0)
            
            if not self.MAP_searchs[simidx].gradients[self.sec2idx[secname]].gfield.is_cached(simidx, it=0):
                kmflm_QE = self.QE_searchs[self.sec2idx[secname]].get_kmflm(simidx)
                self.MAP_searchs[simidx].gradients[self.sec2idx[secname]].gfield.cache_meanfield(kmflm_QE, simidx, it=0)

            #TODO cache QE wflm into the filter directory
            if not self.MAP_searchs[simidx].wf_filter.wf_field.is_cached(simidx, it=0):
                wflm_QE = self.QE_searchs[self.sec2idx[secname]].get_wflm(simidx, self.MAP_searchs[simidx].ivf_filter.lm_max_pri)
                self.MAP_searchs[simidx].wf_filter.wf_field.cache_field(np.array(wflm_QE), simidx, it=0)


    def maxiterdone(self):
        return min([MAP_search.maxiterdone() for MAP_search in self.MAP_searchs])
    

    def __rescale(self, obj, scale):
        lmax = len(obj)-1
        if scale == 'p':
            return obj
        elif scale == 'k':
            return obj * cli(0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float))**2
        else:
            print(f"Unknown scale {scale}")


class QE_lr(Basejob):
    """Quadratic estimate lensing reconstruction Job. Performs tasks such as lensing reconstruction, mean-field calculation, and B-lensing template calculation.
    """
    @check_MPI
    def __init__(self, dlensalot_model, caller=None):
        if caller is not None:
            dlensalot_model.QE_tasks = dlensalot_model.it_tasks
            ## TODO. Current solution to fake an iteration handler for QE to calc blt is to initialize one MAP_job here.
            ## In the future, I want to remove get_template_blm from the iteration_handler for QE.
            if 'calc_blt' in dlensalot_model.QE_tasks:
                self.MAP_job = caller

        super().__init__(dlensalot_model)
        if not os.path.exists(self.libdir_QE):
            os.makedirs(self.libdir_QE)
        self.libdir_MAP = lambda qe_key, simidx, version: opj(self.TEMP, 'MAP/%s'%(qe_key), 'sim%04d'%(simidx) + version)
        self.libdir_blt = lambda simidx: opj(self.TEMP, 'MAP/%s'%(self.k), 'sim%04d'%(simidx) + self.version, 'BLT/')
        for simidx in np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int):
            # calculates all plms even for mf indices. This is not necessarily requested due to potentially simidxs =/= simidxs_mf, but otherwise collect and run must be adapted and its ok like this.
            libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            if not os.path.exists(libdir_MAPidx):
                os.makedirs(libdir_MAPidx)
            if not os.path.exists(self.libdir_blt(simidx)):
                os.makedirs(self.libdir_blt(simidx))

        self.dlensalot_model = dlensalot_model
        
        self.simgen = Sim_generator(dlensalot_model)
        self.simulationdata = self.simgen.simulationdata

        
        if self.qe_filter_directional == 'isotropic':
            self.ivfs = filt_simple.library_fullsky_sepTP(opj(self.libdir_QE, 'ivfs'), self.simulationdata, self.nivjob_geominfo[1]['nside'], self.ttebl, self.cls_len, self.ftebl_len['t'], self.ftebl_len['e'], self.ftebl_len['b'], cache=True)
            if self.qlm_type == 'sepTP':
                self.qlms_dd = qest.library_sepTP(opj(self.libdir_QE, 'qlms_dd'), self.ivfs, self.ivfs, self.cls_len['te'], self.nivjob_geominfo[1]['nside'], lmax_qlm=self.lm_max_qlm[0])
        elif self.qe_filter_directional == 'anisotropic':
            ## Wait for at least one finished run(), as plancklens triggers cinv_calc...
            if len(self.collect_jobs()[0]) - len(self.simidxs) > 0 or len(self.collect_jobs()[0])==0:
                self.init_aniso_filter()

        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.R_unl = lambda: qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.lm_max_qlm[0])[0]

        ## Faking here sims_MAP for calc_blt as iteration_handler needs it
        if 'calc_blt' in self.QE_tasks:
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
    # @log_on_start(logging.DEBUG, "QE.collect_jobs(recalc={recalc}) started")
    # @log_on_end(logging.DEBUG, "QE.collect_jobs(recalc={recalc}) finished: jobs={self.jobs}")
    def collect_jobs(self, recalc=False):

        # QE_tasks overwrites task-list and is needed if MAP lensrec calls QE lensrec
        jobs = list(range(len(self.QE_tasks)))
        for taski, task in enumerate(self.QE_tasks):
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
                    if not os.path.isfile(fn_blt) or True:
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
    # @log_on_start(logging.DEBUG, "QE.run(task={task}) started")
    # @log_on_end(logging.DEBUG, "QE.run(task={task}) finished")
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
                        
        _tasks = self.QE_tasks if task is None else [task]
        for taski, task in enumerate(_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_phi':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    log.info("get_sim_qlm..")
                    self.qlms_dd.get_sim_qlm(self.k, int(idx))
                    log.info("get_sim_qlm done.")
                    if np.all(self.simulationdata.obs_lib.maps == DEFAULT_NotAValue):
                        self.simulationdata.purgecache()
                mpi.barrier()
                
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    ## If meanfield subtraction is requested, only one task must calculate the meanfield first before get_plm() is called, otherwise read-errors because all tasks try calculating/accessing it at once.
                    ## The way I fix this (the next two lines) is a bit unclean.
                    if self.QE_subtract_meanfield:
                        self.qlms_dd.get_sim_qlm_mf(self.k, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
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
    # @log_on_start(logging.DEBUG, "QE.get_sim_qlm(simidx={simidx}) started")
    # @log_on_end(logging.DEBUG, "QE.get_sim_qlm(simidx={simidx}) finished")
    def get_sim_qlm(self, simidx):

        return self.qlms_dd.get_sim_qlm(self.k, int(simidx))


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_wflm(simidx={simidx}) started")
    # @log_on_end(logging.DEBUG, "QE.get_wflm(simidx={simidx}) finished")    
    def get_wflm(self, simidx):
        if self.k in ['ptt']:
            return lambda: alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
            return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        elif self.k in ['p']:
            return lambda: np.array([alm_copy(self.ivfs.get_sim_tmliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1]), alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])])


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_R_unl() started")
    # @log_on_end(logging.DEBUG, "QE.get_R_unl() finished")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.lm_max_qlm[0])[0]


    @log_on_start(logging.DEBUG, "QE.get_meanfield(simidx={simidx}) started")
    @log_on_end(logging.DEBUG, "QE.get_meanfield(simidx={simidx}) finished")
    def get_meanfield(self, simidx):
        # Either return MC MF, filter.qlms_mf, or mfvar
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, simidx))
        fn_mf = opj(self.libdir_QE, 'mf_allsims.npy')
        if type(self.mfvar) == str:
            if self.mfvar == 'qlms_mf':
                # calculate MF estimate using Lewis&Carron trick
                mchain = self.get_mchain(0, 'p')
                return self.filter.get_qlms_mf(1, self.ffi.pbgeom, mchain)
        if self.Nmf > 1:
            if self.mfvar == None:
                # MC MF, and exclude the current simidx
                ret = self.qlms_dd.get_sim_qlm_mf(self.k, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
                np.save(fn_mf, ret) # plancklens already stores that in qlms_dd/ but I want to have this more conveniently without the naming gibberish
                if simidx in self.simidxs_mf:    
                    ret = (ret - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
            else:

                # Take raw meanfield provided by user
                # TODO could do a normalization check here
                ret = np.load(self.mfvar)
                log.info('returning mfvar meanfield')
            return ret
            
        return ret
        

    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_plm_n1(simidx={simidx}, sub_mf={sub_mf}) started")
    # @log_on_end(logging.DEBUG, "QE.get_plm_n1(simidx={simidx}, sub_mf={sub_mf}) finished")
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


    # @log_on_start(logging.DEBUG, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) started")
    # @log_on_end(logging.DEBUG, "QE.get_plm(simidx={simidx}, sub_mf={sub_mf}) finished")
    def get_plm(self, simidx, component=None, sub_mf=True):
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


    # @log_on_start(logging.DEBUG, "QE.get_response_meanfield() started")
    # @log_on_end(logging.DEBUG, "QE.get_response_meanfield() finished")
    def get_response_meanfield(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.lm_max_qlm[0])[0]
        else:
            log.info('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lm_max_qlm[0] + 1, dtype=float)

        return mf_resp

    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_meanfield_normalized(simidx={simidx}) started")
    # @log_on_end(logging.DEBUG, "QE.get_meanfield_normalized(simidx={simidx}) finished")
    def get_meanfield_normalized(self, simidx):
        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        R = qresp.get_response(self.k, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.lm_max_qlm[0])[0]
        WF = self.cpp * pl_utils.cli(self.cpp + pl_utils.cli(R))
        almxfl(mf_QE, pl_utils.cli(R), self.lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, self.cpp > 0, self.lm_max_qlm[1], True)

        return mf_QE


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_blt({simidx}) started")
    # @log_on_end(logging.DEBUG, "QE.get_blt({simidx}) finished")
    def get_blt(self, simidx):
        fn_blt = opj(self.libdir_blt(simidx), 'blt_%s_%04d_p%03d_e%03d_lmax%s'%(self.k, simidx, 0, 0, self.lm_max_blt[0]) + 'perturbative' * self.blt_pert + '.npy')
        if not os.path.exists(fn_blt):
            ## For QE, dlm_mod by construction doesn't do anything, because mean-field had already been subtracted from plm and we don't want to repeat that.
            dlm_mod = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, int(simidx)))
            blt = self.itlib_iterator.get_template_blm(0, 0, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, dlm_mod=dlm_mod, perturbative=self.blt_pert, k=self.k)
            np.save(fn_blt, blt)
        return np.load(fn_blt)
    

    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "QE.get_blt({simidx}) started")
    # @log_on_end(logging.DEBUG, "QE.get_blt({simidx}) finished")
    def get_blt_new(self, simidx):

        def get_template_blm(it, it_e, lmaxb=1024, lmin_plm=self.Lmin, perturbative=False):
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


    # @log_on_start(logging.DEBUG, "get_filter() started")
    # @log_on_end(logging.DEBUG, "get_filter() finished")
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
        if not os.path.exists(self.libdir_QE):
            os.makedirs(self.libdir_QE)
        self.libdir_MAP = lambda qe_key, simidx, version: opj(self.TEMP, 'MAP/%s'%(qe_key), 'sim%04d'%(simidx) + version)
        self.libdir_blt = lambda simidx: opj(self.TEMP, 'MAP/%s'%(self.k), 'sim%04d'%(simidx) + self.version, 'BLT/')
        for simidx in np.array(list(set(np.concatenate([self.simidxs, self.simidxs_mf]))), dtype=int):
            # calculates all plms even for mf indices. This is not necessarily requested due to potentially simidxs =/= simidxs_mf, but otherwise collect and run must be adapted and its ok like this.
            libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
            if not os.path.exists(libdir_MAPidx):
                os.makedirs(libdir_MAPidx)
            if not os.path.exists(self.libdir_blt(simidx)):
                os.makedirs(self.libdir_blt(simidx))
        # TODO Only needed to hand over to ith()
        self.dlensalot_model = dlensalot_model
        
        # FIXME remnant of previous version when jobs were dependent on each other. This can perhaps be simplified now.
        self.simgen = Sim_generator(dlensalot_model)
        self.simulationdata = self.simgen.simulationdata
        self.qe = QE_lr(dlensalot_model, caller=self)
        self.qe.simulationdata = self.simgen.simulationdata # just to be sure, so we have a single truth in MAP_lr. 
        if self.OBD == 'OBD':
            # FIXME not sure why this was here.. that caused mismatch for calc_gradlik niv_job geom and qumaps when truncated
            # nivjob_geomlib_ = get_geom(self.nivjob_geominfo)
            self.tpl = template_dense(self.lmin_teb[2], self.nivjob_geomlib, self.tr, _lib_dir=self.obd_libdir, rescal=self.obd_rescale)
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
        log.info('------ init done ----')

    # # @base_exception_handler
    # @log_on_start(logging.DEBUG, "MAP.map.collect_jobs() started")
    # @log_on_end(logging.DEBUG, "MAP.collect_jobs() finished: jobs={self.jobs}")
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
    # @log_on_start(logging.DEBUG, "MAP.run() started")
    # @log_on_end(logging.DEBUG, "MAP.run() finished")
    def run(self):
        for taski, task in enumerate(self.it_tasks):
            log.info('{}, MAP task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
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


    @log_on_start(logging.DEBUG, "MAP.get_mchain() started")
    @log_on_end(logging.DEBUG, "MAP.get_mchain() finished")
    def get_mchain(self, simidx, key, it=0):
        libdir_MAPidx = self.libdir_MAP(self.k, simidx, self.version)
        itlib_iterator = transform(self, iterator_transformer(self, simidx, self.dlensalot_model))
        itlib_iterator.chain_descr = self.it_chain_descr(self.lm_max_unl[0], self.it_cg_tol(it))
        return itlib_iterator.get_mchain(it=it, key=key)


    # # @base_exception_handler
    # @log_on_start(logging.DEBUG, "MAP.get_plm_it(simidx={simidx}, its={its}) started")
    # @log_on_end(logging.DEBUG, "MAP.get_plm_it(simidx={simidx}, its={its}) finished")
    def get_plm_it(self, simidx, its):
        plms = rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), its)
        return plms


    # # @base_exception_handler
    # @log_on_start(logging.DEBUG, "MAP.get_meanfield_it(it={it}, calc={calc}) started")
    # @log_on_end(logging.DEBUG, "MAP.get_meanfield_it(it={it}, calc={calc}) finished")
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
    # @log_on_start(logging.DEBUG, "MAP.get_meanfields_it(its={its}, calc={calc}) started")
    # @log_on_end(logging.DEBUG, "MAP.get_meanfields_it(its={its}, calc={calc}) finished")
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
    # @log_on_start(logging.DEBUG, "MAP.get_blt_it(simidx={simidx}, it={it}) started")
    # @log_on_end(logging.DEBUG, "MAP.get_blt_it(simidx={simidx}, it={it}) finished")
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
                blt = self.itlib_iterator.get_template_blm(it, it-1, lmaxb=self.lm_max_blt[0], lmin_plm=self.Lmin, dlm_mod=dlm_mod, perturbative=False, k=self.k)
                np.save(fn_blt, blt)
        return np.load(fn_blt)


    # @log_on_start(logging.DEBUG, "get_filter() started")
    # @log_on_end(logging.DEBUG, "get_filter() finished")
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
    # @log_on_start(logging.DEBUG, "collect_jobs() started")
    # @log_on_end(logging.DEBUG, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO a valid job is any requested job?, as BLTs may also be on CFS
        jobs = []
        for idx in self.simidxs:
            jobs.append(idx)
        self.jobs = jobs

        return jobs


    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "run() started")
    # @log_on_end(logging.DEBUG, "run() finished")
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
    

    # # @log_on_start(logging.DEBUG, "get_basemap() started")
    # # @log_on_end(logging.DEBUG, "get_basemap() finished")  
    def get_basemap(self, simidx):
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
                    self.simulationdata.get_sim_sky(simidx, space='alm', spin=0, field='polarization')[1],
                    self.simulationdata.lmax, *self.lm_max_blt
                )
        elif self.basemap == 'lens_ffp10':
                return alm_copy(
                    planck2018_sims.cmb_len_ffp10.get_sim_blm(simidx),
                    None,
                    lmaxout=self.lm_max_blt[0],
                    mmaxout=self.lm_max_blt[1]
                )  
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
        log.info('got inbut Blms')
        blt_QE = self.get_blt_it(simidx, 0)
        
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
                    blt_MAP = self.get_blt_it(simidx, it)
                    bdel_MAP = self.nivjob_geomlib.alm2map(blm_L-blt_MAP, *self.lm_max_blt, nthreads=4)
                    blt_L_MAP = self.lib[maskflavour][maskid].map2cl(bdel_MAP)    
                    outputdata[0][2+iti][maskcounter] = blt_L_MAP
                    log.info("Finished MAP delensing for simidx {}, iteration {}".format(simidx, it))

                maskcounter+=1

        np.save(self.fns.format(simidx), outputdata)
            

    # @log_on_start(logging.DEBUG, "get_residualblens() started")
    # @log_on_end(logging.DEBUG, "get_residualblens() finished")
    def get_residualblens(self, simidx, it):
        basemap = self.get_basemap(simidx)
        
        return basemap - self.get_blt_it(simidx, it)
    

    # @base_exception_handler
    # @log_on_start(logging.DEBUG, "read_data() started")
    # @log_on_end(logging.DEBUG, "read_data() finished")
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
