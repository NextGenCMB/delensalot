#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build job model and global params from configuation file.
Each transformer is split into initializing the individual delensalot metamodel root model elements. 
"""

import os, sys
import copy
from os.path import join as opj
import logging
log = logging.getLogger("global_logger")
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp
import hashlib
from itertools import chain, combinations
import re

from delensalot.core.cg import cd_solve

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.sims.data_source import DataSource

from delensalot.core.helper import utils_plancklens
from delensalot.core.handler import OBDBuilder, DataContainer, QEScheduler, MAPScheduler, MapDelenser, PhiAnalyser

from delensalot.config.etc.errorhandler import DelensalotError
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV
from delensalot.config.visitor import transform, transform3d
from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc, generate_plancklenskeys
from delensalot.config.metamodel.delensalot_mm_v3 import DELENSALOT_Model as DELENSALOT_Model_mm_v3, DELENSALOT_Concept_v3
from delensalot.utility.utils_hp import gauss_beam
from delensalot.utils import cli, camb_clfile, load_file

from delensalot.core.MAP.handler import Likelihood, Minimizer
from delensalot.core.MAP import curvature, filter, operator
from delensalot.core.MAP.gradient import Gradient, BirefringenceGradientSub, LensingGradientSub, GradSub

import itertools
from delensalot.config.config_helper import PLANCKLENS_keys


def check_key(key):
    def generate_delensalotcombinations(allowed_strings):
        characters = ['p', 'w', 'f']
        combinations = []
        for r in range(1, 4):
            for comb in itertools.combinations(characters, r):
                combinations.append(''.join(comb))
        combinations = sorted(set(combinations), key=lambda x: [characters.index(c) for c in x])
        result = []
        for s in allowed_strings:
            for prefix in combinations:
                new_string = prefix + s[1:]
                result.append(new_string)
        return result
    keys = generate_delensalotcombinations(PLANCKLENS_keys)
    if key not in keys:
        raise DelensalotError(f"Your input '{key}' is not a valid key. Please choose one of the following: {keys}")

def filter_secondary_and_component(secondary, allowed_chars):
    forbidden_chars_in_sec = 'eb'  # NOTE this filters the last part in case of non-symmetrized keys (e.g. 'pee')
    allowed_set = set("".join(c for c in allowed_chars if c not in forbidden_chars_in_sec))
    keys_to_remove = []
    for key, value in secondary.items():
        if 'component' in value:
            value['component'] = [char for char in value['component'] if char in allowed_set]
            if not value['component']:
                keys_to_remove.append(key)
    for key in keys_to_remove:
        del secondary[key]
    return secondary

def all_combinations(lst):
    return [''.join(comb) for comb in chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1))]

def get_hashcode(s):
    hlib = hashlib.sha256()
    hlib.update(str(s).encode())
    return hlib.hexdigest()[:4]

def atleast_2d(lst):
    if not isinstance(lst, list):
        raise TypeError("Input must be a list.")
    return lst if isinstance(lst[0], list) else [lst]

def atleast_1d(lst):
    return lst if isinstance(lst, list) else [lst]

template_secondaries = ['lensing', 'birefringence']  # NOTE define your desired order here - this must match with all other globally defined lists you might have in the handler.py modules

class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """
    def process_Simulation(dl, si, cf):

        # NOTE DataSource get the full simulation class. The validator will have removed the unneccessary secondaries if they were not explicitly set
        # in the config. DataSource controls everything with sec_info. Later in DataSource, operator_info etc. are filtered accordingly.
        for ope in si.operator_info:
            si.operator_info[ope]['tr'] = dl.tr
        dl.data_source = DataSource(**si.__dict__)

        # NOTE this check key does not catch all possible wrong keys, but at least it catches the most common ones.
        # Plancklens keys should all be correct with this, for delensalot, not so sure, will see over time.
        check_key(cf.analysis.key)
        
        dl.analysis_secondary = filter_secondary_and_component(copy.deepcopy(cf.analysis.secondary), cf.analysis.key.split('_')[0])
        seclist_sorted = sorted(dl.analysis_secondary, key=lambda x: template_secondaries.index(x) if x in template_secondaries else float('inf'))
        complist_sorted = [comp for sec in seclist_sorted for comp in dl.analysis_secondary[sec]['component']]
        # NOTE I update analysis_secondary with whatever is in the cf.analysis keys.
        for sec in dl.analysis_secondary:
            dl.analysis_secondary[sec]['LM_max'] = cf.analysis.LM_max
            dl.analysis_secondary[sec]['lm_max_pri'] = cf.analysis.lm_max_pri
            dl.analysis_secondary[sec]['lm_max_sky'] = cf.analysis.lm_max_sky

            # FIXME I do this here for consistency, but not sure it's the most natural thing to do. Which keys should be priorized?
            # I better only have one key...
            cf.analysis.secondary[sec]['lm_max_pri'] = cf.analysis.lm_max_pri
            cf.analysis.secondary[sec]['lm_max_sky'] = cf.analysis.lm_max_sky
            cf.analysis.secondary[sec]['LM_max'] = cf.analysis.LM_max
        
        # NOTE this is to catch varying Lmin. It also supports that Lmin may come as list, or only a single value.
        # I make sure that the secondaries are sorted accordingly before I assign the Lmin values
        if isinstance(cf.analysis.Lmin, dict):
            dl.Lmin = cf.analysis.Lmin
        elif isinstance(cf.analysis.Lmin, (int, list, np.ndarray)):
            dl.Lmin = {comp: cf.analysis.Lmin if isinstance(cf.analysis.Lmin, int) or len(cf.analysis.Lmin) == 1 
                    else cf.analysis.Lmin[i] for i, comp in enumerate(complist_sorted)}
        dl.CLfids = dl.data_source.get_CLfids(0, dl.analysis_secondary, dl.Lmin)
        
    def process_Analysis(dl, an, cf):
        dl.beam = an.beam
        dl.mask_fn = an.mask
        dl.k = an.key
        dl.lmin_teb = an.lmin_teb
        dl.idxs = an.idxs

        dl.lm_max_pri = an.lm_max_pri
        dl.LM_max = an.LM_max
        dl.lm_max_sky = an.lm_max_sky

        dl.idxs_mf = np.array(an.idxs_mf)
        dl.Nmf = 10000 if cf.maprec != DNaV and cf.maprec.mfvar.startswith('/') else len(dl.idxs_mf)
        
        dl.TEMP = transform(cf, l2T_Transformer())
        
        dl.cls_len = camb_clfile(an.cls_len)
        if an.zbounds[0] == 'nmr_relative':
            dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), an.zbounds[1])
        elif an.zbounds[0] == 'mr_relative':
            _zbounds = df.get_zbounds(hp.read_map(an.mask), np.inf)
            dl.zbounds = df.extend_zbounds(_zbounds, degrees=an.zbounds[1])
        elif type(an.zbounds[0]) in [float, int, np.float64]:
            dl.zbounds = an.zbounds
        if an.zbounds_len[0] == 'extend':
            dl.zbounds_len = df.extend_zbounds(dl.zbounds, degrees=an.zbounds_len[1])
        elif an.zbounds_len[0] == 'max':
            dl.zbounds_len = [-1, 1]
        elif type(an.zbounds_len[0]) in [float, int, np.float64]:
            dl.zbounds_len = an.zbounds_len
        dl.lm_max_pri = an.lm_max_pri
        dl.ttebl = utils_plancklens.gauss_beamtransferfunction(an.beam, dl.lm_max_sky, an.lmin_teb, an.transfunction_desc=='gauss_with_pixwin', cf.noisemodel.geominfo)

    def process_Computing(dl, co, cf):
        dl.tr = co.OMP_NUM_THREADS
        os.environ["OMP_NUM_THREADS"] = str(dl.tr)

    def process_Noisemodel(dl, nm, cf):
        dl.sky_coverage = nm.sky_coverage
        dl.nivjob_geomlib = get_geom(nm.geominfo).restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False)
        dl.nivjob_geominfo = nm.geominfo
        dl.rhits_normalised = nm.rhits_normalised if dl.sky_coverage == 'masked' else None
        dl.fsky = 1.0
        dl.spectrum_type = nm.spectrum_type
        dl.OBD = nm.OBD
        dl.nlev = cf.noisemodel.nlev
        # FIXME this can be replaced with the function in helper.obs
        dl.mask = hp.read_map(cf.analysis.mask) if cf.analysis.mask is not None else None
        # dl.niv_desc = {'T': l2OBD_Transformer.get_niv_desc(cf, dl, mode="T"), 'P': l2OBD_Transformer.get_niv_desc(cf, dl, mode="P")}
        f = lambda x: utils_plancklens.get_niv_desc(dl.nlev, dl.nivjob_geominfo, dl.nivjob_geomlib, dl.rhits_normalised, dl.mask, mode=x)
        dl.niv_desc = {'T': f('T'), 'P': f('P')}

class l2T_Transformer:
    # TODO this could use refactoring. Better name generation
    """global access for custom TEMP directory name, so that any job stores the data at the same place.
    """
    def build(self, cf):
        if cf.job.jobs == ['build_OBD']:
            return cf.obd.libdir
        else:       
            if cf.analysis.TEMP_suffix != '':
                _suffix = cf.analysis.TEMP_suffix
                # NOTE this might not work if I don't know anything about the simulations...
                _secsuffix = "simdata__"+"_".join(f"{key}_{'_'.join(values['component'])}" for key, values in cf.data_source.sec_info.items())
            _suffix += '_OBD' if cf.noisemodel.OBD == 'OBD' else '_lminB'+str(cf.analysis.lmin_teb[2])+_secsuffix
            TEMP =  opj(os.environ['SCRATCH'], 'analysis', _suffix)
            return TEMP

class l2delensalotjob_Transformer(l2base_Transformer):
    """builds delensalot job from configuration file
    """
    def build_datacontainer(self, cf): # TODO make sure this is right
        def extract():
            def _process_Analysis(dl, an, cf):
                dl.k = an.key
                dl.lmin_teb = an.lmin_teb
                dl.idxs = an.idxs
                dl.idxs_mf = np.array(an.idxs_mf) # if dl.version != 'noMF' else np.array([])
                dl.TEMP = transform(cf, l2T_Transformer())
            dl = DELENSALOT_Concept_v3()
            l2base_Transformer.process_Computing(dl, cf.computing, cf)
            _process_Analysis(dl, cf.analysis, cf)
            dl.libdir_suffix = cf.data_source.libdir_suffix
            l2base_Transformer.process_Simulation(dl, cf.data_source, cf)
            data_source = DataSource(**cf.data_source.__dict__)
            ret = {
                "data_source": data_source,
                "k": dl.k,
                'idxs': dl.idxs,
                'idxs_mf': dl.idxs_mf,
                'TEMP': dl.TEMP,
            }
            return ret
        return DataContainer(**extract())

    def build_QE_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)
                def _process_Analysis(dl, an):
                    l2base_Transformer.process_Analysis(dl, an, cf)
                def _process_Noisemodel(dl, nm):
                    l2base_Transformer.process_Noisemodel(dl, nm, cf)
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                def _process_Simulation(dl, si):
                    l2base_Transformer.process_Simulation(dl, si, cf)
                def _process_Qerec(dl, qe):
                    qe_tasks_sorted = ['calc_fields', 'calc_meanfields', 'calc_templates'] if qe.subtract_QE_meanfield else ['calc_fields', 'calc_templates']
                    dl.qe_tasks = [task for task in qe_tasks_sorted if task in qe.tasks]
                    dl.subtract_QE_meanfield = qe.subtract_QE_meanfield
                    dl.estimator_type = qe.estimator_type
                    
                    ## FIXME cg chain currently only works with healpix geometry
                    dl.QE_cg_tol = qe.cg_tol
                    
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.data_source)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

            dl = DELENSALOT_Concept_v3()
            _process_components(dl)
            # FIXME decidle later what to do with this template operator. This can probably be moved to the QE job. It is slightly unnatural as there is no deflection in QE, hence
            # the freedom to choose where it should go. But I also want to have it if I directly instantiate the QE search...

            # lenjob_geomlib = get_geom(dl.analysis_secondary['lensing']['geominfo'])
            # thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
            # lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
            # dl.ffi = deflection(lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.analysis_secondary['lensing']['LM_max'])), dl.analysis_secondary['lensing']['LM_max'][1], numthreads=dl.tr, verbosity=False, epsilon=dl.analysis_secondary['lensing']['epsilon'])
            # _QE_operators_desc = {}
            # filter_operators = []
            # for sec in ['lensing', 'birefringence']:
            #     if sec in dl.analysis_secondary:
            #         sec_data = dl.analysis_secondary[sec]
            #         _QE_operators_desc[sec] = {
            #             "LM_max": dl.LM_max,
            #             "lm_max_pri": dl.lm_max_pri,
            #             "lm_max_sky": dl.lm_max_sky,
            #             # "Lmin": dl.Lmin,
            #             "tr": dl.tr,
            #             "ffi": dl.ffi,
            #             "component": sec_data["component"],
            #             "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring, 'estimates', sec),
            #             "field_fns": QE_secs[sec].klm_fns,
            #         }
            #         if sec == "lensing":
            #             _QE_operators_desc[sec]["perturbative"] = True
            #         filter_operators.append(getattr(operator, sec)(_QE_operators_desc[sec]))
            # template_operator = operator.secondary_operator(filter_operators)


            keystring = cf.analysis.key if len(cf.analysis.key) == 1 else '_'+cf.analysis.key.split('_')[-1] if "_" in cf.analysis.key else cf.analysis.key[-2:]
            QE_filterqest_desc = {
                "estimator_type": dl.estimator_type, # TODO this could be a different value for each secondary
                "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring),
                "nivjob_geominfo": cf.noisemodel.geominfo,
                "niv_desc": dl.niv_desc,
                "nlev": dl.nlev,
                "filtering_spatial_type": cf.noisemodel.spatial_type,
                "cls_len": dl.cls_len,
                "cls_unl": dl.data_source.cls_lib.Cl_dict,
                "ttebl": dl.ttebl,
                "lm_max_ivf": dl.lm_max_sky,
                "lm_max_qlm": dl.LM_max, # TODO this could be a different value for each secondary
                "zbounds": dl.zbounds,
                "obd_libdir": dl.obd_libdir,
                "obd_rescale": dl.obd_rescale,
                "sht_threads": dl.tr,
                "QE_cg_tol": dl.QE_cg_tol,
                "OBD": dl.OBD,
                "lmin_teb": dl.lmin_teb,
            }

            QE_searchs_desc = {sec: {
                    "estimator_key": generate_plancklenskeys(cf.analysis.key)[sec],
                    'CLfids': dl.CLfids[sec],
                    "subtract_meanfield": dl.subtract_QE_meanfield,
                    "QE_filterqest_desc": QE_filterqest_desc,
                    "ID": sec,
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring),
                } for sec in dl.analysis_secondary.keys()}
            
            QE_job_desc = {
                "template_operator": None, # template_operator
                "idxs": cf.analysis.idxs,
                "idxs_mf": dl.idxs_mf,
                "QE_tasks": dl.qe_tasks,
            }

            dl.QE_searchs_desc = QE_searchs_desc
            dl.QE_job_desc = QE_job_desc
            ret = {
                "QE_searchs_desc": QE_searchs_desc,
                "QE_job_desc": QE_job_desc,
                'data_container': self.build_datacontainer(cf),
                'TEMP': dl.TEMP,

            }
            return ret
        return QEScheduler(**extract())


    def build_MAP_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)
                def _process_Analysis(dl, an):
                    l2base_Transformer.process_Analysis(dl, an, cf)
                def _process_Noisemodel(dl, nm):
                    l2base_Transformer.process_Noisemodel(dl, nm, cf)
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                def _process_Simulation(dl, si):
                    l2base_Transformer.process_Simulation(dl, si, cf)
                def _process_Qerec(dl, qe):
                    qe_tasks_sorted = ['calc_fields', 'calc_meanfields', 'calc_templates'] if qe.subtract_QE_meanfield else ['calc_fields', 'calc_templates']
                    dl.qe_tasks = [task for task in qe_tasks_sorted if task in qe.tasks]
                    dl.subtract_QE_meanfield = qe.subtract_QE_meanfield
                    dl.estimator_type = qe.estimator_type
                    
                    ## FIXME cg chain currently only works with healpix geometry
                    dl.QE_cg_tol = qe.cg_tol
                def _process_Itrec(dl, it):
                    dl.tasks = it.tasks
                    dl.cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.itmax = it.itmax
                    
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.data_source)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.maprec)

            dl = DELENSALOT_Concept_v3()
            _process_components(dl)

            QE_scheduler = self.build_QE_lensrec(cf)
            QE_searchs = QE_scheduler.QE_searchs
            _process_components(dl)

            seclist_sorted = ['lensing', 'birefringence']
            dl.analysis_secondary = filter_secondary_and_component(copy.deepcopy(cf.analysis.secondary), cf.analysis.key.split('_')[0])
            secs_run = [sec for sec in seclist_sorted if sec in dl.analysis_secondary]
            noise_info = {"nlev": dl.nlev, 'niv_desc': dl.niv_desc}
            obs_info = {'beam_transferfunction': dl.ttebl}
            libdir = opj(transform(cf, l2T_Transformer()), 'MAP',f"{cf.analysis.key}")

            data_source = DataSource(**cf.data_source.__dict__)
            data_container_desc = {
                "data_source": data_source,
                "k": dl.k,
                'idxs': dl.idxs,
                'idxs_mf': dl.idxs_mf,
                'TEMP': dl.TEMP,
            }
            data_container = DataContainer(**data_container_desc)

            filter_operators = []
            _MAP_operators_desc = {}
            for sec in secs_run:
                _MAP_operators_desc[sec] = {
                    "LM_max": dl.LM_max,
                    'lm_max_in': dl.lm_max_sky,
                    'lm_max_out': dl.lm_max_pri,
                    "component": dl.analysis_secondary[sec]['component'],
                    "libdir": opj(libdir, 'estimate/'),
                }
                if sec == "lensing":
                    _MAP_operators_desc[sec]["perturbative"] = False
                    filter_operators.append(operator.LensingOperator(_MAP_operators_desc[sec]))
                elif sec == 'birefringence':
                    filter_operators.append(operator.BirefringenceOperator(_MAP_operators_desc[sec]))
            sec_operator = operator.SecondaryOperator(filter_operators[::-1]) # NOTE gradients are sorted in the order of the secondaries, but the secondary operator, I want to act birefringence first.

            wf_info = {
                'chain_descr': lambda p2, p5 : [[0, ["diag_cl"], p2, dl.nivjob_geominfo[1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]],
                'cg_tol': 1e-7,
            }
            MAP_wf_desc = {
                'wf_operator': sec_operator,
                'beam_operator': operator.BeamOperator({'transferfunction': obs_info['beam_transferfunction'], 'lm_max': dl.lm_max_sky}),
                'inoise_operator': operator.InoiseOperator(nlev=noise_info['nlev'], lm_max=dl.lm_max_sky),
                'libdir': opj(libdir, 'filter/'),
                "chain_descr": wf_info['chain_descr'](dl.lm_max_pri[0], wf_info['cg_tol']),
                "cls_filt": data_container.cls_lib.Cl_dict,
            }
            wf_filter = filter.WF(MAP_wf_desc)

            MAP_ivf_desc = {
                'ivf_operator': sec_operator,
                'beam_operator': operator.BeamOperator({'transferfunction': obs_info['beam_transferfunction'], 'lm_max': dl.lm_max_sky}),
                'inoise_operator': operator.InoiseOperator(nlev=noise_info['nlev'], lm_max=dl.lm_max_sky),
                'libdir': opj(libdir, 'filter/'),
            }
            ivf_filter = filter.IVF(MAP_ivf_desc)

            quad_desc = {
                "ivf_filter": ivf_filter,
                "wf_filter": wf_filter,
                'data_container': data_container,
                "libdir": libdir,
                "LM_max": dl.LM_max,
                "sec_operator": sec_operator,
                'data_key': 'p',
            }
            subs = []
            chhsall = []
            if 'lensing' in secs_run:
                CLfids_lens = dl.CLfids['lensing']
                quad_desc.update({
                    "component": dl.analysis_secondary['lensing']['component'],
                    "ID": 'lensing',
                    "chh": {comp: CLfids_lens[comp*2][:4000+1] * (0.5 * np.arange(4000+1) * np.arange(1,4000+2))**2 for comp in dl.analysis_secondary['lensing']['component']},
                })
                lens_grad_quad = LensingGradientSub(quad_desc)
                chhsall.extend(list({comp: CLfids_lens[comp*2][:4000+1] * (0.5 * np.arange(4000+1) * np.arange(1,4000+2))**2 for comp in dl.analysis_secondary['lensing']['component']}.values()))
                subs.append(lens_grad_quad)

            if 'birefringence' in secs_run:
                CLfids_bire = dl.CLfids['birefringence']
                quad_desc.update({
                    "component": dl.analysis_secondary['birefringence']['component'],
                    "ID": 'birefringence',
                    "chh": {comp: CLfids_bire[comp*2][:4000+1] for comp in dl.analysis_secondary['birefringence']['component']},
                })
                bire_grad_quad = BirefringenceGradientSub(quad_desc)
                chhsall.extend(list({comp: CLfids_bire[comp*2][:4000+1] for comp in dl.analysis_secondary['birefringence']['component']}.values()))
                subs.append(bire_grad_quad)

            ncompsallsecs = sum([len(dl.analysis_secondary[sec]['component']) for sec in secs_run])
            ipriormatrix = np.ones(shape=(ncompsallsecs,ncompsallsecs,quad_desc['LM_max'][0]+1))
            i_ = 0
            for i, sec in enumerate(secs_run):
                for j, comp in enumerate(dl.analysis_secondary[sec]['component']):
                    chh = chhsall[i_]
                    ipriormatrix[i_,i_] = cli(chh)
                    i_+=1
            joint_desc = {
                'subs': subs,
                'ipriormatrix': ipriormatrix
            }
            gradient = Gradient(**joint_desc)

            MAP_likelihood_descs = {
                idx: {
                    'data_container': data_container,
                    'gradient_lib': gradient,
                    'libdir': libdir,
                    "QE_searchs": QE_searchs,
                    "lm_max_sky": dl.lm_max_sky,
                    "estimator_key": cf.analysis.key,
                    "idx": idx,
                    "idx2": idx,
                } for idx in dl.idxs
            }
            likelihood = [Likelihood(**MAP_likelihood_desc) for MAP_likelihood_desc in MAP_likelihood_descs.values()]
            

            MAP_minimizer_descs = {
                idx: {
                    "estimator_key": cf.analysis.key,
                    "likelihood": likelihood[idx],
                    'itmax': dl.itmax,
                    "libdir": libdir,
                    'idx': idx,
                    'idx2': idx,
                } for idx in dl.idxs
            }
            MAP_minimizers = [Minimizer(**MAP_minimizer_desc) for MAP_minimizer_desc in MAP_minimizer_descs.values()]
            

            MAP_job_desc = {
                "idxs": cf.analysis.idxs,
                "idxs_mf": dl.idxs_mf,
                'data_container': data_container,
                "QE_searchs": QE_searchs,
                "tasks": dl.tasks,
                "MAP_minimizers": MAP_minimizers,
            }
            return MAP_job_desc 

        return MAPScheduler(**extract())


@transform.case(DELENSALOT_Model_mm_v3, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)