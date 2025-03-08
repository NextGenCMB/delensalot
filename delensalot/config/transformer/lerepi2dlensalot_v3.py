#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build job model and global params from configuation file.
Each transformer is split into initializing the individual delensalot metamodel root model elements. 
"""
import os, sys
import copy
from os.path import join as opj
import logging
log = logging.getLogger(__name__)

import numpy as np
import hashlib
import itertools
from itertools import chain, combinations

from lenspyx.lensing import get_geom 

from delensalot.sims.data_source import DataSource

from delensalot.core import cachers
from delensalot.core.helper import utils_plancklens
from delensalot.core.job_handler import OBDBuilder, DataContainer, QEScheduler, MAPScheduler, MapDelenser, PhiAnalyser
from delensalot.core.MAP import curvature, operator
from delensalot.core.MAP.filter import Filter_3d as Filter
from delensalot.core.MAP.handler import Likelihood, Minimizer
from delensalot.core.MAP.gradient import Gradient, BirefringenceGradientSub, LensingGradientSub, GradSub

from delensalot.config.config_manager import set_config
from delensalot.config.config_helper import PLANCKLENS_keys, generate_plancklenskeys, filter_secondary_and_component
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV
from delensalot.config.metamodel.delensalot_mm_v3 import DELENSALOT_Concept_v3
from delensalot.config.etc.errorhandler import DelensalotError
from delensalot.utils import cli, camb_clfile

# from delensalot.core.helper import memorytracker
# memorytracker.MemoryTracker()

seclist_sorted = ['lensing', 'birefringence']

def get_TEMP_dir(cf):
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
    
def check_estimator_key(key):
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
    def process_DataSource(dl, si, cf):
        # NOTE this check key does not catch all possible wrong keys, but at least it catches the most common ones.
        # Plancklens keys should all be correct with this, for delensalot, not so sure, will see over time.
        check_estimator_key(cf.analysis.estimator_key)
        check_estimator_key(cf.data_source.generator_key)

        # NOTE processing comes in two steps.
        #   1.  all infos are validated here. everything is controlled by sec_info / generator_key. If subsequent infos (obs_info, operator_info) contains more info, remove them.
        #   2.  DataContainer checks if data had already been generated by DataSource. If so, it updates the libdir infos accordingly.
        for ope in si.operator_info:
            si.operator_info[ope]['tr'] = dl.tr

        # NOTE remove all operators that are not in sec_info, and add component information to operator_info
        operator_info = copy.deepcopy(si.operator_info)
        to_delete = [ope for ope in operator_info if ope not in si.sec_info]
        for ope in to_delete:
            del operator_info[ope]
        for sec in si.sec_info:
            operator_info[sec]['component'] = [c[0] for c in si.sec_info[sec]['component']]

        for sec in si.sec_info:
            si.sec_info[sec]['LM_max'] = operator_info[sec]['LM_max']
        si.operator_info = operator_info
        
        dl.data_source = DataSource(**si.__dict__)


    def process_Analysis(dl, an, cf):
        dl.beam_FWHM = an.beam_FWHM
        dl.mask_fn = an.mask_fn
        dl.estimator_key = an.estimator_key
        dl.lmin_teb = an.lmin_teb
        dl.idxs = an.idxs

        dl.idxs_mf = np.array(an.idxs_mf)
        dl.Nmf = 10000 if cf.maprec != DNaV and cf.maprec.mfvar.startswith('/') else len(dl.idxs_mf)
        
        dl.TEMP = get_TEMP_dir(cf)

        dl.lm_max_pri = an.lm_max_pri
        dl.LM_max = an.LM_max
        dl.lm_max_sky = an.lm_max_sky

        dl.analysis_secondary = filter_secondary_and_component(copy.deepcopy(cf.analysis.secondary), cf.analysis.estimator_key.split('_')[0])
        seclist_sorted = sorted(dl.analysis_secondary, key=lambda x: template_secondaries.index(x) if x in template_secondaries else float('inf'))
        complist_sorted = [comp for sec in seclist_sorted for comp in dl.analysis_secondary[sec]['component']]

        # NOTE all operators get the same lm_maxes. If I want to use different lm_maxes for the gradients, either,
        # 1. set in gradient classes and overwrite the settings of the operators, or
        # 2. instantiate new operators inside gradient class
        for sec in dl.analysis_secondary:
            dl.analysis_secondary[sec]['LM_max'] = cf.analysis.LM_max
            dl.analysis_secondary[sec]['lm_max_pri'] = cf.analysis.lm_max_pri
            dl.analysis_secondary[sec]['lm_max_sky'] = cf.analysis.lm_max_sky
        
        # NOTE this is to catch varying Lmin. It also supports that Lmin may come as list, or only a single value.
        # I make sure that the secondaries are sorted accordingly before I assign the Lmin values
        if isinstance(cf.analysis.Lmin, dict):
            dl.Lmin = cf.analysis.Lmin
        elif isinstance(cf.analysis.Lmin, (int, list, np.ndarray)):
            dl.Lmin = {comp: cf.analysis.Lmin if isinstance(cf.analysis.Lmin, int) or len(cf.analysis.Lmin) == 1 
                    else cf.analysis.Lmin[i] for i, comp in enumerate(complist_sorted)}
        dl.CLfids = dl.data_source.get_CLfids(0, dl.analysis_secondary, dl.Lmin)

        dl.cls_len = camb_clfile(an.cls_len)
        dl.zbounds = (-1,1)
        dl.zbounds_len = (-1,1)
        dl.transferfunction = utils_plancklens.gauss_beamtransferfunction(an.beam_FWHM, dl.lm_max_sky, an.lmin_teb, an.transfer_has_pixwindow, cf.noisemodel.geominfo)

    def process_Computing(dl, co, cf):
        dl.tr = co.OMP_NUM_THREADS
        os.environ["OMP_NUM_THREADS"] = str(dl.tr)

    def process_Noisemodel(dl, nm, cf):
        dl.nivjob_geomlib = get_geom(nm.geominfo).restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False)
        dl.rhits_normalised_fn = nm.rhits_normalised
        dl.nlev = nm.nlev
        dl.mask_fn = cf.analysis.mask_fn
        f = lambda x: utils_plancklens.get_niv_desc(nm.nlev, nm.geominfo, dl.nivjob_geomlib, dl.rhits_normalised_fn, dl.mask_fn, mode=x)
        buff_eb = f('P')
        buff_t = f('T')
        if "_" in dl.estimator_key:
            dl.data_key = cf.analysis.estimator_key.split('_')[1]
        else:
            dl.data_key = cf.analysis.estimator_key[-2:]
        dl.inv_operator_desc = {
            'niv_desc': {'t': buff_t, 'e': buff_eb, 'b': buff_eb},
            'lm_max': dl.lm_max_sky,
            'nlev': cf.noisemodel.nlev,
            'geom_lib': get_geom(nm.geominfo).restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False),
            'geominfo': nm.geominfo,
            'transferfunction': dl.transferfunction,
            'spectrum_type': nm.spectrum_type,
            'OBD': nm.OBD,
            'sky_coverage': nm.sky_coverage,
            "obd_rescale": cf.obd.rescale,
            "obd_libdir": cf.obd.libdir,
            "filtering_spatial_type": cf.noisemodel.spatial_type,
            'libdir': dl.TEMP,
            'data_key': dl.data_key,
        }


class l2delensalotjob_Transformer(l2base_Transformer):
    """builds delensalot job from configuration file
    """
    def build_datacontainer(self, cf): # TODO make sure this is right
        def extract():
            def _process_Analysis(dl, an, cf):
                dl.estimator_key = an.estimator_key
                if "_" in dl.estimator_key:
                    dl.data_key = an.estimator_key.split('_')[1]
                else:
                    dl.data_key = an.estimator_key[-2:]
                dl.idxs = an.idxs
                dl.idxs_mf = np.array(an.idxs_mf) # if dl.version != 'noMF' else np.array([])
            dl = DELENSALOT_Concept_v3()
            l2base_Transformer.process_Computing(dl, cf.computing, cf)
            _process_Analysis(dl, cf.analysis, cf)
            l2base_Transformer.process_DataSource(dl, cf.data_source, cf)
            ret = {
                "data_source": dl.data_source,
                "estimator_key": dl.estimator_key,
                'data_key': dl.data_key,
                'idxs': dl.idxs,
                'idxs_mf': dl.idxs_mf,
                'mask_fn': cf.analysis.mask_fn,
                'sky_coverage': cf.noisemodel.sky_coverage,
                'lm_max_sky': cf.analysis.lm_max_sky,
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
                def _process_DataSource(dl, si):
                    l2base_Transformer.process_DataSource(dl, si, cf)
                def _process_Qerec(dl, qe):
                    qe_tasks_sorted = ['calc_fields', 'calc_meanfields', 'calc_templates'] if qe.subtract_QE_meanfield else ['calc_fields', 'calc_templates']
                    dl.qe_tasks = [task for task in qe_tasks_sorted if task in qe.tasks]
                    dl.subtract_QE_meanfield = qe.subtract_QE_meanfield
                    dl.estimator_type = qe.estimator_type
                    
                _process_Computing(dl, cf.computing)
                _process_DataSource(dl, cf.data_source)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

            dl = DELENSALOT_Concept_v3()
            _process_components(dl)

            keystring = cf.analysis.estimator_key if len(cf.analysis.estimator_key) == 1 else '_'+cf.analysis.estimator_key.split('_')[-1] if "_" in cf.analysis.estimator_key else cf.analysis.estimator_key[-2:]
            QE_filterqest_desc = {
                "estimator_type": dl.estimator_type, # TODO this could be a different value for each secondary
                "libdir": opj(get_TEMP_dir(cf), 'QE', keystring),
                "cls_len": dl.cls_len,
                "cls_unl": dl.data_source.cls_lib.Cl_dict,
                "lm_max_ivf": dl.lm_max_sky,
                "lm_max_qlm": dl.LM_max, # TODO this could be a different value for each secondary
                "zbounds": dl.zbounds,
                "sht_threads": dl.tr,
                "cg_tol": cf.qerec.cg_tol,
                "lmin_teb": dl.lmin_teb,
                'inv_operator_desc': dl.inv_operator_desc,
            }
            QE_searchs_desc = {sec: {
                "estimator_key": generate_plancklenskeys(cf.analysis.estimator_key)[sec],
                'CLfids': dl.CLfids[sec],
                "subtract_meanfield": dl.subtract_QE_meanfield,
                "QE_filterqest_desc": QE_filterqest_desc,
                "ID": sec,
                "libdir": opj(get_TEMP_dir(cf), 'QE', keystring),
            } for sec in dl.analysis_secondary.keys()}
            
            QE_job_desc = {
                "template_operator": None, # template_operator
                "idxs": cf.analysis.idxs,
                "idxs_mf": dl.idxs_mf,
                "tasks": dl.qe_tasks,
            }

            dl.QE_searchs_desc = QE_searchs_desc
            dl.QE_job_desc = QE_job_desc
            ret = {
                "QE_searchs_desc": QE_searchs_desc,
                "QE_job_desc": QE_job_desc,
                'data_container': self.build_datacontainer(cf),
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
                def _process_DataSource(dl, si):
                    l2base_Transformer.process_DataSource(dl, si, cf)
                def _process_Itrec(dl, it):
                    dl.tasks = it.tasks
                    dl.cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.itmax = it.itmax
                    
                _process_Computing(dl, cf.computing)
                _process_DataSource(dl, cf.data_source)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_OBD(dl, cf.obd)
                _process_Itrec(dl, cf.maprec)

            dl = DELENSALOT_Concept_v3()
            _process_components(dl)

            QE_scheduler = self.build_QE_lensrec(cf)
            QE_searchs = QE_scheduler.QE_searchs
            data_container = self.build_datacontainer(cf)

            _process_components(dl)
            
            secs_run = [sec for sec in seclist_sorted if sec in dl.analysis_secondary]
            libdir = opj(get_TEMP_dir(cf), 'MAP',f"{cf.analysis.estimator_key}")


            # TODO pipeline for all estimator keys changes pipeline as follows:
            # 1. I'll privde a 3-tuple (TEB) data via get_data(). Depending on estimator key, some are empty
            # 2. operators expect 3-tuple, check if some are empty, and acts (spin-0,spin-2)
            # 3. filters expect 3-tuple. Check if some are empty, and acts
            #    a. fwd_op, precon, calc_prep: 3-tuple can be digested
            #    b. ivfres, wf: unclear. 
            filter_operators = []
            _MAP_operators_desc = {}
            for sec in secs_run:
                _MAP_operators_desc[sec] = {
                    "LM_max": dl.LM_max,
                    "component": dl.analysis_secondary[sec]['component'],
                    "libdir": opj(libdir, 'estimate/'),
                }
                if sec == "lensing":
                    _MAP_operators_desc[sec]["spin"] = 2 # TODO needs to change
                    _MAP_operators_desc[sec]["perturbative"] = False
                    _MAP_operators_desc[sec]['lm_max_in'] =  dl.lm_max_sky
                    _MAP_operators_desc[sec]['lm_max_out'] = dl.lm_max_pri
                    _MAP_operators_desc[sec]['data_key'] = dl.data_key
                    filter_operators.append(operator.Lensing(_MAP_operators_desc[sec]))
                elif sec == 'birefringence':
                    _MAP_operators_desc[sec]['lm_max'] = dl.lm_max_pri
                    filter_operators.append(operator.Birefringence(_MAP_operators_desc[sec]))
                    bire_grad_operator = operator.Secondary([filter_operators[-1]])
            sec_operator = operator.Secondary(filter_operators[::-1]) # NOTE gradients are sorted in the order of the secondaries, but the secondary operator, I want to act birefringence first.

            niv = operator.InverseNoiseVariance(**dl.inv_operator_desc)

            wf_info = {
                'chain_descr': lambda p2, p5 : [[0, ["diag_cl"], p2, dl.inv_operator_desc['geominfo'][1]['nside'], np.inf, p5, (lambda i: i - 1)]],
                'cg_tol': cf.maprec.cg_tol,
            }
            if "_" in dl.estimator_key:
                dl.data_key = cf.analysis.estimator_key.split('_')[1]
            else:
                dl.data_key = cf.analysis.estimator_key[-2:]
            if dl.data_key == 'tp':
                allowed_keys = ['tt', 'ee', 'te']
            elif dl.data_key in ['p', 'ee', 'eb']:
                allowed_keys = ['ee']
            elif dl.data_key == 'tt':
                allowed_keys = ['tt']
            cls_filt = {key:val[:dl.lm_max_pri[0]+1] for key, val in data_container.cls_lib.Cl_dict.items() if key in allowed_keys}
            MAP_wfivf_desc = {
                'sec_operator': sec_operator,
                'beam_operator': operator.Beam({'transferfunction': dl.transferfunction, 'lm_max': dl.lm_max_sky, 'data_key': dl.data_key}),
                'inv_operator': niv,
                'libdir': opj(libdir, 'filter/'),
                'add_operator': operator.Add({}),
                "chain_descr": wf_info['chain_descr'](dl.lm_max_pri[0], wf_info['cg_tol']),
                "cls_filt": cls_filt,
            }
            wfivf_filter = Filter(MAP_wfivf_desc)

            quad_desc = {
                "wfivf_filter": wfivf_filter,
                'data_container': data_container,
                "libdir": libdir,
                "LM_max": dl.LM_max,
                'sky_coverage': cf.noisemodel.sky_coverage,
                'geomlib': get_geom(('thingauss', {'lmax': 4500, 'smax': 3})), # FIXME this must match the geom in the operator
            }
            subs = []
            chhsall = []
            if 'lensing' in secs_run:
                CLfids_lens = dl.CLfids['lensing']
                quad_desc.update({
                    "component": dl.analysis_secondary['lensing']['component'],
                    "ID": 'lensing',
                    "sec_operator": sec_operator,
                    "chh": {comp: CLfids_lens[comp*2][:quad_desc['LM_max'][0]+1] * (0.5 * np.arange(quad_desc['LM_max'][0]+1) * np.arange(1,quad_desc['LM_max'][0]+2))**2 for comp in dl.analysis_secondary['lensing']['component']},
                    'data_key': dl.data_key,
                })
                lens_grad_quad = LensingGradientSub(quad_desc)
                chhsall.extend(list({comp: CLfids_lens[comp*2][:quad_desc['LM_max'][0]+1] * (0.5 * np.arange(quad_desc['LM_max'][0]+1) * np.arange(1,quad_desc['LM_max'][0]+2))**2 for comp in dl.analysis_secondary['lensing']['component']}.values()))
                subs.append(lens_grad_quad)

            if 'birefringence' in secs_run:
                CLfids_bire = dl.CLfids['birefringence']
                quad_desc.update({
                    "component": dl.analysis_secondary['birefringence']['component'],
                    "ID": 'birefringence',
                    'sec_operator': bire_grad_operator,
                    "chh": {comp: CLfids_bire[comp*2][:quad_desc['LM_max'][0]+1] for comp in dl.analysis_secondary['birefringence']['component']},
                })
                bire_grad_quad = BirefringenceGradientSub(quad_desc)
                chhsall.extend(list({comp: CLfids_bire[comp*2][:quad_desc['LM_max'][0]+1] for comp in dl.analysis_secondary['birefringence']['component']}.values()))
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
                    "estimator_key": cf.analysis.estimator_key,
                    "idx": idx,
                    "idx2": idx,
                } for idx in dl.idxs
            }
            likelihood = [Likelihood(**MAP_likelihood_desc) for MAP_likelihood_desc in MAP_likelihood_descs.values()]
            
            MAP_minimizer_descs = {
                idx: {
                    "estimator_key": cf.analysis.estimator_key,
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
            set_config(dl)
            return MAP_job_desc
        return MAPScheduler(**extract())