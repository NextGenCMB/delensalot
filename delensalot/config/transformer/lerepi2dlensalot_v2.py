#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build job model and global params from configuation file.
Each transformer is split into initializing the individual delensalot metamodel root model elements. 
"""

import os, sys
import copy
from os.path import join as opj

import logging
log = logging.getLogger(__name__)
loglevel = log.getEffectiveLevel()
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp
import hashlib
from collections import OrderedDict
from itertools import chain, combinations
import re

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

## TODO don't like this import here. Not sure how to remove
from delensalot.core.cg import cd_solve
from delensalot.core.helper import utils_plancklens
from delensalot.sims.data_source import DataSource
from delensalot.config.etc.errorhandler import DelensalotError
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV

from delensalot.utils import cli, camb_clfile, load_file
from delensalot.utility.utils_hp import gauss_beam

from delensalot.core.MAP import field as MAP_field, operator
from delensalot.core.handler import OBD_builder, DataContainer, QE_lr_v2, MAP_lr_v2, Map_delenser, Phi_analyser

from delensalot.config.visitor import transform, transform3d
from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc
from delensalot.config.metamodel.delensalot_mm_v2 import DELENSALOT_Model as DELENSALOT_Model_mm, DELENSALOT_Concept_v2

import itertools

PLANCKLENS_keys_fund = ['ptt', 'xtt', 'p_p', 'x_p', 'p', 'x', 'stt', 's', 'ftt','f_p', 'f','dtt', 'ntt','n', 'a_p',
                    'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                    'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']
PLANCKLENS_keys = PLANCKLENS_keys_fund + ['p_tp', 'x_tp', 'p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb', 'ptt_bh_n',
                                'ptt_bh_s', 'ptt_bh_f', 'ptt_bh_d', 'dtt_bh_p', 'stt_bh_p', 'ftt_bh_d',
                                'p_bh_s', 'p_bh_n']

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

def generate_plancklenskeys(input_str):
    def split_at_first(s, blacklist={'t', 'e', 'b'}):
        match = re.search(f"[{''.join(blacklist)}]", s)
        if match:
            return s[:match.start()], s[match.start():]
        return s, ''
    lensing_components = {'p', 'w'}
    birefringence_components = {'f'}
    valid_suffixes = {'p', 'ee', 'eb'}
    transtable = str.maketrans({'p':"p", 'f':"a", 'w':"x"})
    if "_" in input_str:
        components_part, suffix = input_str.split('_')
    else:
        components_part, suffix = split_at_first(input_str)  # last character as suffix
    lensing = sorted(components_part[i] for i in range(len(components_part)) if components_part[i] in lensing_components)
    birefringence = sorted(components_part[i] for i in range(len(components_part)) if components_part[i] in birefringence_components)
    secondary_key = {}
    if lensing:
        secondary_key['lensing'] = {comp: comp.translate(transtable) + "_" + suffix if "_" in input_str else comp.translate(transtable)+ suffix for comp in lensing}
    if birefringence:
        secondary_key['birefringence'] = {comp: comp.translate(transtable) + "_" + suffix if "_" in input_str else comp.translate(transtable) + suffix for comp in birefringence}

    for sec, val in secondary_key.items():
        for comp in val.values():
            if comp not in PLANCKLENS_keys:
                raise DelensalotError(f"Your input '{input_str}' is not a valid key, it generated '{comp}' which is not a valid Plancklens key.")
    print(f'the generated secondary keys for Plancklens are {input_str} - > {secondary_key}')
    return secondary_key

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
    def process_DataSource(dl, si, cf):

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
        # NOTE I update analysis_seocndary with whatever is in the cf.analysis keys.
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
        if loglevel <= 20:
            dl.verbose = True
        elif loglevel >= 30:
            dl.verbose = False
        dl.beam = an.beam
        dl.mask_fn = an.mask
        dl.k = an.key
        dl.lmin_teb = an.lmin_teb
        dl.simidxs = an.simidxs

        dl.lm_max_pri = an.lm_max_pri
        dl.LM_max = an.LM_max
        dl.lm_max_sky = an.lm_max_sky

        dl.simidxs_mf = np.array(an.simidxs_mf)
        dl.Nmf = 10000 if cf.itrec != DNaV and cf.itrec.mfvar.startswith('/') else len(dl.simidxs_mf)
        
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
        if dl.sky_coverage == 'masked':
            dl.rhits_normalised = nm.rhits_normalised
            dl.fsky = np.mean(l2OBD_Transformer.get_niv_desc(cf, dl, mode="P")[0][1])  # Expensive, could affect future fsky calc
        else:
            dl.fsky = 1.0
        dl.spectrum_type = nm.spectrum_type
        dl.OBD = nm.OBD
        dl.nlev = l2OBD_Transformer.get_nlev(cf)
        # FIXME this can be replaced with the function in helper.obs
        dl.niv_desc = {'T': l2OBD_Transformer.get_niv_desc(cf, dl, mode="T"), 'P': l2OBD_Transformer.get_niv_desc(cf, dl, mode="P")}


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


# FIXME remove this class
class l2OBD_Transformer:
    """Transformer for generating a delensalot model for the calculation of the OBD matrix
    """
    def get_nlev(cf):
        return cf.noisemodel.nlev
    
    def get_niv_desc(cf, dl, mode):
        """Generate noise inverse variance (NIV) description for temperature ('T') or polarization ('P')."""
        nlev = l2OBD_Transformer.get_nlev(cf)
        masks, noisemodel_rhits_map = l2OBD_Transformer.get_masks(cf, dl)
        noisemodel_norm = np.max(noisemodel_rhits_map)

        niv_map = getattr(cf.noisemodel, f"niv{mode.lower()}_map")
        if niv_map is None:
            if dl.nivjob_geominfo[0] != 'healpix':
                assert 0, 'needs testing, please choose Healpix geom for nivjob for now'
            pixel_area = hp.nside2pixarea(dl.nivjob_geominfo[1]['nside'], degrees=True) * 3600  # Convert to arcmin²
            niv_desc = [np.array([pixel_area / nlev[mode] ** 2]) / noisemodel_norm] + masks
        else:
            niv_desc = [np.load(niv_map)] + masks

        return niv_desc

    def get_masks(cf, dl):
        # TODO refactor. This here generates a mask from the rhits map..
        # but this should really be detached from one another
        masks = []
        if cf.noisemodel.rhits_normalised is not None:
            msk = df.get_nlev_mask(cf.noisemodel.rhits_normalised[1], hp.read_map(cf.noisemodel.rhits_normalised[0]))
        else:
            msk = np.ones(shape=dl.nivjob_geomlib.npix())
        masks.append(msk)
        if cf.analysis.mask is not None:
            if type(cf.analysis.mask) == str:
                _mask = cf.analysis.mask
            elif cf.noisemodel.mask[0] == 'nlev':
                noisemodel_rhits_map = msk.copy()
                _mask = df.get_nlev_mask(cf.analysis.mask[1], noisemodel_rhits_map)
                _mask = np.where(_mask>0., 1., 0.)
        else:
            _mask = np.ones(shape=dl.nivjob_geomlib.npix())
        masks.append(_mask)

        return masks, msk


class l2delensalotjob_Transformer(l2base_Transformer):
    """builds delensalot job from configuration file
    """
    def build_datacontainer(self, cf):
        def extract():
            def _process_Analysis(dl, an, cf):
                dl.k = an.key
                dl.lmin_teb = an.lmin_teb
                dl.simidxs = an.simidxs
                dl.simidxs_mf = np.array(an.simidxs_mf) # if dl.version != 'noMF' else np.array([])
                dl.TEMP = transform(cf, l2T_Transformer())
            dl = DELENSALOT_Concept_v2()
            l2base_Transformer.process_Computing(dl, cf.computing, cf)
            _process_Analysis(dl, cf.analysis, cf)
            dl.libdir_suffix = cf.data_source.libdir_suffix
            l2base_Transformer.process_DataSource(dl, cf.data_source, cf)
            return dl
        return DataContainer(extract())


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
                    
                    ## FIXME cg chain currently only works with healpix geometry
                    dl.QE_cg_tol = qe.cg_tol
                    dl.chain_model = qe.chain
                    dl.chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    dl.chain_descr = lambda p2, p5 : [[0, ["diag_cl"], p2, dl.nivjob_geominfo[1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]]
                    
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_DataSource(dl, cf.data_source)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

            dl = DELENSALOT_Concept_v2()
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
                "chain_descr": dl.chain_descr,
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
                "simidxs": cf.analysis.simidxs,
                "simidxs_mf": dl.simidxs_mf,
                "QE_tasks": dl.qe_tasks,
            }

            dl.QE_searchs_desc = QE_searchs_desc
            dl.QE_job_desc = QE_job_desc
            return dl
        return QE_lr_v2(extract())


    def build_MAP_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Itrec(dl, it):
                    dl.it_tasks = it.tasks
                    # chain
                    dl.it_chain_descr = lambda p2, p5 : [[0, ["diag_cl"], p2, dl.nivjob_geominfo[1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]]
            
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.itmax = it.itmax

                    dl.soltn_cond = it.soltn_cond
                    lenjob_geomlib = get_geom(dl.analysis_secondary['lensing']['geominfo'])
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    dl.ffi = deflection(lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.analysis_secondary['lensing']['LM_max'])), dl.analysis_secondary['lensing']['LM_max'][1], numthreads=dl.tr, verbosity=False, epsilon=dl.analysis_secondary['lensing']['epsilon'])
                
                _process_Itrec(dl, cf.itrec)

            dl = DELENSALOT_Concept_v2()
            dl = self.build_QE_lensrec(cf)
            QE_searchs = dl.QE_searchs
            _process_components(dl)

            MAP_libdir_prefix = opj(transform(cf, l2T_Transformer()), 'MAP',f"{cf.analysis.key}")

            MAP_searchs_desc = {
                "estimator_key": cf.analysis.key,
                'libdir': MAP_libdir_prefix,
                "CLfids": dl.CLfids,
                "itmax" : dl.itmax,
                # 'lenjob_info': {
                #     'zbounds': dl.zbounds,
                #     'epsilon': dl.analysis_secondary['lensing']['epsilon'],
                #     'geominfo': ('thingauss', {'lmax': 4500, 'smax': 3}),
                # },
                'lm_maxs': {
                    'LM_max': dl.LM_max,
                    'lm_max_pri': dl.lm_max_pri,
                    'lm_max_sky': dl.lm_max_sky
                },
                'wf_info': {
                    'chain_descr': dl.it_chain_descr,
                    'cg_tol': dl.it_cg_tol(0),
                },
                'noise_info': {
                    'nlev': dl.nlev,
                    'niv_desc': dl.niv_desc
                },
                'obs_info': {
                    'beam_transferfunction': dl.ttebl,
                },
            }
            MAP_job_desc = {
                "simidxs": cf.analysis.simidxs,
                "simidxs_mf": dl.simidxs_mf,
                "QE_searchs": QE_searchs,
                "it_tasks": dl.it_tasks,  
            }
            dl.MAP_job_desc, dl.MAP_searchs_desc = MAP_job_desc, MAP_searchs_desc
            return dl

        return MAP_lr_v2(extract())


    def build_OBD_builder(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)
                def _process_Analysis(dl, an):
                    dl.mask_fn = an.mask
                    dl.lmin_teb = an.lmin_teb
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
                def _process_OBD(dl, od):
                    dl.libdir = od.libdir if type(od.libdir) == str else 'nopath'
                    dl.nlev_dep = od.nlev_dep
                    dl.rescale = od.rescale
                    if os.path.isfile(opj(dl.libdir,'tniti.npy')):
                        # TODO need to test if it is the right tniti.npy
                        # TODO dont exit, rather skip job
                        log.warning("tniti.npy in destination dir {} already exists.".format(dl.libdir))
                        log.warning("Please check your settings.")
                def _process_Noisemodel(dl, nm):
                    l2base_Transformer.process_Noisemodel(dl, nm, cf)
                    
                dl.TEMP = transform(cf, l2T_Transformer())

                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_OBD(dl, cf.obd)
                
                return dl

            dl = DELENSALOT_Concept_v2()
            _process_components(dl)
            return dl

        return OBD_builder(extract())


    def build_delenser(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)
                def _process_Noisemodel(dl, nm):
                    l2base_Transformer.process_Noisemodel(dl, nm, cf)
                def _process_Qerec(dl, qe):
                    pass
                def _process_Itrec(dl, it):
                    pass
                def _process_Madel(dl, ma):
                    pass

                dl.blt_pert = cf.qerec.blt_pert
                _process_Computing(dl, cf.computing)
                dl.libdir_suffix = cf.data_source.obs_info['noise_info']['libdir_suffix']
                dl.data_source = DataSource(**cf.data_source.__dict__)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Madel(dl, cf.madel)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                return dl

            dl = DELENSALOT_Concept_v2()
            _process_components(dl)
            return dl

        return Map_delenser(extract())


    def build_phianalyser(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)

                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)

                def _process_Noisemodel(dl, nm):
                    l2base_Transformer.process_Noisemodel(dl, nm, cf)
        
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
      
                def _process_DataSource(dl, si):
                    dl.libdir_suffix = cf.data_source.obs_info['noise_info']['libdir_suffix']
                    l2base_Transformer.process_DataSource(dl, si, cf)

                def _process_Qerec(dl, qe):
                    pass
                def _process_Itrec(dl, it):
                    pass
                def _process_Phianalysis(dl, pa):
                    dl.custom_WF_TEMP = pa.custom_WF_TEMP
                    dl.its = np.arange(dl.itmax)

                    # At modelbuild-stage I want to test if WF exists.
                    if type(dl.custom_WF_TEMP) == str:
                        fn = opj(dl.custom_WF_TEMP,'WFemp_%s_simall%s_itall%s_avg.npy')%(dl.k, len(dl.simidxs), len(dl.its))
                        if not os.path.isfile(fn):
                            log.error("WF @ {} does not exsit. Please check your settings.".format(fn))
                            # sys.exit()
                    elif dl.custom_WF_TEMP is None or type(dl.custom_WF_TEMP) == int:
                        dl.custom_WF_TEMP = None
      
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_DataSource(dl, cf.data_source)

                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)
                _process_Phianalysis(dl, cf.phana)

            dl = DELENSALOT_Concept_v2()
            _process_components(dl)
            return dl

        ## FIXME build a correct model. For now quick and dirty, MAP_lensrec contains all information (and more)
        return Phi_analyser(extract())
    

@transform.case(DELENSALOT_Model_mm, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DELENSALOT_Model_mm, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)