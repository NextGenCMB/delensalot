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

from delensalot.core.helper import utils_plancklens
from delensalot.core.job_handler import OBDBuilder, DataContainer, QEScheduler, MAPScheduler#, MapDelenser, PhiAnalyser
from delensalot.core.MAP import curvature, operator
from delensalot.core.MAP.filter import Filter_3d as Filter
from delensalot.core.MAP.handler import Likelihood, Minimizer
from delensalot.core.MAP.gradient import Gradient, BirefringenceGradientSub, LensingGradientSub, GradSub

from delensalot.config.config_manager import set_config
from delensalot.config.config_helper import PLANCKLENS_keys, generate_plancklenskeys, filter_secondary_and_component
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV
from delensalot.config.metamodel.delensalot_mm import DELENSALOT_Concept
from delensalot.config.etc.errorhandler import DelensalotError
from delensalot.utils import cli, camb_clfile
from delensalot.config.config_manager import get_config

# from delensalot.core.helper import memorytracker
# memorytracker.MemoryTracker()

def _op_builder_lensing(dl, libdir, extras):
    """
    Lensing operator D: acts on *primary* (pri) alms and outputs *sky* alms.
    In the full chain we want X_dat = B * F_beta * D * X + n  => apply D first, then F.
    """
    desc = {
        "LM_max": dl.LM_max,
        "component": dl.analysis_secondary["lensing"]["component"],
        "libdir": opj(libdir, "estimate/"),
        "sht_tr": dl.sht_tr,

        # IMPORTANT: forward should map pri -> sky
        "lm_max_in": dl.lm_max_pri,
        "lm_max_out": dl.lm_max_sky,

        "data_key": dl.data_key,
        "perturbative": False,
    }
    return operator.Lensing(desc)


def _op_builder_bire(dl, libdir, extras):
    """
    Birefringence operator F_beta: acts on *sky* alms (same harmonic band as pri in your setup,
    but conceptually it follows D). Your implementation uses lm_max and does alm2map/map2alm internally.
    """
    desc = {
        "LM_max": dl.LM_max,
        "component": dl.analysis_secondary["birefringence"]["component"],
        "libdir": opj(libdir, "estimate/"),
        "sht_tr": dl.sht_tr,
        "lm_max": dl.lm_max_sky,   # IMPORTANT: match sky band used in filtering legs
        "perturbative": False,
    }
    return operator.Birefringence(desc)


def _grad_builder_lensing(dl, libdir, extras):
    """
    Lensing gradient must see the FULL secondary chain (D then F), not just the lensing op.
    """
    wfivf_filter = extras["wfivf_filter"]
    data_container = extras["data_container"]
    full_sec_operator = extras["sec_operator"]

    CLfids_lens = dl.CLfids["lensing"]
    L = np.arange(dl.LM_max[0] + 1, dtype=float)

    # Your convention: chi-chi prior for (phi, omega) in "comp*2" keys, with kappa/phi scaling baked here
    chh_dict = {
        comp: CLfids_lens[comp * 2][: dl.LM_max[0] + 1] * (0.5 * L * (L + 1.0)) ** 2
        for comp in dl.analysis_secondary["lensing"]["component"]
    }

    quad_desc = {
        "wfivf_filter": wfivf_filter,
        "data_container": data_container,
        "libdir": libdir,
        "LM_max": dl.LM_max,
        "sht_tr": dl.sht_tr,
        "component": dl.analysis_secondary["lensing"]["component"],
        "ID": "lensing",
        "sec_operator": full_sec_operator,
        "chh": chh_dict,
        "data_key": dl.data_key,
        # "geomlib": get_geom(("thingauss", {"lmax": dl.lm_max_pri[0] + 2048, "smax": 3})),
    }

    lens_grad = LensingGradientSub(quad_desc)
    return lens_grad, list(chh_dict.values())


def _grad_builder_bire(dl, libdir, extras):
    """
    Birefringence gradient must also see the FULL chain (D then F), because
    your WF/IVF legs are produced with the full model.
    """
    wfivf_filter = extras["wfivf_filter"]
    data_container = extras["data_container"]
    full_sec_operator = extras["sec_operator"]

    CLfids_bire = dl.CLfids["birefringence"]
    chh_dict = {
        comp: CLfids_bire[comp * 2][: dl.LM_max[0] + 1]
        for comp in dl.analysis_secondary["birefringence"]["component"]
    }

    quad_desc = {
        "wfivf_filter": wfivf_filter,
        "data_container": data_container,
        "libdir": libdir,
        "LM_max": dl.LM_max,
        "sht_tr": dl.sht_tr,
        "component": dl.analysis_secondary["birefringence"]["component"],
        "ID": "birefringence",
        "sec_operator": full_sec_operator,
        "chh": chh_dict,
    }

    bire_grad = BirefringenceGradientSub(quad_desc)
    return bire_grad, list(chh_dict.values())


class SecondaryRegistry:
    """Registry holding separate builders for operator and gradient-sub for each secondary."""
    _registry = {}

    @classmethod
    def register(cls, name, op_builder, grad_builder):
        cls._registry[name] = {"op": op_builder, "grad": grad_builder}

    @classmethod
    def build_op(cls, name, dl, libdir, extras=None):
        return cls._registry[name]["op"](dl, libdir, extras or {})

    @classmethod
    def build_grad(cls, name, dl, libdir, extras=None):
        return cls._registry[name]["grad"](dl, libdir, extras or {})


def process_all_components(dl, cf):
    l2base_Transformer.process_Computing(dl, cf.computing, cf)
    l2base_Transformer.process_DataSource(dl, cf.data_source, cf)
    l2base_Transformer.process_Analysis(dl, cf.analysis, cf)
    l2base_Transformer.process_Noisemodel(dl, cf.noisemodel, cf)
    dl.obd_libdir = cf.obd.libdir
    dl.obd_rescale = cf.obd.rescale
    dl.tasks = cf.maprec.tasks
    dl.cg_tol = (lambda itr: cf.maprec.cg_tol if itr <= 1 else cf.maprec.cg_tol)
    dl.itmax = cf.maprec.itmax
    if "_" in cf.analysis.estimator_key:
        dl.data_key = cf.analysis.estimator_key.split("_")[1]
    else:
        dl.data_key = cf.analysis.estimator_key[-2:]


def build_cls_filt_from_container(data_container, dl):
    if dl.data_key == "tp":
        allowed_keys = ["tt", "ee", "te"]
    elif dl.data_key in ["p", "ee", "eb"]:
        allowed_keys = ["ee"]
    elif dl.data_key == "tt":
        allowed_keys = ["tt"]
    else:
        allowed_keys = list(data_container.cls_lib.Cl_dict.keys())
    return {key: val[: dl.lm_max_pri[0] + 1]
            for key, val in data_container.cls_lib.Cl_dict.items() if key in allowed_keys}


def build_chain_descr(dl, cf):
    def chain_descr(p2, p5):
        return [[0, ["diag_cl"], p2, dl.inv_operator_desc['geominfo'][1]['nside'], np.inf, p5, (lambda i: i - 1)]]
    return lambda p2, p5: chain_descr(p2, p5)


def build_iprior_matrix_from_chhs(chh_list, ncomps, LMmax0):
    ipriormatrix = np.zeros((ncomps, ncomps, LMmax0 + 1))
    for i in range(ncomps):
        ipriormatrix[i, i, :] = cli(chh_list[i])
    return ipriormatrix


def get_TEMP_dir(cf):
    if cf.job.jobs == ['build_OBD']:
        return cf.obd.libdir
    else:       
        if cf.analysis.TEMP_suffix != '':
            _suffix = cf.analysis.TEMP_suffix
            if cf.data_source.flavour == 'obs' or cf.data_source.flavour == 'sky':
                _secsuffix = '_unspecified_data'
            else:
                _secsuffix = "_datawith_" + ''.join(''.join(map(str, v['component'])) for v in cf.data_source.sec_info.values() if isinstance(v, dict) and 'component' in v)
        _suffix += '_OBD' if cf.noisemodel.OBD == 'OBD' else _secsuffix
        TEMP =  opj(os.environ['SCRATCH'], 'delensalot_analysis', _suffix)
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


class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """

    def process_DataSource(dl, si, cf):
        analysis_secondary = filter_secondary_and_component(copy.deepcopy(cf.analysis.secondary), cf.analysis.estimator_key.split('_')[0])
        # NOTE build the order of the secondaries according to analysis.seclist_sorted, and only keep what is listed in analysis.secondary
        dl.seclist_sorted = [s for s in cf.analysis.operator_order if s in analysis_secondary.keys()]
        dl.seclist_genSim_sorted = ([s for s in cf.data_source.operator_order if s in si.sec_info])

        dl.template_index_secondaries_genSim = {val: i for i, val in enumerate(dl.seclist_genSim_sorted)}
        # NOTE remove all sec_info that is not in seclist_sorted
        si.sec_info = {k:v for k, v in si.sec_info.items() if k in dl.seclist_genSim_sorted}
        
        # NOTE this check key does not catch all possible wrong keys, but at least it catches the most common ones.
        # Plancklens keys should all be correct with this, for delensalot, not so sure, will see over time.
        check_estimator_key(cf.analysis.estimator_key)
        # FIXME implement generator_key, then run next line
        # check_estimator_key(cf.data_source.generator_key)

        # NOTE processing comes in two steps.
        #   1.  all infos are validated here. everything is controlled by sec_info / generator_key. If subsequent infos (obs_info, operator_info) contains more info, remove them.
        #   2.  DataContainer checks if data had already been generated by DataSource. If so, it updates the libdir infos accordingly.
        for ope in si.operator_info:
            if ope in si.sec_info:
                si.operator_info[ope]['tr'] = dl.sht_tr

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
        si.operator_info = {k:v for k, v in sorted(operator_info.items(), key=lambda x: dl.template_index_secondaries_genSim.get(x[0], ''))}
        if (si.gaussianized_sims == True) or (si.gaussianized_sims == 'len'):
            si.libdir_suffix = "Gaussian_lensed_sims"
        elif si.gaussianized_sims == "unl":
            si.libdir_suffix = "Gaussian_unlensed_sims"
        else:
            si.libdir_suffix = "_then_".join(dl.seclist_genSim_sorted)
            # NOTE adding specific naming when secondary power spectra are modified
            clmod_tags = [
                f"{sec}_Clx{float(info['cl_modifier_factor']):g}"
                for sec, info in si.sec_info.items()
                if 'cl_modifier_factor' in info and not np.isclose(float(info['cl_modifier_factor']), 1.0)
            ]
            if clmod_tags:
                si.libdir_suffix += "_" + "_".join(clmod_tags)
        si.fixed_secondary_seed = getattr(cf.data_source, 'fixed_secondary_seed', None)
        set_config(cf)
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
        
        dl.template_index_secondaries = {val: i for i, val in enumerate(dl.seclist_sorted)}
        dl.analysis_secondary = {k:v for k, v in sorted(dl.analysis_secondary.items(), key=lambda x: dl.template_index_secondaries.get(x[0], ''))}
        complist_sorted = [comp for sec in dl.seclist_sorted if sec in dl.analysis_secondary for comp in dl.analysis_secondary[sec]['component']]

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
        dl.noLmin = {key: 1. for key in dl.Lmin.keys()}
        dl.CLfids = dl.data_source.get_CLfids(0, dl.analysis_secondary, dl.Lmin)
        dl.CLfidsNoLmin = dl.data_source.get_CLfids(0, dl.analysis_secondary, dl.noLmin)

        dl.cls_len = camb_clfile(an.cls_len)
        dl.zbounds = (-1,1)
        dl.zbounds_len = (-1,1)
        dl.transferfunction = utils_plancklens.gauss_beamtransferfunction(an.beam_FWHM, dl.lm_max_sky, an.lmin_teb, an.transfer_has_pixwindow, cf.noisemodel.geominfo)

    def process_Computing(dl, co, cf):
        dl.sht_tr = co.OMP_NUM_THREADS
        os.environ["OMP_NUM_THREADS"] = str(dl.sht_tr)

    def process_Noisemodel(dl, nm, cf):
        dl.nivjob_geomlib = get_geom(nm.geominfo) # .restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False)
        dl.noisemodel_geominfo = nm.geominfo
        dl.rhits_normalised_fn = nm.rhits_normalised if isinstance(nm.rhits_normalised, str) else None
        dl.nlev = nm.nlev
        dl.mask_fn = cf.analysis.mask_fn
        f = lambda x: utils_plancklens.get_niv_desc(nm.nlev, nm.geominfo, dl.nivjob_geomlib, np.load(dl.rhits_normalised_fn) if isinstance(nm.rhits_normalised, str) else None, dl.mask_fn, mode=x)
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
            'geom_lib': get_geom(nm.geominfo), #.restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False),
            'geominfo': nm.geominfo,
            'transferfunction': dl.transferfunction,
            'spectrum_type': nm.spectrum_type,
            'OBD': nm.OBD,
            'filtering_type': cf.qerec.filtering_type,
            "obd_rescale": cf.obd.rescale,
            "obd_libdir": cf.obd.libdir,
            'libdir': dl.TEMP,
            'data_key': dl.data_key,
            "sht_tr": dl.sht_tr,
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
            dl = DELENSALOT_Concept()
            l2base_Transformer.process_Computing(dl, cf.computing, cf)
            _process_Analysis(dl, cf.analysis, cf)
            l2base_Transformer.process_DataSource(dl, cf.data_source, cf)
            mask_ = cf.analysis.mask_fn if cf.analysis.mask_fn is not None else ''
            ret = {
                "data_source": dl.data_source,
                "estimator_key": dl.estimator_key,
                'data_key': dl.data_key,
                'idxs': dl.idxs,
                'idxs_mf': dl.idxs_mf,
                'mask_fn': cf.analysis.mask_fn,
                'sky_coverage': "masked" if os.path.isfile(mask_) else "full",
                'lm_max_sky': cf.analysis.lm_max_sky,
            }
            return ret
        return DataContainer(**extract())


    def build_QE_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction job (QE)
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
                    dl.TP_strategy = qe.TP_strategy
                    
                _process_Computing(dl, cf.computing)
                _process_DataSource(dl, cf.data_source)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

            dl = DELENSALOT_Concept()
            _process_components(dl)
            mask_ = cf.analysis.mask_fn if cf.analysis.mask_fn is not None else ''
            if cf.qerec.estimator_key is not None:
                if cf.qerec.estimator_key != cf.analysis.estimator_key:
                    est_key_loc =  cf.qerec.estimator_key
                else:
                    est_key_loc = cf.analysis.estimator_key
            else:
                est_key_loc = cf.analysis.estimator_key
            keystring = est_key_loc if len(est_key_loc) == 1 else '_'+est_key_loc.split('_')[-1] if "_" in est_key_loc else est_key_loc[-2:]
            QE_filterqest_desc = {
                "TP_strategy": dl.TP_strategy, # TODO this could be a different value for each secondary
                "libdir": opj(get_TEMP_dir(cf), 'QE', keystring),
                "cls_len": dl.cls_len,
                "cls_unl": dl.data_source.cls_lib.Cl_dict,
                "lm_max_ivf": dl.lm_max_sky,
                "lm_max_qlm": dl.LM_max, # TODO this could be a different value for each secondary
                "zbounds": dl.zbounds,
                "sht_threads": dl.sht_tr,
                "cg_tol": cf.qerec.cg_tol,
                "lmin_teb": dl.lmin_teb,
                'inv_operator_desc': dl.inv_operator_desc,
            }

            buff = generate_plancklenskeys(est_key_loc)
            QE_searchs_desc = {sec: {
                "estimator_key": buff[sec],
                'CLfids': dl.CLfids[sec],
                "CLfidsNoLmin": dl.CLfidsNoLmin[sec], # Note I need this solely to keep the low L in the meanfield
                "subtract_meanfield": dl.subtract_QE_meanfield,
                "QE_filterqest_desc": QE_filterqest_desc,
                "ID": sec,
                "libdir": opj(get_TEMP_dir(cf), 'QE', keystring),
                "qmflm_fn": cf.qerec.qmflm_fns[sec] if cf.qerec.qmflm_fns is not None else None,
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
        def extract():
            dl = DELENSALOT_Concept()
            process_all_components(dl, cf)

            QE_scheduler = self.build_QE_lensrec(cf)
            QE_searchs = QE_scheduler.QE_searchs
            data_container = self.build_datacontainer(cf)

            seclist_local = [s for s in dl.seclist_sorted if s in dl.analysis_secondary]
            libdir = opj(get_TEMP_dir(cf), "MAP", f"{cf.analysis.estimator_key}")
            os.makedirs(opj(libdir, "estimate/"), exist_ok=True)
            os.makedirs(opj(libdir, "filter/"), exist_ok=True)

            niv = operator.InverseNoiseVariance(**dl.inv_operator_desc)
            beam_op = operator.Beam(
                {"transferfunction": dl.transferfunction, "lm_max": dl.lm_max_sky, "data_key": dl.data_key}
            )
            add_op = operator.Add({})

            set_config(dl)

            SecondaryRegistry.register("lensing", _op_builder_lensing, _grad_builder_lensing)
            SecondaryRegistry.register("birefringence", _op_builder_bire, _grad_builder_bire)

            filter_ops = []
            ops_map = {}
            for sec in seclist_local:
                op_obj = SecondaryRegistry.build_op(sec, dl, libdir, extras={})
                filter_ops.append(op_obj)
                ops_map[sec] = op_obj

            sec_operator = operator.Secondary(filter_ops)

            mask_ = cf.analysis.mask_fn if cf.analysis.mask_fn is not None else ""
            MAP_wfivf_desc = {
                "filtering_type": cf.maprec.filtering_type,
                "sky_coverage": "masked" if os.path.isfile(mask_) else "full",
                "sec_operator": sec_operator,
                "beam_operator": beam_op,
                "inv_operator": niv,
                "libdir": opj(libdir, "filter/"),
                "add_operator": add_op,
                "chain_descr": build_chain_descr(dl, cf)(dl.lm_max_pri[0], cf.maprec.cg_tol),
                "cls_filt": build_cls_filt_from_container(data_container, dl),
                "sht_tr": dl.sht_tr,
            }
            wfivf_filter = Filter(MAP_wfivf_desc)

            grad_subs = []
            chh_all = []

            extras = {
                "data_container": data_container,
                "wfivf_filter": wfivf_filter,
                "operators": ops_map,          # optional, if subs still want the per-sec object
                "sec_operator": sec_operator,
                "seclist_local": seclist_local,
            }
            for sec in seclist_local:
                grad_obj, chh_list = SecondaryRegistry.build_grad(sec, dl, libdir, extras=extras)
                grad_subs.append(grad_obj)
                chh_all.extend(chh_list)

            ncompsallsecs = sum(len(dl.analysis_secondary[sec]["component"]) for sec in seclist_local)
            if len(chh_all) != ncompsallsecs:
                raise RuntimeError(
                    f"Mismatch: chh_all length {len(chh_all)} != ncompsallsecs {ncompsallsecs}"
                )

            ipriormatrix = build_iprior_matrix_from_chhs(chh_all, ncompsallsecs, dl.LM_max[0])
            gradient = Gradient(**{"subs": grad_subs, "ipriormatrix": ipriormatrix})

            MAP_likelihood_desc = {
                "data_container": data_container,
                "gradient_lib": gradient,
                "libdir": libdir,
                "QE_searchs": QE_searchs,
            }
            likelihood = Likelihood(**MAP_likelihood_desc)

            use_QE_for_lowL = cf.maprec.use_QE_for_lowL
            use_QE_starting_point = cf.maprec.use_QE_starting_point
            MAP_minimizer_desc = {
                "likelihood": likelihood,
                "itmax": dl.itmax,
                "libdir": libdir,
                "use_QE_starting_point": use_QE_starting_point,
                "use_QE_for_lowL": use_QE_for_lowL,
            }
            MAP_minimizer = Minimizer(**MAP_minimizer_desc)

            MAP_job_desc = {
                "idxs": cf.analysis.idxs,
                "idxs_mf": dl.idxs_mf,
                "data_container": data_container,
                "QE_searchs": QE_searchs,
                "tasks": dl.tasks,
                "MAP_minimizer": MAP_minimizer,
            }

            set_config(dl)
            return MAP_job_desc

        return MAPScheduler(**extract())