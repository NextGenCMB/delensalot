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

## TODO don't like this import here. Not sure how to remove
from delensalot.core.cg import cd_solve

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.sims.sims_lib import Simhandler

from delensalot.utils import cli, camb_clfile, load_file
from delensalot.utility.utils_hp import gauss_beam

from delensalot.core.QE import field as QE_field
from delensalot.core.MAP import field as MAP_field, operator

from delensalot.core.iterator import steps
from delensalot.core.handler import OBD_builder, Sim_generator, QE_lr, QE_lr_new, MAP_lr, MAP_lr_operator, Map_delenser, Phi_analyser

from delensalot.config.visitor import transform, transform3d
from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm, DLENSALOT_Concept
from collections import OrderedDict

from itertools import chain, combinations
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

class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """
    # @log_on_start(logging.DEBUG, "process_Simulation() started")
    # @log_on_end(logging.DEBUG, "process_Simulation() finished")
    def process_Simulation(dl, si, cf):
        # NOTE I want everything to be 'controlled' by fid_info['sec_components'],
        # so if something is not in there, sec_info, operator_info, etc. need to be updated
        buffer_secinfo = copy.deepcopy(si.sec_info)
        buffer_operatorinfo = copy.deepcopy(si.operator_info)
        allowed_secs = si.fid_info['sec_components'].keys()
        for key, val in si.sec_info.items():
            if key not in allowed_secs:
                buffer_secinfo.pop(key)
            elif val['component'] != si.fid_info['sec_components'][key]:
                    comps_ = [c[0] for c in atleast_1d(si.fid_info['sec_components'][key])]
                    buffer_secinfo[key]['component'] = comps_
        for key, val in si.operator_info.items():
            if key not in allowed_secs:
                buffer_operatorinfo.pop(key)
            elif val['component'] != si.fid_info['sec_components'][key]:
                    comps_ = [c[0] for c in atleast_1d(si.fid_info['sec_components'][key])]
                    buffer_operatorinfo[key]['component'] = comps_

        si.sec_info = buffer_secinfo
        si.operator_info = buffer_operatorinfo
        dl.simulationdata = Simhandler(**si.__dict__)
        if 'cls_lib' in dl.simulationdata.__dict__:
            # FIXME this doesnt look safe, order of a dict is not guaranteed
            # FIXME if obs maps are given, sumulationdata.fid_info is not the correct thing to use here. to make it work it needs to be updated with the info from analysis.secondaries
            # print(dl.cls_len.keys())    
            dl.CLfids = {
                secclk: {comp: va for comp, va in zip(secclv, val)}
                for (secclk, secclv), val in zip(dl.simulationdata.fid_info['sec_components'].items(), dl.simulationdata.get_sim_fidsec(0, secondary=None, components=None))
            }
        else:
            dl.CLfids = {sec: {} for sec in allowed_secs}
        # FIXME this is a hack, just to add a secondary to reconstruction although its not in the simulationdata. (i'll need this in the future anyway, in case obsdata is provided)
        # for sec in cf.analysis.secondaries:


    # @log_on_start(logging.DEBUG, "_process_Analysis() started")
    # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
    def process_Analysis(dl, an, cf):
        if loglevel <= 20:
            dl.verbose = True
        elif loglevel >= 30:
            dl.verbose = False
        dl.dlm_mod_bool = cf.madel.dlm_mod
        dl.beam = an.beam
        dl.mask_fn = an.mask
        dl.k = an.key
        dl.lmin_teb = an.lmin_teb
        dl.version = an.version
        dl.simidxs = an.simidxs

        # FIXME makes this a clean implementation
        dl.simidxs_mf = np.array(an.simidxs_mf) if dl.version != 'noMF' else np.array([])
        dl.Nmf = 10000 if cf.itrec.mfvar.startswith('/') else (0 if dl.version == 'noMF' else len(dl.simidxs_mf))
        
        dl.TEMP = transform(cf, l2T_Transformer())
        dl.cls_unl = camb_clfile(an.cls_unl)
        dl.cls_len = camb_clfile(an.cls_len)
        dl.Lmin = an.Lmin
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
        dl.lm_max_ivf = an.lm_max_ivf

        beam_factor = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0])
        lmin_mask = np.arange(dl.lm_max_ivf[0] + 1)[:, None] >= dl.lmin_teb
        if an.transfunction_desc == 'gauss_no_pixwin':
            transf =(beam_factor[:, None] * lmin_mask)
        elif an.transfunction_desc == 'gauss_with_pixwin':
            assert cf.noisemodel.geominfo[0] == 'healpix', 'implement non-healpix pixelwindow function'
            pixwin_factor = hp.pixwin(cf.noisemodel.geominfo[1]['nside'], lmax=dl.lm_max_ivf[0])
            transf = (beam_factor[:, None] * pixwin_factor[:, None] * lmin_mask)
        dl.ttebl = dict(zip('teb', transf.T))


    # @log_on_start(logging.DEBUG, "_process_Meta() started")
    # @log_on_end(logging.DEBUG, "_process_Meta() finished")
    def process_Meta(dl, me, cf):
        dl.dversion = me.version


    def process_Computing(dl, co, cf):
        dl.tr = co.OMP_NUM_THREADS
        os.environ["OMP_NUM_THREADS"] = str(dl.tr)


class l2T_Transformer:
    # TODO this could use refactoring. Better name generation
    """global access for custom TEMP directory name, so that any job stores the data at the same place.
    """

    # # @log_on_start(logging.DEBUG, "build() started")
    # # @log_on_end(logging.DEBUG, "build() finished")
    def build(self, cf):
        if cf.job.jobs == ['build_OBD']:
            return cf.obd.libdir
        else:       
            if cf.analysis.TEMP_suffix != '':
                _suffix = cf.analysis.TEMP_suffix
            _suffix += '_OBD' if cf.noisemodel.OBD == 'OBD' else '_lminB'+str(cf.analysis.lmin_teb[2])+"_simdata"+str("".join(cf.simulationdata.fid_info['sec_components']))
            TEMP =  opj(os.environ['SCRATCH'], 'analysis', _suffix)

            return TEMP


    def ofj(desc, kwargs):
        for key, val in kwargs.items():
            buff = desc
            if type(val) == str:
                buff += "_{}{}".format(key, val)
            elif type(val) == int:
                buff += "_{}{:03d}".format(key, val)
            elif type(val) == float:
                buff += "_{}{:.3f}".format(key, val)

        return buff


class l2OBD_Transformer:
    """Transformer for generating a delensalot model for the calculation of the OBD matrix
    """

    # @log_on_start(logging.DEBUG, "get_nlev() started")
    # @log_on_end(logging.DEBUG, "get_nlev() finished")
    def get_nlev(cf):
        return cf.noisemodel.nlev


    # @log_on_start(logging.DEBUG, "get_nivt_desc() started")
    # @log_on_end(logging.DEBUG, "get_nivt_desc() finished")
    def get_nivt_desc(cf, dl):
        nlev = l2OBD_Transformer.get_nlev(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf, dl)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        if cf.noisemodel.nivt_map is None:
            if dl.nivjob_geominfo[0] == 'healpix':
                niv_desc = [np.array([hp.nside2pixarea(dl.nivjob_geominfo[1]['nside'], degrees=True) * 60 ** 2 / nlev['T'] ** 2])/noisemodel_norm] + masks
            else:
                assert 0, 'needs testing, please choose Healpix geom for nivjob for now'
                vamin =  4*np.pi * (180/np.pi)**2 / get_geom(cf.itrec.lenjob_geominfo).npix()
                niv_desc = [np.array([vamin * 60 ** 2 / nlev['T'] ** 2])/noisemodel_norm] + masks
        else:
            niv = np.load(cf.noisemodel.nivt_map)
            niv_desc = [niv] + masks
        return niv_desc


    # @log_on_start(logging.DEBUG, "get_nivp_desc() started")
    # @log_on_end(logging.DEBUG, "get_nivp_desc() finished")
    def get_nivp_desc(cf, dl):
        nlev = l2OBD_Transformer.get_nlev(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf, dl)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        if cf.noisemodel.nivp_map is None:
            if dl.nivjob_geominfo[0] == 'healpix':
                niv_desc = [[np.array([hp.nside2pixarea(dl.nivjob_geominfo[1]['nside'], degrees=True) * 60 ** 2 / nlev['P'] ** 2])/noisemodel_norm] + masks]
            else:
                assert 0, 'needs testing, pleasechoose Healpix geom for nivjob for now'
                vamin =  4*np.pi * (180/np.pi)**2 / get_geom(cf.itrec.lenjob_geominfo).npix()
                niv_desc = [[np.array([vamin * 60 ** 2 / nlev['P'] ** 2])/noisemodel_norm] + masks]
        else:
            niv = np.load(cf.noisemodel.nivp_map)
            niv_desc = [[niv] + masks]
        return niv_desc


    # @log_on_start(logging.DEBUG, "get_masks() started")
    # @log_on_end(logging.DEBUG, "get_masks() finished")
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
    def build_generate_sim(self, cf):
        def extract():
            def _process_Analysis(dl, an, cf):
                dl.k = an.key
                dl.lmin_teb = an.lmin_teb
                dl.version = an.version
                dl.simidxs = an.simidxs
                dl.simidxs_mf = np.array(an.simidxs_mf) # if dl.version != 'noMF' else np.array([])

                dl.TEMP = transform(cf, l2T_Transformer())

            dl = DLENSALOT_Concept()
            _process_Analysis(dl, cf.analysis, cf)
            l2base_Transformer.process_Meta(dl, cf.meta, cf)
            dl.libdir_suffix = cf.simulationdata.obs_info['noise_info']['libdir_suffix']
            l2base_Transformer.process_Simulation(dl, cf.simulationdata, cf)
            # dl.simulationdata = Simhandler(**cf.simulationdata.__dict__)
            return dl
        return Sim_generator(extract())


    def build_QE_lensrec_new(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):
                def _process_Meta(dl, me):
                    l2base_Transformer.process_Meta(dl, me, cf)
                def _process_Computing(dl, co):
                    l2base_Transformer.process_Computing(dl, co, cf)
                def _process_Analysis(dl, an):
                    l2base_Transformer.process_Analysis(dl, an, cf)
                def _process_Noisemodel(dl, nm):
                    dl.sky_coverage = nm.sky_coverage
                    dl.nivjob_geomlib = get_geom(nm.geominfo).restrict(*np.arccos(dl.zbounds[::-1]), northsouth_sym=False)
                    dl.nivjob_geominfo = nm.geominfo
                    if dl.sky_coverage == 'masked':
                        dl.rhits_normalised = nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_nivp_desc(cf, dl)[0][1])  # Expensive, could affect future fsky calc
                    else:
                        dl.fsky = 1.0
                    dl.spectrum_type = nm.spectrum_type
                    dl.OBD = nm.OBD
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    
                    def compute_spectrum(cls_key, nlev_key, transf_key, spectrum_type):
                        cls = dl.cls_len if spectrum_type == 'len' else dl.cls_unl
                        return cli(cls[cls_key][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev[nlev_key])**2 * cli(dl.ttebl[transf_key] ** 2)) * (dl.ttebl[transf_key] > 0)
                    # Isotropic approximation to the filtering (using 'len' for lensed spectra)
                    dl.ftebl_len = {key: compute_spectrum(cls_key, nlev_key, transf_key, 'len') 
                                    for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}
                    # Same using unlensed spectra (using 'unl' for unlensed spectra)
                    dl.ftebl_unl = {key: compute_spectrum(cls_key, nlev_key, transf_key, 'unl') 
                                    for key, (cls_key, nlev_key, transf_key) in zip('teb', [('tt', 'T', 't'), ('ee', 'P', 'e'), ('bb', 'P', 'b')])}
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                def _process_Simulation(dl, si):
                    l2base_Transformer.process_Simulation(dl, si, cf)
                def _process_Qerec(dl, qe):
                    qe_tasks_sorted = ['calc_fields', 'calc_meanfields', 'calc_templates'] if qe.subtract_QE_meanfield else ['calc_fields', 'calc_templates']
                    dl.qe_tasks = [task for task in qe_tasks_sorted if task in qe.tasks]       
                    dl.qe_filter_directional = qe.filter_directional

                    dl.subtract_QE_meanfield = qe.subtract_QE_meanfield
                    dl.estimator_type = qe.estimator_type
                    ## FIXME cg chain currently only works with healpix geometry
                    dl.QE_cg_tol = qe.cg_tol
                    dl.chain_model = qe.chain
                    dl.chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    dl.chain_descr = lambda p2, p5 : [[0, ["diag_cl"], p2, dl.nivjob_geominfo[1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]]
                    dl.cl_analysis = qe.cl_analysis
                    # NOTE I need this for the template generation, but I don't want to pull this from itrec
                    dl.lenjob_geominfo = cf.itrec.lenjob_geominfo
                    dl.lenjob_geomlib = get_geom(cf.itrec.lenjob_geominfo)
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*cf.analysis.secondaries['lensing']['lm_max'])), cf.analysis.secondaries['lensing']['lm_max'][1], numthreads=dl.tr, verbosity=False, epsilon=cf.itrec.epsilon)

                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.simulationdata)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

            dl = DLENSALOT_Concept()
            _process_components(dl)

            if len(cf.analysis.key)==1:
                keystring = cf.analysis.key
            elif "_" in cf.analysis.key:
                keystring = cf.analysis.key.split('_')[-1]
            else:
                keystring = cf.analysis.key[1:]

            QE_filterqest_desc = {
                "estimator_key": cf.analysis.key, #FIXME this should be a different value for each secondary
                "estimator_type": dl.estimator_type, #FIXME this should be a different value for each secondary
                "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring),
                "simulationdata": Simhandler(**cf.simulationdata.__dict__),
                "nivjob_geominfo": cf.noisemodel.geominfo,
                "nivt_desc": dl.nivt_desc,
                "nivp_desc": dl.nivp_desc,
                "qe_filter_directional": cf.qerec.filter_directional,
                "cls_unl": dl.cls_unl,
                "cls_len": dl.cls_len,
                "ttebl": dl.ttebl,
                "ftebl_len": dl.ftebl_len,
                "ftebl_unl":  dl.ftebl_unl,
                "lm_max_ivf": dl.lm_max_ivf,
                "lm_max_qlm": cf.analysis.secondaries['lensing']['lm_max'], #FIXME this should be a different value for each secondary
                "lm_max_unl": cf.itrec.lm_max_unl,
                "lm_max_len": dl.lm_max_ivf,
                "version": dl.version,
                "zbounds": dl.zbounds,
                "obd_libdir": dl.obd_libdir,
                "obd_rescale": dl.obd_rescale,
                "sht_threads": dl.tr,
                "QE_cg_tol": dl.QE_cg_tol,
                "beam": dl.beam,
                "OBD": dl.OBD,
                "lmin_teb": dl.lmin_teb,
                "chain_descr": dl.chain_descr,
            }

            QE_secs_descs = {sec: {
                    "ID": sec,
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring, 'estimates', sec),
                    'lm_max': val['lm_max'],
                    'component': val['component'],
                    'CLfids': dl.CLfids[sec],
                } for sec, val in cf.analysis.secondaries.items()}
            QE_secs = {name: QE_field.secondary(field_desc) for name, field_desc in QE_secs_descs.items()}
            
            # this is template fields
            # FIXME template fields 'components' are potentially not needed, as template will be generated using all components simulatenously
            QE_template_descs = {sec: {
                    "ID": sec,
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring, 'templates', sec),
                    'lm_max': val['lm_max'],
                    'component': all_combinations(val['component']),
                } for sec, val in cf.analysis.secondaries.items()}

            if len(cf.analysis.secondaries) > 1:
                QE_templates = [QE_field.template(field_desc) for name, field_desc in QE_template_descs.items()]
                template_operator_descs = cf.qerec.template_operator_info
                for sec, val in cf.qerec.template_operator_info.items():
                    template_operator_descs[sec].update({
                        "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring, 'templates', sec),
                        "field_fns": QE_secs["lensing"].klm_fns,
                        "ffi": dl.ffi,
                        })
                template_operators = { # NOTE joint can be build by combinatorics
                    "lensing": operator.lensing(template_operator_descs["lensing"]),
                    "birefringence": operator.birefringence(template_operator_descs["birefringence"]),
                }
                # FIXME do this later
                template_desc = {
                    'fields': QE_templates,
                    'operators': template_operators,
                }
            else:
                template_desc = None

            # TODO can I simplify this?
            dl.cls_unl.update({comp: val for comp_dict in dl.CLfids.values() for comp, val in comp_dict.items()})

            QE_searchs_desc = {sec: {
                    "ID": sec,
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', keystring, sec),
                    "QE_filterqest_desc": QE_filterqest_desc,
                    "secondary": QE_secs[sec],
                    "estimator_key": cf.analysis.key, # FIXME
                    "cls_len": dl.cls_len,
                    "cls_unl": dl.cls_unl,
                    "simidxs": cf.analysis.simidxs,
                    "simidxs_mf": dl.simidxs_mf,
                    "subtract_meanfield": dl.subtract_QE_meanfield,
                } for sec in cf.analysis.secondaries.keys()}
            
            QE_handler_desc = {
                "template_info": template_desc,
                "simidxs": cf.analysis.simidxs,
                "simidxs_mf": dl.simidxs_mf,
                "QE_tasks": dl.qe_tasks,
                "simulationdata": QE_filterqest_desc['simulationdata'],
            }

            dl.QE_searchs_desc = QE_searchs_desc
            dl.QE_handler_desc = QE_handler_desc
            return dl
        return QE_lr_new(extract())


    def build_MAP_lensrec_operator(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):

                def _process_Itrec(dl, it):
                    dl.it_tasks = it.tasks
                    dl.lm_max_unl = it.lm_max_unl
                    dl.epsilon = it.epsilon
                    # chain
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    dl.it_chain_descr = lambda p2, p5 : [[0, ["diag_cl"], p2, 2048, np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]]
                    
                    dl.lenjob_geominfo = it.lenjob_geominfo
                    dl.lenjob_geomlib = get_geom(it.lenjob_geominfo)
            
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.it_filter_directional = it.filter_directional
                    dl.itmax = it.itmax
                    dl.iterator_typ = it.iterator_typ

                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    dl.soltn_cond = it.soltn_cond
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = cf.analysis.secondaries['lensing']['lm_max'][0]
                        dl.stepper_model.mmax_qlm = cf.analysis.secondaries['lensing']['lm_max'][1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                        # dl.stepper = steps.nrstep(dl.LM_max[0], dl.LM_max[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*cf.analysis.secondaries['lensing']['lm_max'])), cf.analysis.secondaries['lensing']['lm_max'][1], numthreads=dl.tr, verbosity=False, epsilon=dl.epsilon)
                
                _process_Itrec(dl, cf.itrec)

            dl = DLENSALOT_Concept()
            dl = self.build_QE_lensrec_new(cf)
            QE_searchs = dl.QE_searchs
            _process_components(dl)

            MAP_libdir_prefix = opj(transform(cf, l2T_Transformer()), 'MAP'+"_"+"".join(cf.analysis.secondaries.keys())+"_"+get_hashcode(OrderedDict(cf.analysis.secondaries)), '{estimator_key}'.format(estimator_key=cf.analysis.key.split('_')[-1]))

            # input: all kwargs needed to build the MAP fields
            MAP_secondaries = {sec: MAP_field.secondary({
                    "ID": sec,
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    'lm_max': cf.analysis.secondaries[sec]['lm_max'],
                    "component": val['component'],
                    'CLfids': dl.CLfids[sec],
                    'fns': {comp: f'klm_{comp}_simidx{{idx}}_it{{it}}' for comp in val['component']}, # This could be hardcoded, but I want to keep it flexible
                    'increment_fns': {comp: f'kinclm_{comp}_simidx{{idx}}_it{{it}}' for comp in val['component']},
                    'meanfield_fns': {comp: f'kmflm_{comp}_simidx{{idx}}_it{{it}}' for comp in val['component']},
                }) for sec, val in cf.analysis.secondaries.items()}

            # input: all kwargs needed to build the MAP search
            _MAP_operators_desc = {}
            filter_operators = []
            gradients_operators = {}
            if 'lensing' in cf.analysis.secondaries:
                _MAP_operators_desc['lensing'] = {
                    'lm_max': dl.lm_max_ivf,
                    "LM_max": cf.analysis.secondaries['lensing']['lm_max'],
                    "Lmin": dl.Lmin,
                    "perturbative": False,
                    "component": [item for  item in cf.analysis.secondaries['lensing']['component']],
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    'field_fns': MAP_secondaries["lensing"].fns,
                    "ffi": dl.ffi,}
                _MAP_operators_desc['spin_raise'] = {
                    'lm_max': dl.lm_max_unl,}
                filter_operators.append(operator.lensing(_MAP_operators_desc['lensing']))
                gradients_operators['lensing'] = operator.joint([operator.spin_raise(_MAP_operators_desc['spin_raise']), *filter_operators])

            if 'birefringence' in cf.analysis.secondaries:
                _MAP_operators_desc['birefringence'] = {
                    'lm_max': dl.lm_max_ivf,
                    "Lmin": dl.Lmin,
                    "component": ['f'],
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    'field_fns': MAP_secondaries['birefringence'].fns,}
                _MAP_operators_desc['multiply'] = {
                    'factor': -1j,}
                filter_operators.append(operator.birefringence(_MAP_operators_desc['birefringence']))
                gradients_operators['birefringence'] = operator.joint([operator.multiply(_MAP_operators_desc['multiply']), *filter_operators])

            ivf_operator = operator.ivf_operator(filter_operators)
            wf_operator = operator.wf_operator(filter_operators) #TODO this is ivf_operator*ivf_operator^dagger, could be implemented via ivf.

            def chh(CL, lmax, scale='k'):
                return CL[:lmax+1] * (0.5 * np.arange(lmax+1) * np.arange(1, lmax+2))**2
                
            gfield_descs = [{
                "ID": gradient_name,
                "libdir": opj(MAP_libdir_prefix, 'gradients'),
                "libdir_prior": opj(MAP_libdir_prefix, 'estimates'),
                "lm_max": cf.analysis.secondaries[gradient_name]['lm_max'],
                "meanfield_fns": f'mf_glm_{gradient_name}_simidx{{idx}}_it{{it}}',
                "quad_fns": f'quad_glm_{gradient_name}_simidx{{idx}}_it{{it}}',
                "prior_fns": 'klm_{component}_simidx{idx}_it{it}', # prior is just field, and then we do a simple divide by spectrum (almxfl)
                "total_increment_fns": f'ginclm_{gradient_name}_simidx{{idx}}_it{{it}}',    
                "total_fns": f'gtotlm_{gradient_name}_simidx{{idx}}_it{{it}}',    
                "chh": {comp: chh(dl.CLfids[gradient_name][comp*2], lmax=cf.analysis.secondaries[gradient_name]['lm_max'][0]) for comp in cf.analysis.secondaries[gradient_name]['component']},
                "component": [item for  item in cf.analysis.secondaries['lensing']['component']] if gradient_name == 'lensing' else ['f'],
            } for gradient_name, gradient_operator in gradients_operators.items()]
            MAP_gfields = {gfield_desc["ID"]: MAP_field.gradient(gfield_desc) for gfield_desc in gfield_descs}
            gradient_descs = {}
            for gradient_name, gradient_operator in gradients_operators.items():
                gradient_descs.update({ gradient_name: {
                    "ID": gradient_name,
                    "secondary": MAP_secondaries[gradient_name],
                    "gfield": MAP_gfields[gradient_name],
                    "noisemodel_coverage": dl.it_filter_directional,
                    "estimator_key":  cf.analysis.key,
                    "simulationdata": dl.simulationdata,
                    "lm_max_ivf": dl.lm_max_ivf,
                    "lm_max_unl": dl.lm_max_unl,
                    "LM_max": cf.analysis.secondaries[gradient_name]['lm_max'],
                    'itmax': dl.itmax,
                    "gradient_operator": gradient_operator,
                    "ffi": dl.ffi,
                }})

            MAP_ivffilter_field_desc = {
                "ID": "ivf",
                "libdir": opj(MAP_libdir_prefix, 'filter'),
                "lm_max": dl.lm_max_ivf,
                "component": 1,
                "fns": "ivf_simidx{idx}_it{it}",
            }
            MAP_WFfilter_field_desc = {
                "ID": "WF",
                "libdir": opj(MAP_libdir_prefix, 'filter'),
                "lm_max": dl.lm_max_ivf,
                "component": 1,
                "fns": "WF_simidx{idx}_it{it}",
            }

            MAP_ivf_desc = {
                "ID": "ivf",
                'ivf_operator': ivf_operator,
                "ivf_field": MAP_field.filter(MAP_ivffilter_field_desc),
                'beam': operator.beam({"beamwidth": cf.analysis.beam, "lm_max":dl.lm_max_ivf}),
                "ttebl": dl.ttebl,
                "lm_max_ivf": dl.lm_max_ivf,
                "nlev": dl.nlev,
            }
            MAP_wf_desc = {
                "ID": "polarization",
                'wf_operator': wf_operator,
                "wf_field": MAP_field.filter(MAP_WFfilter_field_desc),
                'beam': operator.beam({"beamwidth": cf.analysis.beam, "lm_max":dl.lm_max_ivf}),
                'nlev': dl.nlev,
                "chain_descr": dl.it_chain_descr(dl.lm_max_unl[0], dl.it_cg_tol(0)),
                "ttebl": dl.ttebl,
                "cls_filt": dl.cls_unl,
                "lm_max_ivf": dl.lm_max_ivf,
                "lm_max_unl": dl.lm_max_unl,
                "nlev": dl.nlev,
                "ffi": dl.ffi,
            }
            
            template_desc = copy.deepcopy(dl.QE_handler_desc["template_info"])
            # template_descs = {
            #     "libdir": opj(MAP_libdir_prefix, 'templates'),
            #     "template_operators": template_operators,
            # }

            lp1 = 2 * np.arange(3000 + 1) + 1
            n = len([v for val in cf.analysis.secondaries.values() for v in val['component']])
            def dotop(glms1, glms2):
                ret = 0.
                N = 0
                for lmax, mmax in zip([cf.analysis.secondaries[gradient_name]['lm_max'][0] for n in range(n)], [cf.analysis.secondaries[gradient_name]['lm_max'][1] for n in range(n)]):
                    siz = hp.Alm.getsize(lmax, mmax)
                    cl = hp.alm2cl(glms1[N:N+siz], glms2[N:N+siz], None, mmax, None)
                    ret += np.sum(cl * (2 * np.arange(len(cl)) + 1))
                    N += siz
                return ret
            curvature_desc = {
                "ID": "curvature",
                "field": MAP_field.curvature(
                    {"ID": "curvature",
                    "libdir": opj(MAP_libdir_prefix, 'curvature'),
                    "fns": {'yk': f"diff_grad1d_simidx{{idx}}_it{{it}}m{{itm1}}",
                            'sk': f"incr_grad1d_simidx{{idx}}_it{{it}}m{{itm1}}",
                    }
                }),
                "bfgs_desc": {'dot_op': dotop, # lambda rlm1, rlm2: np.sum(lp1 * hp.alm2cl(rlm1, rlm2)),
                },
            }
            desc = {
                "itmax": dl.itmax,
            }
            MAP_searchs_desc = {
                'gradient_descs': gradient_descs,
                'MAP_secondaries': MAP_secondaries,
                'filter_desc': {'ivf': MAP_ivf_desc, 'wf': MAP_wf_desc},
                'curvature_desc': curvature_desc,
                "desc" : desc,
                "template_descs": template_desc,
            }
            MAP_handler_desc = {
                'simulationdata': dl.simulationdata,
                "simidxs": cf.analysis.simidxs,
                "simidxs_mf": dl.simidxs_mf,
                "QE_searchs": QE_searchs,
                "it_tasks": dl.it_tasks,
            }
            dl.MAP_handler_desc, dl.MAP_searchs_desc = MAP_handler_desc, MAP_searchs_desc
            return dl

        return MAP_lr_operator(extract())


    def build_QE_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        # @log_on_start(logging.DEBUG, "extract() started")
        # @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                # @log_on_start(logging.DEBUG, "_process_Meta() started")
                # @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                # @log_on_start(logging.DEBUG, "_process_Computing() started")
                # @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                # @log_on_start(logging.DEBUG, "_process_Analysis() started")
                # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                # @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                # @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.sky_coverage = nm.sky_coverage
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False)
                    if dl.sky_coverage == 'masked':
                        dl.rhits_normalised = nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_nivp_desc(cf, dl)[0][1]) ## calculating fsky, but quite expensive. and if nivp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)

                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)


                # @log_on_start(logging.DEBUG, "_process_OBD() started")
                # @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale


                # @log_on_start(logging.DEBUG, "_process_Simulation() started")
                # @log_on_end(logging.DEBUG, "_process_Simulation() finished")       
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.obs_info['noise_info']['libdir_suffix']
                    l2base_Transformer.process_Simulation(dl, si, cf)


                # @log_on_start(logging.DEBUG, "_process_Qerec() started")
                # @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):
                    dl.blt_pert = qe.blt_pert
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    if dl.QE_subtract_meanfield:
                        qe_tasks_sorted = ['calc_phi', 'calc_meanfield', 'calc_blt']
                    else:
                        qe_tasks_sorted = ['calc_phi', 'calc_blt']
                    qe_tasks_extracted = []
                    for taski, task in enumerate(qe_tasks_sorted):
                        if task in qe.tasks:
                            qe_tasks_extracted.append(task)
                        else:
                            break
                    dl.qe_tasks = qe_tasks_extracted
                        
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type

                    ## FIXME cg chain currently only works with healpix geometry
                    dl.cg_tol = qe.cg_tol
                    if qe.chain == None:
                        dl.chain_descr = lambda a,b: None
                        dl.chain_model = dl.chain_descr
                    else:
                        dl.chain_model = qe.chain
                        dl.chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                        if dl.chain_model.p6 == 'tr_cg':
                            _p6 = cd_solve.tr_cg
                        if dl.chain_model.p7 == 'cache_mem':
                            _p7 = cd_solve.cache_mem()
                        dl.chain_descr = lambda p2, p5 : [
                            [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]
                    dl.qe_filter_directional = qe.filter_directional
                    dl.cl_analysis = qe.cl_analysis


                # @log_on_start(logging.DEBUG, "_process_Itrec() started")
                # @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    dl.it_tasks = it.tasks
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    dl.epsilon = it.epsilon
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    
                    dl.lenjob_geominfo = it.lenjob_geominfo
                    dl.lenjob_geomlib = get_geom(it.lenjob_geominfo)
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)

                    # TODO this needs cleaner implementation
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol*0.1
                    dl.it_filter_directional = it.filter_directional
                    dl.itmax = it.itmax
                    dl.iterator_typ = it.iterator_typ

                    # TODO this needs cleaner implementation
                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    dl.soltn_cond = it.soltn_cond

                    # TODO this needs cleaner implementation
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = dl.lm_max_qlm[0]
                        dl.stepper_model.mmax_qlm = dl.lm_max_qlm[1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)

                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=dl.verbose, epsilon=dl.epsilon)


                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.simulationdata)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                # TODO this needs cleaner implementation. 
                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp[0]:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                elif cf.analysis.cpp.endswith('dat'):
                    # assume its a camb-like file
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] 
                elif os.path.exists(os.path.dirname(cf.analysis.cpp)):
                    # FIXME this implicitly assumes that all cpp.npy comes as convergence
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                    LL = np.arange(0,dl.lm_max_qlm[0] + 1,1)
                    k2p = lambda x: np.nan_to_num(x/(LL*(LL+1))**2/(2*np.pi))
                    dl.cpp = k2p(dl.cpp)
                    
                dl.cpp[:dl.Lmin] *= 0.

            dl = DLENSALOT_Concept()
            _process_components(dl)
            ## TODO. Current solution to fake an iteration handler for QE to calc blt is to initialize one here.
            ## In the future, I want to remove get_template_blm from the iteration_handler, at least for QE.
            ## this would then also simplify the QE transformer a lot (no MAP dependency anymore)
            if 'calc_blt' in dl.qe_tasks or 'calc_blt' in dl.it_tasks:
                dl.MAP_job = transform3d(cf, 'MAP_lensrec', l2delensalotjob_Transformer())
            return dl

        return QE_lr(extract())


    def build_MAP_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        # @log_on_start(logging.DEBUG, "extract() started")
        # @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                # @log_on_start(logging.DEBUG, "_process_Meta() started")
                # @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                # @log_on_start(logging.DEBUG, "_process_Computing() started")
                # @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                # @log_on_start(logging.DEBUG, "_process_Analysis() started")
                # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                # @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                # @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.sky_coverage = nm.sky_coverage
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    if dl.sky_coverage == 'masked':
                        dl.rhits_normalised = nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_nivp_desc(cf, dl)[0][1]) ## calculating fsky, but quite expensive. and if nivp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    

                # @log_on_start(logging.DEBUG, "_process_OBD() started")
                # @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale


                # @log_on_start(logging.DEBUG, "_process_Simulation() started")
                # @log_on_end(logging.DEBUG, "_process_Simulation() finished")       
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.obs_info['noise_info']['libdir_suffix']
                    l2base_Transformer.process_Simulation(dl, si, cf)


                # @log_on_start(logging.DEBUG, "_process_Qerec() started")
                # @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):

                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    dl.blt_pert = qe.blt_pert
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    if dl.QE_subtract_meanfield:
                        qe_tasks_sorted = ['calc_phi', 'calc_meanfield', 'calc_blt']
                    else:
                        qe_tasks_sorted = ['calc_phi', 'calc_blt']
                    qe_tasks_extracted = []
                    for taski, task in enumerate(qe_tasks_sorted):
                        if task in qe.tasks:
                            qe_tasks_extracted.append(task)
                        else:
                            break
                    dl.qe_tasks = qe_tasks_extracted
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type
                    dl.cg_tol = qe.cg_tol

                    if qe.chain == None:
                        dl.chain_descr = lambda a,b: None
                        dl.chain_model = dl.chain_descr
                    else:
                        dl.chain_model = qe.chain
                        dl.chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                        
                        if dl.chain_model.p6 == 'tr_cg':
                            _p6 = cd_solve.tr_cg
                        if dl.chain_model.p7 == 'cache_mem':
                            _p7 = cd_solve.cache_mem()
                        dl.chain_descr = lambda p2, p5 : [
                            [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

                    dl.qe_filter_directional = qe.filter_directional
                    dl.cl_analysis = qe.cl_analysis


                # @log_on_start(logging.DEBUG, "_process_Itrec() started")
                # @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    dl.it_tasks = it.tasks
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    dl.epsilon = it.epsilon
                    # chain
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    
                    dl.lenjob_geominfo = it.lenjob_geominfo
                    dl.lenjob_geomlib = get_geom(it.lenjob_geominfo)
            
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.it_filter_directional = it.filter_directional
                    dl.itmax = it.itmax
                    dl.iterator_typ = it.iterator_typ

                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    dl.soltn_cond = it.soltn_cond
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = dl.lm_max_qlm[0]
                        dl.stepper_model.mmax_qlm = dl.lm_max_qlm[1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                        # dl.stepper = steps.nrstep(dl.lm_max_qlm[0], dl.lm_max_qlm[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=dl.verbose, epsilon=dl.epsilon)
                
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.simulationdata)

                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                elif cf.analysis.cpp.endswith('dat'):
                    # assume its a camb-like file
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] 
                elif os.path.exists(os.path.dirname(cf.analysis.cpp)):
                    # FIXME this implicitly assumes that all cpp.npy comes as convergence
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                    LL = np.arange(0,dl.lm_max_qlm[0] + 1,1)
                    k2p = lambda x: np.nan_to_num(x/(LL*(LL+1))**2/(2*np.pi))
                    dl.cpp = k2p(dl.cpp)
                dl.cpp[:dl.Lmin] *= 0.

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        return MAP_lr(extract())


    def build_OBD_builder(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        # @log_on_start(logging.DEBUG, "extract() started")
        # @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                # @log_on_start(logging.DEBUG, "_process_Computing() started")
                # @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = int(os.environ.get('OMP_NUM_THREADS', co.OMP_NUM_THREADS))


                # @log_on_start(logging.DEBUG, "_process_Analysis() started")
                # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
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


                # @log_on_start(logging.DEBUG, "_process_OBD() started")
                # @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.libdir = od.libdir if type(od.libdir) == str else 'nopath'
                    dl.nlev_dep = od.nlev_dep
                    dl.rescale = od.rescale

                    if os.path.isfile(opj(dl.libdir,'tniti.npy')):
                        # TODO need to test if it is the right tniti.npy
                        # TODO dont exit, rather skip job
                        log.warning("tniti.npy in destination dir {} already exists.".format(dl.libdir))
                        log.warning("Please check your settings.")


                # @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                # @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.lmin_b = dl.lmin_teb[2]
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    dl.masks, dl.rhits_map = l2OBD_Transformer.get_masks(cf, dl)
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    
                dl.TEMP = transform(cf, l2T_Transformer())

                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_OBD(dl, cf.obd)
                
                return dl

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        return OBD_builder(extract())


    def build_delenser(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        # @log_on_start(logging.DEBUG, "extract() started")
        # @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                # @log_on_start(logging.DEBUG, "_process_Meta() started")
                # @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                # @log_on_start(logging.DEBUG, "_process_Computing() started")
                # @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)
                    log.debug("OMP_NUM_THREADS: {} and {}".format(dl.tr, os.environ.get('OMP_NUM_THREADS')))


                # @log_on_start(logging.DEBUG, "_process_Analysis() started")
                # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    # super(l2base_Transformer, self)
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                # @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                # @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    # thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    ## this is for delensing, and pospace doesn't support truncated maps, therefore no restrict here
                    # dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)


                # @log_on_start(logging.DEBUG, "_process_Qerec() started")
                # @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):
                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    dl.blt_pert = qe.blt_pert
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type
                    dl.qe_filter_directional = qe.filter_directional


                # @log_on_start(logging.DEBUG, "_process_Itrec() started")
                # @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1
                    dl.it_filter_directional = it.filter_directional
                    dl.itmax = it.itmax
                    dl.iterator_typ = it.iterator_typ
                    dl.soltn_cond = it.soltn_cond
             

                def _process_Madel(dl, ma):
                    dl.data_from_CFS = ma.data_from_CFS
                    dl.its = [0] if ma.iterations == [] else ma.iterations
                    dl.TEMP = transform(cf, l2T_Transformer())
                    dl.libdir_iterators = lambda qe_key, simidx, version: opj(dl.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
                    dl.analysis_path = dl.TEMP.split('/')[-1]
                    dl.blt_pert = cf.qerec.blt_pert
                    dl.basemap = ma.basemap

                    ## Masking
                    if cf.noisemodel.rhits_normalised is not None:
                        _mask_path = cf.noisemodel.rhits_normalised[0]
                        dl.base_mask = np.nan_to_num(hp.read_map(_mask_path))
                    else:
                        dl.base_mask = np.ones(shape=hp.nside2npix(cf.noisemodel.geominfo[1]['nside']))
                    noisemodel_rhits_map = df.get_nlev_mask(np.inf, dl.base_mask)
                    if ma.nlevels == None or ma.nlevels == [] or ma.nlevels == False:
                        dl.nlevels = np.array([np.inf])
                    else:
                        dl.nlevels = ma.nlevels
                    dl.masks = {'nlevel': {nlevel: [] for nlevel in dl.nlevels}}
                    dl.binmasks = {'nlevel': {nlevel: [] for nlevel in dl.nlevels}}
                    if len(ma.masks_fn) > 0:
                        if os.path.exists(ma.masks_fn[0]):
                            dl.masks_fromfn = [load_file(m) for m in ma.masks_fn]
                            dl.masks.update({'mask': {maskid: [] for maskid in range(len(ma.masks_fn))}})
                            dl.binmasks.update({'mask': {maskid: [] for maskid in range(len(ma.masks_fn))}})
                        else:
                           log.warning("I was expecting a mask from masks_fn, but couldn't find it. {}".format(ma.masks_fn[0]))
                           dl.masks_fromfn = []
                    else:
                        dl.masks_fromfn = []
                    for maskflavour, masks in dl.masks.items():
                        for maskid, mask in masks.items():
                            if maskflavour == 'nlevel':
                                dl.masks[maskflavour][maskid] = df.get_nlev_mask(maskid, noisemodel_rhits_map)
                            else:
                                dl.masks[maskflavour][maskid] = dl.masks_fromfn[maskid]
                            dl.binmasks[maskflavour][maskid] = np.where(dl.masks[maskflavour][maskid]>0,1,0)

                    ## Binning and power spectrum calculator specific preparation
                    if ma.Cl_fid == 'ffp10':
                        dl.cls_unl = camb_clfile(cf.analysis.cls_unl)
                        dl.cls_len = camb_clfile(cf.analysis.cls_len)
                        dl.clg_templ = dl.cls_len['ee']
                        dl.clc_templ = dl.cls_len['bb']
                        dl.clg_templ[0] = 1e-32
                        dl.clg_templ[1] = 1e-32

                    dl.binning = ma.binning
                    if dl.binning == 'binned':
                        dl.lmax = ma.lmax
                        assert dl.lmax >= 1024, "if lmax too small, power spectrum calculation will be biased"
                        dl.lmax_mask = 3*ma.lmax-1
                        dl.edges = ma.edges
                        dl.edges_center = (dl.edges[1:]+dl.edges[:-1])/2.
                        dl.sha_edges = hashlib.sha256()
                        dl.sha_edges.update((str(dl.edges)).encode())
                        dl.dirid = dl.sha_edges.hexdigest()[:4]
                        dl.ct = dl.clc_templ[np.array(dl.edges, dtype=int)] # TODO marginalising over binrange would probably be better

                    elif dl.binning == 'unbinned':
                        dl.lmax = ma.lmax
                        dl.lmax_mask = 3*ma.lmax-1
                        dl.edges = np.arange(0,dl.lmax+2)
                        dl.edges_center = dl.edges[1:]
                        dl.ct = np.ones(shape=len(dl.edges_center))
                        dl.sha_edges = hashlib.sha256()
                        dl.sha_edges.update(('unbinned').encode())
                        dl.dirid = dl.sha_edges.hexdigest()[:4]

                    dl.vers_str = '/{}'.format(dl.version) if dl.version != '' else 'base'

                    if ma.spectrum_calculator == None:
                        log.info("Using Healpy as powerspectrum calculator")
                        dl.cl_calc = hp
                    else:
                        dl.cl_calc = ma.spectrum_calculator       


                def _process_Config(dl, co):
                    dl.outdir_plot_rel = co.outdir_plot_rel
                    dl.outdir_plot_root = co.outdir_plot_root          
                    dl.outdir_plot_abs = opj(dl.outdir_plot_root, dl.outdir_plot_rel)
                    
                    if not os.path.isdir(dl.outdir_plot_abs):
                        os.makedirs(dl.outdir_plot_abs)
                    log.info('Plots will be stored at {}'.format(dl.outdir_plot_abs))


                def _check_powspeccalculator(clc):
                    if dl.binning == 'binned':
                        if 'map2cl_binned' not in clc.__dict__:
                            log.error("Spectrum calculator doesn't provide needed function map2cl_binned() for binned spectrum calculation")
                            sys.exit()
                    elif dl.binning == 'unbinned':
                        if 'map2cl' not in clc.__dict__:
                            if 'anafast' not in clc.__dict__:
                                log.error("Spectrum calculator doesn't provide needed function map2cl() or anafast() for unbinned spectrum calculation")
                                sys.exit()


                dl.blt_pert = cf.qerec.blt_pert
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                dl.libdir_suffix = cf.simulationdata.obs_info['noise_info']['libdir_suffix']
                dl.simulationdata = Simhandler(**cf.simulationdata.__dict__)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Madel(dl, cf.madel)
                _process_Config(dl, cf.config)
                _check_powspeccalculator(dl.cl_calc)

                # Need a few attributes for predictions (like ftebl, lm_max_qlm, ..)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp[0]:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                elif cf.analysis.cpp.endswith('dat'):
                    # assume its a camb-like file
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] 
                elif os.path.exists(os.path.dirname(cf.analysis.cpp)):
                    # FIXME this implicitly assumes that all cpp.npy comes as convergence
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                    LL = np.arange(0,dl.lm_max_qlm[0] + 1,1)
                    k2p = lambda x: np.nan_to_num(x/(LL*(LL+1))**2/(2*np.pi))
                    dl.cpp = k2p(dl.cpp)
                dl.cpp[:dl.Lmin] *= 0.

                return dl

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        return Map_delenser(extract())


    def build_phianalyser(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        # @log_on_start(logging.DEBUG, "extract() started")
        # @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                # @log_on_start(logging.DEBUG, "_process_Meta() started")
                # @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                # @log_on_start(logging.DEBUG, "_process_Computing() started")
                # @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                # @log_on_start(logging.DEBUG, "_process_Analysis() started")
                # @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                # @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                # @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.sky_coverage = nm.sky_coverage
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    if dl.sky_coverage == 'masked':
                        dl.rhits_normalised = nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_nivp_desc(cf, dl)[0][1]) ## calculating fsky, but quite expensive. and if nivp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
        

                # @log_on_start(logging.DEBUG, "_process_OBD() started")
                # @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale


                # @log_on_start(logging.DEBUG, "_process_Simulation() started")
                # @log_on_end(logging.DEBUG, "_process_Simulation() finished")       
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.obs_info['noise_info']['libdir_suffix']
                    l2base_Transformer.process_Simulation(dl, si, cf)


                # @log_on_start(logging.DEBUG, "_process_Qerec() started")
                # @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):

                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    dl.blt_pert = qe.blt_pert
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    if dl.QE_subtract_meanfield:
                        qe_tasks_sorted = ['calc_phi', 'calc_meanfield', 'calc_blt']
                    else:
                        qe_tasks_sorted = ['calc_phi', 'calc_blt']
                    qe_tasks_extracted = []
                    for taski, task in enumerate(qe_tasks_sorted):
                        if task in qe.tasks:
                            qe_tasks_extracted.append(task)
                        else:
                            break
                    dl.qe_tasks = qe_tasks_extracted
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type
                    dl.cg_tol = qe.cg_tol

                    if qe.chain == None:
                        dl.chain_descr = lambda a,b: None
                        dl.chain_model = dl.chain_descr
                    else:
                        dl.chain_model = qe.chain
                        dl.chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                        
                        if dl.chain_model.p6 == 'tr_cg':
                            _p6 = cd_solve.tr_cg
                        if dl.chain_model.p7 == 'cache_mem':
                            _p7 = cd_solve.cache_mem()
                        dl.chain_descr = lambda p2, p5 : [
                            [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

                    dl.qe_filter_directional = qe.filter_directional
                    dl.cl_analysis = qe.cl_analysis


                # @log_on_start(logging.DEBUG, "_process_Itrec() started")
                # @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    dl.it_tasks = it.tasks
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    dl.epsilon = it.epsilon
                    # chain
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.nivjob_geominfo[1]['nside']
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    
                    dl.lenjob_geominfo = it.lenjob_geominfo
                    dl.lenjob_geomlib = get_geom(it.lenjob_geominfo)
            
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol
                    dl.it_filter_directional = it.filter_directional
                    dl.itmax = it.itmax
                    dl.iterator_typ = it.iterator_typ

                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    dl.soltn_cond = it.soltn_cond
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = dl.lm_max_qlm[0]
                        dl.stepper_model.mmax_qlm = dl.lm_max_qlm[1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                        # dl.stepper = steps.nrstep(dl.lm_max_qlm[0], dl.lm_max_qlm[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=dl.verbose, epsilon=dl.epsilon)


                # @log_on_start(logging.DEBUG, "_process_Phianalysis() started")
                # @log_on_end(logging.DEBUG, "_process_Phianalysis() finished")       
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
      
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.simulationdata)

                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)
                _process_Phianalysis(dl, cf.phana)

                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                elif cf.analysis.cpp.endswith('dat'):
                    # assume its a camb-like file
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] 
                elif os.path.exists(os.path.dirname(cf.analysis.cpp)):
                    # FIXME this implicitly assumes that all cpp.npy come as convergence
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                    LL = np.arange(0,dl.lm_max_qlm[0] + 1,1)
                    k2p = lambda x: np.nan_to_num(x/(LL*(LL+1))**2/(2*np.pi))
                    dl.cpp = k2p(dl.cpp)
                dl.cpp[:dl.Lmin] *= 0.

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        ## FIXME build a correct model. For now quick and dirty, MAP_lensrec contains all information (and more)
        return Phi_analyser(extract())
    

@transform.case(DLENSALOT_Model_mm, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)