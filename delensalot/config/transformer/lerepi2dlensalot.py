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


class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """    
    def __init__(self):
        pass

    # @log_on_start(logging.DEBUG, "process_Simulation() started")
    # @log_on_end(logging.DEBUG, "process_Simulation() finished")
    def process_Simulation(dl, si, cf):
        dl.simulationdata = Simhandler(**si.__dict__)

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

        # dl.simidxs_mf = np.array(an.simidxs_mf) if dl.version != 'noMF' else np.array([])
        # # dl.simidxs_mf = dl.simidxs if dl.simidxs_mf.size == 0 else dl.simidxs_mf
        # dl.Nmf = 0 if dl.version == 'noMF' else len(dl.simidxs_mf)


        # FIXME makes this a clean implementation
        dl.simidxs_mf = np.array(an.simidxs_mf) if dl.version != 'noMF' else np.array([])
        dl.Nmf = 0 if dl.version == 'noMF' else len(dl.simidxs_mf) # FIXME dont make intermediate parameters depend on each other...
        if cf.itrec.mfvar.startswith('/'):
            dl.Nmf = 10000 # The actual number doesnt matter, as long as it is bigger than 1
        
        
        dl.TEMP_suffix = an.TEMP_suffix
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
        dl.lm_max_blt = an.lm_max_blt


        dl.transfunction_desc = an.transfunction_desc
        if dl.transfunction_desc == 'gauss_no_pixwin':
            transf_tlm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
            transf_elm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
            transf_blm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
        elif dl.transfunction_desc == 'gauss_with_pixwin':
            assert cf.noisemodel.geominfo[0] == 'healpix', 'implement non-healpix pixelwindow function'
            transf_tlm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(cf.noisemodel.geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
            transf_elm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(cf.noisemodel.geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
            transf_blm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(cf.noisemodel.geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
        dl.ttebl = {'t': transf_tlm, 'e': transf_elm, 'b':transf_blm}

        # Isotropic approximation to the filtering (used eg for response calculations)
        ftl_len = cli(dl.cls_len['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['T'])**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
        fel_len = cli(dl.cls_len['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['P'])**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
        fbl_len = cli(dl.cls_len['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['P'])**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
        dl.ftebl_len = {'t': ftl_len, 'e': fel_len, 'b':fbl_len}

        # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
        ftl_unl = cli(dl.cls_unl['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['T'])**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
        fel_unl = cli(dl.cls_unl['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['P'])**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
        fbl_unl = cli(dl.cls_unl['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev['P'])**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
        dl.ftebl_unl = {'t': ftl_unl, 'e': fel_unl, 'b':fbl_unl}

    # @log_on_start(logging.DEBUG, "_process_Meta() started")
    # @log_on_end(logging.DEBUG, "_process_Meta() finished")
    def process_Meta(dl, me, cf):
        dl.dversion = me.version


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
            _suffix += '_OBD' if cf.noisemodel.OBD == 'OBD' else '_lminB'+str(cf.analysis.lmin_teb[2])
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
                dl.simidxs_mf = np.array(an.simidxs_mf) if dl.version != 'noMF' else np.array([])
                # dl.simidxs_mf = dl.simidxs_mf if dl.simidxs_mf.size == 0 else np.array(dl.simidxs)

                dl.TEMP_suffix = an.TEMP_suffix
                dl.TEMP = transform(cf, l2T_Transformer())

            dl = DLENSALOT_Concept()
            _process_Analysis(dl, cf.analysis, cf)
            l2base_Transformer.process_Meta(dl, cf.meta, cf)
            dl.libdir_suffix = cf.simulationdata.libdir_suffix
            dl.simulationdata = Simhandler(**cf.simulationdata.__dict__)
            return dl
        return Sim_generator(extract())


    def build_QE_lensrec_new(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        def extract():
            def _process_components(dl):


                def _process_Meta(dl, me):
                    dl.dversion = me.version


                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


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


                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale

     
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
                    l2base_Transformer.process_Simulation(dl, si, cf)


                def _process_Qerec(dl, qe):
                    dl.blt_pert = qe.blt_pert
                    dl.subtract_QE_meanfield = qe.subtract_QE_meanfield
                    if dl.subtract_QE_meanfield:
                        qe_tasks_sorted = ['calc_fields', 'calc_meanfields', 'calc_templates']
                    else:
                        qe_tasks_sorted = ['calc_fields', 'calc_templates']
                    qe_tasks_extracted = []
                    for taski, task in enumerate(qe_tasks_sorted):
                        if task in qe.tasks:
                            qe_tasks_extracted.append(task)
                        else:
                            break
                    dl.qe_tasks = qe_tasks_extracted
                        
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.estimator_type = qe.estimator_type

                    ## FIXME cg chain currently only works with healpix geometry
                    dl.QE_cg_tol = qe.cg_tol
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


                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Simulation(dl, cf.simulationdata)
                _process_OBD(dl, cf.obd)
                _process_Qerec(dl, cf.qerec)

                # cf.analysis.CLfids is assumed to be a camb-like file with the field name as dictionary key and their power spectra
                
                #FIXME for now I assume this is a simple np.array with pp,ww,bb,pw,pb,wb. This needs to be generalized
                _keys = ['pp', 'ww', 'bb']
                # dl.Clfids = np.load(cf.analysis.CLfids)
                dl.CLfids = {_keys[i]: camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] for i in range(len(_keys))}
                # dl.CLfids = {key: val[:dl.lm_max_qlm[0] + 1] for key, val in zip(_keys, dl.CLfids)}
                # if cf.analysis.CLfields.endswith('dat'):
                    # dl.CLfields = (cf.analysis.CLfields, load_secondaries=True)
                #NOTE assuming these are convergence power spectra, and they come as 

            dl = DLENSALOT_Concept()
            _process_components(dl)
            dl.lenjob_geominfo = cf.itrec.lenjob_geominfo
            dl.lenjob_geomlib = get_geom(cf.itrec.lenjob_geominfo)
            thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
            dl.lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)

            dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=False, epsilon=cf.itrec.epsilon)

            QE_fields_descs = {
                "lensing":{
                    "ID": "lensing",
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'estimates/lensing'),
                    'lm_max': dl.lm_max_qlm,
                    'components': "alpha_omega",
                    'CLfids': {'alpha': dl.CLfids['pp'], 'omega': dl.CLfids['ww']},
                    'qlm_fns': {"alpha": 'qlm_alpha_simidx{idx}', "omega": 'qlm_omega_simidx{idx}'},
                    'klm_fns': {"alpha": 'klm_alpha_simidx{idx}', "omega": 'klm_omega_simidx{idx}'},
                    'qmflm_fns': {"alpha": 'qmflm_alpha_simidx{idx}', "omega": 'qmflm_omega_simidx{idx}'},
                },
                "birefringence":{
                    "ID": 'birefringence',
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'estimates/birefringence'),
                    'lm_max': dl.lm_max_qlm, #FIXME betalm?
                    'components': "beta",
                    'CLfids': {"beta": dl.CLfids['bb']},
                    "qlm_fns": {"beta": 'qlm_beta_simidx{idx}'},
                    "klm_fns": {"beta": 'klm_beta_simidx{idx}'},
                    'qmflm_fns': {"beta": 'qmflm_beta_simidx{idx}'},
                }
            }
            
            QE_fields = {name: QE_field.base(field_desc) for name, field_desc in QE_fields_descs.items()}
            
            QE_template_descs = {  # templates need a fn, that's all
                "lensing:": {
                    "ID": "lensing",
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'templates'),
                    'components': "alpha_omega",
                    'CLfids': None,
                    "klm_fns": {"alpha": 'klm_alpha_template_simidx{idx}', "omega": 'klm_omega_template_simidx{idx}'},
                    },
                "birefringence": {
                    "ID": "birefringence",
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'templates'),
                    'components': "beta",
                    'CLfids': None,
                    "klm_fns": {"beta": 'klm_beta_template_simidx{idx}'},
                    },
            }
                # {
                # "ID": "joint",
                # "klm_fns": {"joint": 'klm_joint_template_simidx{idx}'},
                # }
            templates = [QE_field.base(field_desc) for name, field_desc in QE_template_descs.items()]
            template_operator_descs = {
                "lensing": {
                    "Lmin": dl.Lmin,
                    "perturbative": dl.blt_pert,
                    "lm_max": dl.lm_max_blt,
                    "lm_max_qlm": dl.lm_max_qlm,
                    "components": 'alpha_omega',
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'estimates/lensing'),
                    "field_fns": QE_fields_descs["lensing"]['klm_fns'],
                    "ffi": dl.ffi,
                },
                "birefringence": {
                    "Lmin": dl.Lmin,
                    "lm_max": dl.lm_max_blt, # FIXME betatemp_lm?
                    "components": 'beta',
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'estimates/birefringence'),
                    "field_fns": QE_fields_descs['birefringence']['klm_fns'],
                },
            }
            template_operators = { # NOTE joint can be build by combinatorics
                "lensing": operator.lensing(template_operator_descs["lensing"]),
                "birefringence": operator.birefringence(template_operator_descs["birefringence"]),
            }

            QE_filterqest_desc = {
                "estimator_key": cf.analysis.key,
                "estimator_type": dl.estimator_type,
                "libdir": opj(transform(cf, l2T_Transformer()), 'QE'),
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
                "lm_max_qlm": dl.lm_max_qlm,
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

            QE_searchs_desc = {
                "lensing": {
                    "ID": "lensing",
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'lensing'),
                    "QE_filterqest_desc": QE_filterqest_desc,
                    "field": QE_fields["lensing"],
                    "estimator_key": cf.analysis.key,
                    "cls_len": dl.cls_len,
                    "cls_unl": dl.cls_unl,
                    "simidxs": cf.analysis.simidxs,
                    "simidxs_mf": dl.simidxs_mf,
                    "subtract_meanfield": dl.subtract_QE_meanfield,

                },
                "birefringence": {
                    "ID": "birefringence",
                    "libdir": opj(transform(cf, l2T_Transformer()), 'QE', 'birefringence'),
                    "QE_filterqest_desc": QE_filterqest_desc,
                    "field": QE_fields["birefringence"],
                    "estimator_key": cf.analysis.key,
                    "cls_len": dl.cls_len,
                    "cls_unl": dl.cls_unl,
                    "simidxs": cf.analysis.simidxs,
                    "simidxs_mf": dl.simidxs_mf,
                    "subtract_meanfield": dl.subtract_QE_meanfield,
                },
            }

            QE_handler_desc = {
                "template_operators": template_operators,
                "templates": templates,
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
                    dl.ffi = deflection(dl.lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=False, epsilon=dl.epsilon)
                
                _process_Itrec(dl, cf.itrec)

            dl = DLENSALOT_Concept()
            dl = self.build_QE_lensrec_new(cf)
            QE_searchs = dl.QE_searchs
            _process_components(dl)

            MAP_libdir_prefix = opj(transform(cf, l2T_Transformer()), 'MAP', '{estimator_key}'.format(estimator_key=cf.analysis.key.split('_')[-1]))

            # input: all kwargs needed to build the MAP fields
            MAP_fields_descs = [{
                    "ID": "lensing",
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    'lm_max': dl.lm_max_qlm,
                    "components": 'alpha_omega',
                    'CLfids': {'alpha': dl.CLfids['pp'], 'omega': dl.CLfids['ww']},
                    'fns': {"alpha": 'klm_alpha_simidx{idx}_it{it}', "omega": 'klm_omega_simidx{idx}_it{it}'}, # This could be hardcoded, but I want to keep it flexible
                    'increment_fns': {"alpha": 'kinclm_alpha_simidx{idx}_it{it}', "omega": 'kinclm_omega_simidx{idx}_it{it}'},
                    'meanfield_fns': {"alpha": 'kmflm_alpha_simidx{idx}_it{it}', "omega": 'kmflm_omega_simidx{idx}_it{it}'},
                },{
                    "ID": 'birefringence',
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    'lm_max': dl.lm_max_qlm, # FIXME betalm?
                    "components": 'beta',
                    'CLfids': {'beta': dl.CLfids['bb']},
                    "fns": {"beta": 'klm_beta_simidx{idx}_it{it}'}, # This could be hardcoded, but I want to keep it flexible
                    'increment_fns': {"beta": 'kinclm_beta_simidx{idx}_it{it}'},
                    'meanfield_fns': {"beta": 'kmflm_beta_simidx{idx}_it{it}'},
                },
            ]
            MAP_fields = {field_desc["ID"]: MAP_field.base(field_desc) for field_desc in MAP_fields_descs}

            # input: all kwargs needed to build the MAP search
            _MAP_operators_desc = {}
            _MAP_operators_desc['lensing_operator'] = {
                'lm_max': dl.lm_max_ivf,
                "lm_max_qlm": dl.lm_max_qlm,
                "Lmin": dl.Lmin,
                "perturbative": False,
                "components": 'alpha_omega',
                "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                'field_fns': MAP_fields["lensing"].fns,
                "ffi": dl.ffi,
            }
            _MAP_operators_desc['birefringence_operator'] = {
                'lm_max': dl.lm_max_ivf,
                "Lmin": dl.Lmin,
                "components": 'beta',
                "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                'field_fns': MAP_fields['birefringence'].fns,
            }
            _MAP_operators_desc['spin_raise'] = {
                'lm_max': dl.lm_max_ivf,
            }
            _MAP_operators_desc['multiply'] = {
                'factor': -1j,
            }
            filter_operators = []
            gradients_operators = {}
            # This depends on what is in the data
            cf.build = 'lensingplusbirefringence'
            if cf.build == 'lensingplusbirefringence':
                filter_operators.append(operator.lensing(_MAP_operators_desc['lensing_operator']))
                filter_operators.append(operator.birefringence(_MAP_operators_desc['birefringence_operator']))
                gradients_operators['lensing'] = operator.joint([*filter_operators, operator.spin_raise(_MAP_operators_desc['spin_raise'])])
                gradients_operators['birefringence'] = operator.joint([*filter_operators, operator.multiply(_MAP_operators_desc['multiply'])])
            if cf.build == 'lensing':
                filter_operators.append(operator.lensing(_MAP_operators_desc['lensing_operator']))
                gradients_operators['lensing'] = operator.joint([*filter_operators, operator.spin_raise(_MAP_operators_desc['spin_raise'])])    
            if cf.build == 'birefringence':
                filter_operators.append(operator.birefringence(_MAP_operators_desc['birefringence_operator']))
                gradients_operators['birefringence'] = operator.joint([*filter_operators, operator.multiply(_MAP_operators_desc['multiply'])])
            ivf_operator = operator.ivf_operator(filter_operators)
            WF_operator = operator.WF_operator(filter_operators) #TODO this is ivf_operator*ivf_operator^dagger, could be implemented via ivf.

            gfield_descs = [{
                "ID": gradient_name,
                "libdir": opj(MAP_libdir_prefix, 'gradients'),
                "libdir_prior": opj(MAP_libdir_prefix, 'estimates'),
                "lm_max": dl.lm_max_qlm,
                "meanfield_fns": 'mf_glm_{gradient_name}_simidx{idx}_it{it}'.format(gradient_name=gradient_name, it="{it}", idx="{idx}"),
                "quad_fns": 'quad_glm_{gradient_name}_simidx{idx}_it{it}'.format(gradient_name=gradient_name, it="{it}", idx="{idx}"),
                "prior_fns": 'klm_{component}_simidx{idx}_it{it}'.format(component="{component}", it="{it}", idx="{idx}"), # prior is just field, and then we do a simple divide by spectrum (almxfl)
                "total_increment_fns": 'ginclm_{gradient_name}_simidx{idx}_it{it}'.format(gradient_name=gradient_name, it="{it}", idx="{idx}"),    
                "total_fns": 'gtotlm_{gradient_name}_simidx{idx}_it{it}'.format(gradient_name=gradient_name, it="{it}", idx="{idx}"),    
                "chh": {"alpha": np.ones(shape=dl.lm_max_qlm[0]+1), "omega": np.ones(shape=dl.lm_max_qlm[0]+1)} if gradient_name == "lensing" else {"beta": np.ones(shape=dl.lm_max_qlm[0]+1)}, #FIXME this is prior times scaling factor
                "components": 'alpha_omega' if gradient_name == 'lensing' else 'beta',
            } for gradient_name, gradient_operator in gradients_operators.items()]
            MAP_gfields = {gfield_desc["ID"]: MAP_field.gradient(gfield_desc) for gfield_desc in gfield_descs}
            gradient_descs = {}
            for gradient_name, gradient_operator in gradients_operators.items():
                gradient_descs.update({ gradient_name: {
                    "ID": gradient_name,
                    "field": MAP_fields[gradient_name],
                    "gfield": MAP_gfields[gradient_name],
                    "noisemodel_coverage": dl.it_filter_directional,
                    "estimator_key":  cf.analysis.key,
                    "simulationdata": dl.simulationdata,
                    "lm_max_ivf": dl.lm_max_ivf,
                    "lm_max_qlm": dl.lm_max_qlm,
                    'itmax': dl.itmax,
                    "gradient_operator": gradient_operator,
                    "ffi": dl.ffi,
                }})

            MAP_ivffilter_field_desc = {
                "ID": "ivf",
                "libdir": opj(MAP_libdir_prefix, 'filter'),
                "lm_max": dl.lm_max_ivf,
                "components": 1,
                "fns": "ivf_simidx{idx}_it{it}",
            }
            MAP_WFfilter_field_desc = {
                "ID": "WF",
                "libdir": opj(MAP_libdir_prefix, 'filter'),
                "lm_max": dl.lm_max_ivf,
                "components": 1,
                "fns": "WF_simidx{idx}_it{it}",
            }
            MAP_filter_desc = {
                "ID": "polarization",
                'ivf_operator': ivf_operator,
                'WF_operator': WF_operator,
                "ivf_field": MAP_field.filter(MAP_ivffilter_field_desc),
                "WF_field": MAP_field.filter(MAP_WFfilter_field_desc),
                'beam': operator.beam({"beamwidth": cf.analysis.beam, "lm_max":dl.lm_max_ivf}),
                'Ninv_desc': [dl.nivt_desc, dl.nivp_desc],
                "simulationdata": dl.simulationdata,
                "chain_descr": dl.chain_descr(dl.lm_max_unl[0], dl.it_cg_tol(0)),
                "ttebl": dl.ttebl,
                "cls_filt": dl.cls_unl,
                "lm_max_ivf": dl.lm_max_ivf,

            }
            
            template_operators = {
                "lensing": operator.lensing({
                    "Lmin": dl.Lmin,
                    "perturbative": False,
                    "lm_max": dl.lm_max_blt,
                    "lm_max_qlm": dl.lm_max_qlm,
                    "components": 'alpha_omega',
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    "field_fns": MAP_fields["lensing"].fns,
                    "ffi": dl.ffi,
                }),
                "birefringence": operator.birefringence({
                    "Lmin": dl.Lmin,
                    "lm_max": dl.lm_max_blt,
                    "components": 'beta',
                    "libdir": opj(MAP_libdir_prefix, 'estimates/'),
                    "field_fns": MAP_fields['birefringence'].fns,
                })
            }

            curvature_desc = {
                "ID": "curvature",
                "field": MAP_field.curvature(
                    {"ID": "curvature",
                    "libdir": opj(MAP_libdir_prefix, 'curvature'),
                    "fns": 'diff_klm_{gradient_name}_simidx{idx}_it{it}m{itm1}'.format(gradient_name=gradient_name, it="{it}", itm1="{itm1}", idx="{idx}"),
                }),
                "bfgs_desc": {},
            }

            desc = {
                "itmax": dl.itmax,
            }
            template_descs = {
                "libdir": opj(MAP_libdir_prefix, 'templates'),
                "template_operators": template_operators,
            }
            MAP_searchs_desc = {
                'gradient_descs': gradient_descs,
                'MAP_fields': MAP_fields,
                'filter_desc': MAP_filter_desc,
                'curvature_desc': curvature_desc,
                "desc" : desc,
                "template_descs": template_descs,
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
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
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
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
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
                    dl.TEMP_suffix = an.TEMP_suffix,
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
                dl.libdir_suffix = cf.simulationdata.libdir_suffix
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
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
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