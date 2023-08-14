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
from plancklens.qcinv import cd_solve

from lenspyx.remapping import utils_geom as lug
from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.sims.sims_lib import Simhandler

from delensalot.utils import cli, camb_clfile, load_file
from delensalot.utility.utils_hp import gauss_beam

from delensalot.core.iterator import steps
from delensalot.core.handler import OBD_builder, Sim_generator, QE_lr, MAP_lr, Map_delenser
from delensalot.core.opfilt.bmodes_ninv import template_dense

from delensalot.config.visitor import transform, transform3d
from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm, DLENSALOT_Concept


class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """    
    def __init__(self):
        pass


    @log_on_start(logging.DEBUG, "process_Simulation() started")
    @log_on_end(logging.DEBUG, "process_Simulation() finished")
    def process_Simulation(dl, si, cf):
        dl.simulationdata = Simhandler(**si.__dict__)


    @log_on_start(logging.DEBUG, "_process_Analysis() started")
    @log_on_end(logging.DEBUG, "_process_Analysis() finished")
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
        dl.simidxs_mf = np.array(an.simidxs_mf) if dl.version != 'noMF' else np.array([])
        # dl.simidxs_mf = dl.simidxs if dl.simidxs_mf.size == 0 else dl.simidxs_mf
        dl.Nmf = 0 if dl.version == 'noMF' else len(dl.simidxs_mf)
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
            assert dl.nivjob_geominfo[0] == 'healpix', 'implement non-healpix pixelwindow function'
            transf_tlm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl.nivjob_geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
            transf_elm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl.nivjob_geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
            transf_blm = gauss_beam(df.a2r(an.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl.nivjob_geominfo[1]['nside'], lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
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


    @log_on_start(logging.DEBUG, "_process_Meta() started")
    @log_on_end(logging.DEBUG, "_process_Meta() finished")
    def process_Meta(dl, me, cf):
        dl.dversion = me.version


class l2T_Transformer:
    # TODO this could use refactoring. Better name generation
    """global access for custom TEMP directory name, so that any job stores the data at the same place.
    """

    # @log_on_start(logging.DEBUG, "build() started")
    # @log_on_end(logging.DEBUG, "build() finished")
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

    @log_on_start(logging.DEBUG, "get_nlev() started")
    @log_on_end(logging.DEBUG, "get_nlev() finished")
    def get_nlev(cf):
        return cf.noisemodel.nlev


    @log_on_start(logging.DEBUG, "get_nivt_desc() started")
    @log_on_end(logging.DEBUG, "get_nivt_desc() finished")
    def get_nivt_desc(cf, dl):
        nlev = l2OBD_Transformer.get_nlev(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf, dl)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        if cf.noisemodel.nivt_map is None:
            if dl.nivjob_geominfo[0] == 'healpix':
                ninv_desc = [np.array([hp.nside2pixarea(dl.nivjob_geominfo[1]['nside'], degrees=True) * 60 ** 2 / nlev['T'] ** 2])/noisemodel_norm] + masks
            else:
                assert 0, 'needs testing, please choose Healpix geom for nivjob for now'
                vamin =  4*np.pi * (180/np.pi)**2 / get_geom(cf.itrec.lenjob_geominfo).npix()
                ninv_desc = [np.array([vamin * 60 ** 2 / nlev['T'] ** 2])/noisemodel_norm] + masks
        else:
            niv = np.load(cf.noisemodel.nivt_map)
            ninv_desc = [niv] + masks
        return ninv_desc


    @log_on_start(logging.DEBUG, "get_nivp_desc() started")
    @log_on_end(logging.DEBUG, "get_nivp_desc() finished")
    def get_nivp_desc(cf, dl):
        nlev = l2OBD_Transformer.get_nlev(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf, dl)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        if cf.noisemodel.nivp_map is None:
            if dl.nivjob_geominfo[0] == 'healpix':
                ninv_desc = [[np.array([hp.nside2pixarea(dl.nivjob_geominfo[1]['nside'], degrees=True) * 60 ** 2 / nlev['P'] ** 2])/noisemodel_norm] + masks]
            else:
                assert 0, 'needs testing, pleasechoose Healpix geom for nivjob for now'
                vamin =  4*np.pi * (180/np.pi)**2 / get_geom(cf.itrec.lenjob_geominfo).npix()
                ninv_desc = [[np.array([vamin * 60 ** 2 / nlev['P'] ** 2])/noisemodel_norm] + masks]
        else:
            niv = np.load(cf.noisemodel.nivp_map)
            ninv_desc = [[niv] + masks]
        return ninv_desc


    @log_on_start(logging.DEBUG, "get_masks() started")
    @log_on_end(logging.DEBUG, "get_masks() finished")
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


    def build_QE_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        @log_on_start(logging.DEBUG, "extract() started")
        @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                @log_on_start(logging.DEBUG, "_process_Meta() started")
                @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                @log_on_start(logging.DEBUG, "_process_Computing() started")
                @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
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


                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                    nivjob_geomlib_ = get_geom(cf.noisemodel.geominfo)
                    dl.tpl = template_dense(dl.lmin_teb[2], nivjob_geomlib_, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)


                @log_on_start(logging.DEBUG, "_process_Simulation() started")
                @log_on_end(logging.DEBUG, "_process_Simulation() finished")       
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
                    l2base_Transformer.process_Simulation(dl, si, cf)


                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
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


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
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
                if dl.OBD == 'OBD':
                    _process_OBD(dl, cf.obd)
                else:
                    dl.tpl = None
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
        @log_on_start(logging.DEBUG, "extract() started")
        @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                @log_on_start(logging.DEBUG, "_process_Meta() started")
                @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                @log_on_start(logging.DEBUG, "_process_Computing() started")
                @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.sky_coverage = nm.sky_coverage
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    if dl.sky_coverage == 'masked':
                        dl.rhits_normalised = nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_nivp_desc(cf, dl)[0][1]) ## calculating fsky, but quite expensive. and if ninvp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    

                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                    nivjob_geomlib_ = get_geom(cf.noisemodel.geominfo)
                    dl.tpl = template_dense(dl.lmin_teb[2], nivjob_geomlib_, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)


                @log_on_start(logging.DEBUG, "_process_Simulation() started")
                @log_on_end(logging.DEBUG, "_process_Simulation() finished")       
                def _process_Simulation(dl, si):
                    dl.libdir_suffix = cf.simulationdata.libdir_suffix
                    l2base_Transformer.process_Simulation(dl, si, cf)


                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
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


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
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
                if dl.OBD  == 'OBD':
                    _process_OBD(dl, cf.obd)
                else:
                    dl.tpl = None
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
        @log_on_start(logging.DEBUG, "extract() started")
        @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                @log_on_start(logging.DEBUG, "_process_Computing() started")
                @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = int(os.environ.get('OMP_NUM_THREADS', co.OMP_NUM_THREADS))


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
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


                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.libdir = od.libdir if type(od.libdir) == str else 'nopath'
                    dl.nlev_dep = od.nlev_dep
                    dl.rescale = od.rescale

                    if os.path.isfile(opj(dl.libdir,'tniti.npy')):
                        # TODO need to test if it is the right tniti.npy
                        # TODO dont exit, rather skip job
                        log.warning("tniti.npy in destination dir {} already exists.".format(dl.libdir))
                        log.warning("Please check your settings.")


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
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
        @log_on_start(logging.DEBUG, "extract() started")
        @log_on_end(logging.DEBUG, "extract() finished")
        def extract():
            def _process_components(dl):
                @log_on_start(logging.DEBUG, "_process_Meta() started")
                @log_on_end(logging.DEBUG, "_process_Meta() finished")
                def _process_Meta(dl, me):
                    dl.dversion = me.version


                @log_on_start(logging.DEBUG, "_process_Computing() started")
                @log_on_end(logging.DEBUG, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = co.OMP_NUM_THREADS
                    os.environ["OMP_NUM_THREADS"] = str(dl.tr)
                    log.debug("OMP_NUM_THREADS: {} and {}".format(dl.tr, os.environ.get('OMP_NUM_THREADS')))


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    # super(l2base_Transformer, self)
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.nivjob_geomlib = get_geom(nm.geominfo)
                    dl.nivjob_geominfo = nm.geominfo
                    # thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                    ## this is for delensing, and pospace doesn't support truncated maps, therefore no restrict here
                    # dl.nivjob_geomlib = dl.nivjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
                    dl.nlev = l2OBD_Transformer.get_nlev(cf)


                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):
                    dl.nivt_desc = l2OBD_Transformer.get_nivt_desc(cf, dl)
                    dl.nivp_desc = l2OBD_Transformer.get_nivp_desc(cf, dl)
                    dl.blt_pert = qe.blt_pert
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type
                    dl.qe_filter_directional = qe.filter_directional


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
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
    

@transform.case(DLENSALOT_Model_mm, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)