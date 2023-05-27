#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build delensalot model from configuation file
The transform.case functions at the very bottom of this module choose, depending on the delensalot job, the transformer class. The transformer class build a suitable delensalot model from the configuration file, which can be understood from the core functions and dependencies (such as plancklens).
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

from delensalot.utils import cli, camb_clfile
from delensalot.config.visitor import transform, transform3d

from delensalot.core.iterator import steps
from delensalot.utility.utils_hp import gauss_beam
from delensalot.core.handler import OBD_builder, Sim_generator, QE_lr, MAP_lr, Map_delenser
from delensalot.core.opfilt.bmodes_ninv import template_dense


from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm, DLENSALOT_Concept


# TODO swap rhits with ninv
class l2base_Transformer:
    """Initializes attributes needed across all Jobs, or which are at least handy to have
    """    
    def __init__(self):
        pass


    @log_on_start(logging.DEBUG, "_process_Data() started")
    @log_on_end(logging.DEBUG, "_process_Data() finished")
    def process_Data(dl, da, cf):
        if loglevel <= 20:
            dl.verbose = True
        elif loglevel >= 30:
            dl.verbose = False
        # package_
        _package = da.package_
        # module_
        _module = da.module_
        # class_
        dl._class = da.class_
        # class_parameters -> sims
        dl.sims_class_parameters = da.class_parameters
        if 'fg' in dl.sims_class_parameters:
            dl.fg = dl.sims_class_parameters['fg']
        dl._sims_full_name = '{}.{}'.format(_package, _module)
        dl.sims_beam = da.beam
        dl.sims_lmax_transf = da.lmax_transf
        dl.sims_nlev_t = da.nlev_t
        dl.sims_nlev_p = da.nlev_p
        dl.sims_nside = da.nside
        dl.epsilon = da.epsilon
        dl.parameter_maps = da.maps
        dl.parameter_phi = da.phi


    @log_on_start(logging.DEBUG, "_process_Analysis() started")
    @log_on_end(logging.DEBUG, "_process_Analysis() finished")
    def process_Analysis(dl, an, cf):
        # dlm_mod
        dl.dlm_mod_bool = cf.madel.dlm_mod
        # beam
        dl.beam = an.beam
        # mask
        dl.mask_fn = an.mask
        # key -> k
        dl.k = an.key
        # reconstruction_method
        dl.reconstruction_method = an.reconstruction_method
        # lmin_teb
        dl.lmin_teb = an.lmin_teb
        # version -> version
        dl.version = an.version
        # simidxs
        dl.simidxs = an.simidxs
        # simidxs_mf
        dl.simidxs_mf = an.simidxs_mf if dl.version != 'noMF' else []
        # print(dl.simidxs_mf)
        dl.simidxs_mf = dl.simidxs_mf if dl.simidxs_mf != [] else dl.simidxs
        # print(dl.simidxs_mf)
        dl.Nmf = 0 if dl.version == 'noMF' else len(dl.simidxs_mf)
        # TEMP_suffix -> TEMP_suffix
        dl.TEMP_suffix = an.TEMP_suffix
        dl.TEMP = transform(cf, l2T_Transformer())
        # Lmin
        #TODO give user freedom about fiducial model. But needed here for dl.cpp
        dl.cls_unl = camb_clfile(an.cls_unl)
        # if 
        dl.cls_len = camb_clfile(an.cls_len)
        dl.Lmin = an.Lmin
        # zbounds -> zbounds
        if an.zbounds[0] == 'nmr_relative':
            dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), an.zbounds[1])
        elif an.zbounds[0] == 'mr_relative':
            _zbounds = df.get_zbounds(hp.read_map(an.mask), np.inf)
            dl.zbounds = df.extend_zbounds(_zbounds, degrees=an.zbounds[1])
        elif type(an.zbounds[0]) in [float, int, np.float64]:
            dl.zbounds = an.zbounds
        # zbounds_len
        if an.zbounds_len[0] == 'extend':
            dl.zbounds_len = df.extend_zbounds(dl.zbounds, degrees=an.zbounds_len[1])
        elif an.zbounds_len[0] == 'max':
            dl.zbounds_len = [-1, 1]
        elif type(an.zbounds_len[0]) in [float, int, np.float64]:
            dl.zbounds_len = an.zbounds_len
        # pbounds -> pb_ctr, pb_extent
        dl.pb_ctr, dl.pb_extent = an.pbounds
        # lm_max_ivf -> lm_ivf
        dl.lm_max_ivf = an.lm_max_ivf

        dl.lm_max_blt = an.lm_max_blt


    @log_on_start(logging.DEBUG, "_process_Meta() started")
    @log_on_end(logging.DEBUG, "_process_Meta() finished")
    def process_Meta(dl, me, cf):
        dl.dversion = me.version


class l2T_Transformer:
    # TODO this needs a big refactoring. Suggest working via cachers
    """global access for custom TEMP directory name, so that any job stores the data at the same place.
    """

    # @log_on_start(logging.INFO, "build() started")
    # @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        if cf.job.jobs == ['build_OBD']:
            return cf.obd.libdir
        else:
            _suffix = cf.data.class_
            if 'fg' in cf.data.class_parameters:
                _suffix +='_%s'%(cf.data.class_parameters['fg'])
            if cf.noisemodel.OBD:
                _suffix += '_OBD'
            else:
                _suffix += '_lminB'+str(cf.analysis.lmin_teb[2])
               

            if cf.analysis.TEMP_suffix != '':
                _suffix += '_'+cf.analysis.TEMP_suffix
            TEMP =  opj(os.environ['SCRATCH'], cf.data.package_, cf.data.module_.split('.')[-1], _suffix)

            return TEMP


    # @log_on_start(logging.INFO, "build_delsuffix() started")
    # @log_on_end(logging.INFO, "build_delsuffix() finished")
    def build_delsuffix(self, dl):
        if dl.version == '':
            return os.path.join(dl.TEMP, 'plotdata', 'base')
        else:
            return os.path.join(dl.TEMP, 'plotdata', dl.version)


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

    @log_on_start(logging.DEBUG, "get_nlevt() started")
    @log_on_end(logging.DEBUG, "get_nlevt() finished")
    def get_nlevt(cf):
        if type(cf.noisemodel.nlev_t) in [float, np.float64, int]:
            _nlev_t = cf.noisemodel.nlev_t
        elif type(cf.noisemodel.nlev_t) == tuple:
            _nlev_t = np.load(cf.noisemodel.nlev_t[1])
            _nlev_t[:3] = 0
            if cf.noisemodel.nlev_t[0] == 'cl':
                # assume that nlev comes as cl. Scale to arcmin
                _nlev_t = df.c2a(_nlev_t)
                
        return _nlev_t


    @log_on_start(logging.DEBUG, "get_nlevp() started")
    @log_on_end(logging.DEBUG, "get_nlevp() finished")
    def get_nlevp(cf):
        _nlev_p = 0
        if type(cf.noisemodel.nlev_p) in [float, np.float64, int]:
                _nlev_p = cf.noisemodel.nlev_p
        elif type(cf.noisemodel.nlev_p) == tuple:
            _nlev_p = np.load(cf.noisemodel.nlev_p[1])
            _nlev_p[:3] = 0
            if cf.noisemodel.nlev_p[0] == 'cl':
                # assume that nlev comes as cl. Scale to arcmin
                _nlev_p = df.c2a(_nlev_p)
        
        return _nlev_p


    @log_on_start(logging.DEBUG, "get_ninvt() started")
    @log_on_end(logging.DEBUG, "get_ninvt() finished")
    def get_ninvt(cf, nside=np.nan):
        if np.isnan(nside):
            nside = cf.data.nside
        nlev_t = l2OBD_Transformer.get_nlevt(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        ninv_desc = [np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_t ** 2])/noisemodel_norm] + masks

        return ninv_desc


    @log_on_start(logging.DEBUG, "get_ninvp() started")
    @log_on_end(logging.DEBUG, "get_ninvp() finished")
    def get_ninvp(cf, nside=np.nan):
        if np.isnan(nside):
            nside = cf.data.nside
        nlev_p = l2OBD_Transformer.get_nlevp(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        ninv_desc = [[np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_p ** 2])/noisemodel_norm] + masks]

        return ninv_desc


    @log_on_start(logging.DEBUG, "get_masks() started")
    @log_on_end(logging.DEBUG, "get_masks() finished")
    def get_masks(cf):
        # TODO refactor. This here generates a mask from the rhits map..
        # but this should really be detached from one another
        masks = []
        if cf.noisemodel.rhits_normalised is not None:
            msk = df.get_nlev_mask(cf.noisemodel.rhits_normalised[1], hp.read_map(cf.noisemodel.rhits_normalised[0]))
        else:
            msk = np.ones(shape=hp.nside2npix(cf.data.nside))
        masks.append(msk)
        if cf.analysis.mask is not None:
            if type(cf.analysis.mask) == str:
                _mask = cf.analysis.mask
            elif cf.noisemodel.mask[0] == 'nlev':
                noisemodel_rhits_map = msk.copy()
                _mask = df.get_nlev_mask(cf.analysis.mask[1], noisemodel_rhits_map)
                _mask = np.where(_mask>0., 1., 0.)
        else:
            _mask = np.ones(shape=hp.nside2npix(cf.data.nside))
        masks.append(_mask)

        return masks, msk


class l2delensalotjob_Transformer(l2base_Transformer):
    """builds delensalot job from configuration file
    """
    def build_generate_sim(self, cf):
        def extract():
            dl = DLENSALOT_Concept()    
            l2base_Transformer.process_Data(dl, cf.data, cf)
            l2base_Transformer.process_Analysis(dl, cf.analysis, cf)
            l2base_Transformer.process_Meta(dl, cf.meta, cf)
            return dl
        return Sim_generator(extract())


    def build_QE_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        @log_on_start(logging.INFO, "extract() started")
        @log_on_end(logging.INFO, "extract() finished")
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
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    # sky_coverage
                    dl.sky_coverage = nm.sky_coverage
                    # TODO assuming that masked sky comes with a hits-count map. If not, take mask
                    if dl.sky_coverage == 'masked':
                        # rhits_normalised
                        dl.rhits_normalised = dl.masks if nm.rhits_normalised is None else nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_ninvp(cf)[0][1]) ## calculating fsky, but quite expensive. and if ninvp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    # spectrum_type
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    # nlev_t
                    dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
                    # nlev_p
                    dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                    

                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                    if cf.noisemodel.ninvjob_geometry == 'healpix_geometry':
                        dl.ninvjob_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_geometry = dl.ninvjob_geometry.restrict(*thtbounds, northsouth_sym=False)
                    dl.tpl = template_dense(dl.lmin_teb[2], dl.ninvjob_geometry, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)

        
                @log_on_start(logging.DEBUG, "_process_Data() started")
                @log_on_end(logging.DEBUG, "_process_Data() finished")       
                def _process_Data(dl, da):
                    l2base_Transformer.process_Data(dl, da, cf)
                    # transferfunction
                    dl.transferfunction = da.transferfunction
                    if dl.transferfunction == 'gauss_no_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    elif dl.transferfunction == 'gauss_with_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    dl.ttebl = {'t': transf_tlm, 'e': transf_elm, 'b':transf_blm}

                    # Isotropic approximation to the filtering (used eg for response calculations)
                    ftl_len = cli(dl.cls_len['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_len = cli(dl.cls_len['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_len = cli(dl.cls_len['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_len = {'t': ftl_len, 'e': fel_len, 'b':fbl_len}

                    # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                    ftl_unl = cli(dl.cls_unl['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_unl = cli(dl.cls_unl['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_unl = cli(dl.cls_unl['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_unl = {'t': ftl_unl, 'e': fel_unl, 'b':fbl_unl}
        

                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):

                    dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                    dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                    # blt_pert
                    dl.blt_pert = qe.blt_pert
                    # qe_tasks
                    dl.qe_tasks = qe.tasks
                    # QE_subtract_meanfield
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    ## if QE_subtract_meanfield is True, mean-field needs to be calculated either way.
                    ## also move calc_meanfield to the front, so it is calculated first. The following lines assume that all other tasks are in the right order...
                    ## TODO allow user to provide task-list unsorted
                    if 'calc_phi' in dl.qe_tasks:
                        if dl.QE_subtract_meanfield:
                            if 'calc_meanfield' not in dl.qe_tasks:
                                dl.qe_tasks = ['calc_meanfield'].append(dl.qe_tasks)
                            elif dl.qe_tasks[0] != 'calc_meanfield':
                                if dl.qe_tasks[1] == 'calc_meanfield':
                                    buffer = copy.deepcopy(dl.qe_tasks[0])
                                    dl.qe_tasks[0] = 'calc_meanfield'
                                    dl.qe_tasks[1] = buffer
                                else:
                                    buffer = copy.deepcopy(dl.qe_tasks[0])
                                    dl.qe_tasks[0] = 'calc_meanfield'
                                    dl.qe_tasks[2] = buffer
                    # lmax_qlm
                    dl.lm_max_qlm = qe.lm_max_qlm

                    dl.qlm_type = qe.qlm_type

                    # ninvjob_qe_geometry
                    if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                        # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                        # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                        dl.ninvjob_qe_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(1), np.arccos(-1))
                        dl.ninvjob_qe_geometry = dl.ninvjob_qe_geometry.restrict(*thtbounds, northsouth_sym=False)
                    elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                        dl.ninvjob_qe_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_qe_geometry = dl.ninvjob_qe_geometry.restrict(*thtbounds, northsouth_sym=False)

                    # cg_tol
                    dl.cg_tol = qe.cg_tol

                    # chain
                    if qe.chain == None:
                        dl.chain_descr = lambda a,b: None
                        dl.chain_model = dl.chain_descr
                    else:
                        dl.chain_model = qe.chain
                        dl.chain_model.p3 = dl.sims_nside
                        
                        if dl.chain_model.p6 == 'tr_cg':
                            _p6 = cd_solve.tr_cg
                        if dl.chain_model.p7 == 'cache_mem':
                            _p7 = cd_solve.cache_mem()
                        dl.chain_descr = lambda p2, p5 : [
                            [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

                    # filter
                    dl.qe_filter_directional = qe.filter_directional

                    # qe_cl_analysis
                    dl.cl_analysis = qe.cl_analysis


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    # tasks
                    dl.it_tasks = it.tasks
                    # lmaxunl
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    # chain
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.sims_nside
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    
                    # lenjob_geometry
                    # TODO lm_max_unl should be a bit larger here for geometry, perhaps add + X (~500)
                    # dl.lenjob_geometry = lug.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2, zbounds=dl.zbounds_len) if it.lenjob_geometry == 'thin_gauss' else None
                    dl.lenjob_geometry = lug.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2)
                    # lenjob_pbgeometry
                    dl.lenjob_pbgeometry = lug.pbdGeometry(dl.lenjob_geometry, lug.pbounds(dl.pb_ctr, dl.pb_extent)) if it.lenjob_pbgeometry == 'pbdGeometry' else None
                    
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    # cg_tol
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol*0.1
                    # filter
                    dl.it_filter_directional = it.filter_directional
                    # itmax
                    dl.itmax = it.itmax
                    # iterator_typ
                    dl.iterator_typ = it.iterator_typ

                    # mfvar
                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    # soltn_cond
                    dl.soltn_cond = it.soltn_cond
                    # stepper
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = dl.lm_max_qlm[0]
                        dl.stepper_model.mmax_qlm = dl.lm_max_qlm[1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                        # dl.stepper = steps.nrstep(dl.lm_max_qlm[0], dl.lm_max_qlm[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
                    dl.ffi = deflection(dl.lenjob_geometry, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=dl.verbose, epsilon=dl.epsilon)
                
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Data(dl, cf.data)
                if dl.OBD:
                    _process_OBD(dl, cf.obd)
                else:
                    dl.tpl = None
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                else:
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] ## TODO could be added via 'fiducial' parameter in dlensalot config for user
                dl.cpp[:dl.Lmin] *= 0.

                if dl.it_filter_directional == 'anisotropic':
                    # ninvjob_geometry
                    if cf.noisemodel.ninvjob_geometry == 'healpix_geometry':
                        dl.ninvjob_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_geometry = dl.ninvjob_geometry.restrict(*thtbounds, northsouth_sym=False)

            dl = DLENSALOT_Concept()
            _process_components(dl)
            ## TODO. Current solution to fake an iteration handler for QE to calc blt is to initialize one here.
            ## In the future, I want to remove get_template_blm from the iteration_handler, at least for QE.
            if 'calc_blt' in dl.qe_tasks or 'calc_blt' in dl.it_tasks:
                dl.MAP_job = transform3d(dl, 'MAP_lensrec', l2delensalotjob_Transformer())
            return dl

        return QE_lr(extract())


    def build_MAP_lensrec(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        @log_on_start(logging.INFO, "extract() started")
        @log_on_end(logging.INFO, "extract() finished")
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
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    # sky_coverage
                    dl.sky_coverage = nm.sky_coverage
                    # TODO assuming that masked sky comes with a hits-count map. If not, take mask
                    if dl.sky_coverage == 'masked':
                        # rhits_normalised
                        dl.rhits_normalised = dl.masks if nm.rhits_normalised is None else nm.rhits_normalised
                        dl.fsky = np.mean(l2OBD_Transformer.get_ninvp(cf)[0][1]) ## calculating fsky, but quite expensive. and if ninvp changes, this could have negative effect on fsky calc
                    else:
                        dl.fsky = 1.0
                    # spectrum_type
                    dl.spectrum_type = nm.spectrum_type

                    dl.OBD = nm.OBD
                    # nlev_t
                    dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
                    # nlev_p
                    dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                    

                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.obd_libdir = od.libdir
                    dl.obd_rescale = od.rescale
                    if cf.noisemodel.ninvjob_geometry == 'healpix_geometry':
                        dl.ninvjob_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_geometry = dl.ninvjob_geometry.restrict(*thtbounds, northsouth_sym=False)
                    dl.tpl = template_dense(dl.lmin_teb[2], dl.ninvjob_geometry, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)

        
                @log_on_start(logging.DEBUG, "_process_Data() started")
                @log_on_end(logging.DEBUG, "_process_Data() finished")       
                def _process_Data(dl, da):
                    l2base_Transformer.process_Data(dl, da, cf)
                    # transferfunction
                    dl.transferfunction = da.transferfunction
                    if dl.transferfunction == 'gauss_no_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    elif dl.transferfunction == 'gauss_with_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    dl.ttebl = {'t': transf_tlm, 'e': transf_elm, 'b':transf_blm}

                    # Isotropic approximation to the filtering (used eg for response calculations)
                    ftl_len = cli(dl.cls_len['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_len = cli(dl.cls_len['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_len = cli(dl.cls_len['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_len = {'t': ftl_len, 'e': fel_len, 'b':fbl_len}

                    # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                    ftl_unl = cli(dl.cls_unl['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_unl = cli(dl.cls_unl['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_unl = cli(dl.cls_unl['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_unl = {'t': ftl_unl, 'e': fel_unl, 'b':fbl_unl}
        

                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):

                    dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                    dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                    # blt_pert
                    dl.blt_pert = qe.blt_pert
                    # qe_tasks
                    dl.qe_tasks = qe.tasks
                    # QE_subtract_meanfield
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
                    ## if QE_subtract_meanfield is True, mean-field needs to be calculated either way.
                    ## also move calc_meanfield to the front, so it is calculated first. The following lines assume that all other tasks are in the right order...
                    ## TODO allow user to provide task-list unsorted
                    if 'calc_phi' in dl.qe_tasks:
                        if dl.QE_subtract_meanfield:
                            if 'calc_meanfield' not in dl.qe_tasks:
                                dl.qe_tasks = ['calc_meanfield'].append(dl.qe_tasks)
                            elif dl.qe_tasks[0] != 'calc_meanfield':
                                if dl.qe_tasks[1] == 'calc_meanfield':
                                    buffer = copy.deepcopy(dl.qe_tasks[0])
                                    dl.qe_tasks[0] = 'calc_meanfield'
                                    dl.qe_tasks[1] = buffer
                                else:
                                    buffer = copy.deepcopy(dl.qe_tasks[0])
                                    dl.qe_tasks[0] = 'calc_meanfield'
                                    dl.qe_tasks[2] = buffer
                    # lmax_qlm
                    dl.lm_max_qlm = qe.lm_max_qlm

                    dl.qlm_type = qe.qlm_type

                    # ninvjob_qe_geometry
                    if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                        # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                        # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                        dl.ninvjob_qe_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(1), np.arccos(-1))
                        dl.ninvjob_qe_geometry = dl.ninvjob_qe_geometry.restrict(*thtbounds, northsouth_sym=False)
                    elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                        dl.ninvjob_qe_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_qe_geometry = dl.ninvjob_qe_geometry.restrict(*thtbounds, northsouth_sym=False)

                    # cg_tol
                    dl.cg_tol = qe.cg_tol

                    # chain
                    if qe.chain == None:
                        dl.chain_descr = lambda a,b: None
                        dl.chain_model = dl.chain_descr
                    else:
                        dl.chain_model = qe.chain
                        dl.chain_model.p3 = dl.sims_nside
                        
                        if dl.chain_model.p6 == 'tr_cg':
                            _p6 = cd_solve.tr_cg
                        if dl.chain_model.p7 == 'cache_mem':
                            _p7 = cd_solve.cache_mem()
                        dl.chain_descr = lambda p2, p5 : [
                            [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

                    # filter
                    dl.qe_filter_directional = qe.filter_directional

                    # qe_cl_analysis
                    dl.cl_analysis = qe.cl_analysis


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    # tasks
                    dl.it_tasks = it.tasks
                    # lmaxunl
                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    # chain
                    dl.it_chain_model = it.chain
                    dl.it_chain_model.p3 = dl.sims_nside
                    if dl.it_chain_model.p6 == 'tr_cg':
                        _p6 = cd_solve.tr_cg
                    if dl.it_chain_model.p7 == 'cache_mem':
                        _p7 = cd_solve.cache_mem()
                    dl.it_chain_descr = lambda p2, p5 : [
                        [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
                    
                    # lenjob_geometry
                    # TODO lm_max_unl should be a bit larger here for geometry, perhaps add + X (~500)
                    # dl.lenjob_geometry = lug.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2, zbounds=dl.zbounds_len) if it.lenjob_geometry == 'thin_gauss' else None
                    dl.lenjob_geometry = lug.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2)
                    # lenjob_pbgeometry
                    dl.lenjob_pbgeometry = lug.pbdGeometry(dl.lenjob_geometry, lug.pbounds(dl.pb_ctr, dl.pb_extent)) if it.lenjob_pbgeometry == 'pbdGeometry' else None
                    
                    if dl.version == '' or dl.version == None:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                    else:
                        dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                    # cg_tol
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 1 else it.cg_tol*0.1
                    # filter
                    dl.it_filter_directional = it.filter_directional
                    # itmax
                    dl.itmax = it.itmax
                    # iterator_typ
                    dl.iterator_typ = it.iterator_typ

                    # mfvar
                    if it.mfvar == 'same' or it.mfvar == '':
                        dl.mfvar = None
                    elif it.mfvar.startswith('/'):
                        if os.path.isfile(it.mfvar):
                            dl.mfvar = it.mfvar
                        else:
                            log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
                            sys.exit()
                    # soltn_cond
                    dl.soltn_cond = it.soltn_cond
                    # stepper
                    dl.stepper_model = it.stepper
                    if dl.stepper_model.typ == 'harmonicbump':
                        dl.stepper_model.lmax_qlm = dl.lm_max_qlm[0]
                        dl.stepper_model.mmax_qlm = dl.lm_max_qlm[1]
                        dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, a=dl.stepper_model.a, b=dl.stepper_model.b, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                        # dl.stepper = steps.nrstep(dl.lm_max_qlm[0], dl.lm_max_qlm[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
                    dl.ffi = deflection(dl.lenjob_geometry, np.zeros(shape=hp.Alm.getsize(*dl.lm_max_qlm)), dl.lm_max_qlm[1], numthreads=dl.tr, verbosity=dl.verbose, epsilon=dl.epsilon)
                
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Data(dl, cf.data)
                if dl.OBD:
                    _process_OBD(dl, cf.obd)
                else:
                    dl.tpl = None
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                if 'smoothed_phi_empiric_halofit' in cf.analysis.cpp:
                    dl.cpp = np.load(cf.analysis.cpp)[:dl.lm_max_qlm[0] + 1,1]
                else:
                    dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] ## TODO could be added via 'fiducial' parameter in dlensalot config for user
                dl.cpp[:dl.Lmin] *= 0.

                if dl.it_filter_directional == 'anisotropic':
                    # ninvjob_geometry
                    if cf.noisemodel.ninvjob_geometry == 'healpix_geometry':
                        dl.ninvjob_geometry = lug.Geom.get_healpix_geometry(dl.sims_nside)
                        thtbounds = (np.arccos(dl.zbounds[1]), np.arccos(dl.zbounds[0]))
                        dl.ninvjob_geometry = dl.ninvjob_geometry.restrict(*thtbounds, northsouth_sym=False)

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        return MAP_lr(extract())


    def build_OBD_builder(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        @log_on_start(logging.INFO, "extract() started")
        @log_on_end(logging.INFO, "extract() finished")
        def extract():
            def _process_components(dl):
                @log_on_start(logging.INFO, "_process_Computing() started")
                @log_on_end(logging.INFO, "_process_Computing() finished")
                def _process_Computing(dl, co):
                    dl.tr = int(os.environ.get('OMP_NUM_THREADS', co.OMP_NUM_THREADS))


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    dl.TEMP_suffix = an.TEMP_suffix,
                    dl.mask_fn = an.mask
                    dl.lmin_teb = an.lmin_teb


                @log_on_start(logging.DEBUG, "_process_OBD() started")
                @log_on_end(logging.DEBUG, "_process_OBD() finished")
                def _process_OBD(dl, od):
                    dl.nside = od.nside
                    dl.libdir = od.libdir
                    dl.nlev_dep = od.nlev_dep
                    dl.beam = od.beam
                    dl.nside = od.nside
                    dl.lmax = od.lmax
                    dl.rescale = od.rescale

                    if os.path.isfile(opj(dl.libdir,'tniti.npy')):
                        # TODO need to test if it is the right tniti.npy
                        log.warning("tniti.npy in destination dir {} already exists.".format(dl.libdir))
                        log.warning("Exiting. Please check your settings.")


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, nm):
                    dl.lmin_b = dl.lmin_teb[2]
                    dl.geom = lug.Geom.get_healpix_geometry(dl.sims_nside)
                    dl.masks, dl.rhits_map = l2OBD_Transformer.get_masks(cf)
                    dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                    dl.ninv_p_desc = l2OBD_Transformer.get_ninvp(cf, dl.nside)
                    


                dl.TEMP = transform(cf, l2T_Transformer())
                # dl.TEMP = dl.libdir

                _process_Computing(dl, cf.computing)
                _process_Analysis(dl, cf.analysis)
                _process_OBD(dl, cf.obd)
                _process_Noisemodel(dl, cf.noisemodel)
                
                return dl

            dl = DLENSALOT_Concept()
            _process_components(dl)
            return dl

        return MAP_lr(extract())


    def build_delenser(self, cf):
        """Transformer for generating a delensalot model for the lensing reconstruction jobs (QE and MAP)
        """
        @log_on_start(logging.INFO, "extract() started")
        @log_on_end(logging.INFO, "extract() finished")
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
                    log.info("OMP_NUM_THREADS: {} and {}".format(dl.tr, os.environ.get('OMP_NUM_THREADS')))


                @log_on_start(logging.DEBUG, "_process_Analysis() started")
                @log_on_end(logging.DEBUG, "_process_Analysis() finished")
                def _process_Analysis(dl, an):
                    # super(l2base_Transformer, self)
                    l2base_Transformer.process_Analysis(dl, an, cf)


                @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
                @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
                def _process_Noisemodel(dl, an):
                    dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
                    dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)


                @log_on_start(logging.DEBUG, "_process_Data() started")
                @log_on_end(logging.DEBUG, "_process_Data() finished")       
                def _process_Data(dl, da):
                    l2base_Transformer.process_Data(dl, da, cf)
                    dl.data_type = 'map'
                    dl.data_field = 'qu'
                    # transferfunction
                    dl.transferfunction = da.transferfunction
                    if dl.transferfunction == 'gauss_no_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    elif dl.transferfunction == 'gauss_with_pixwin':
                        # Fiducial model of the transfer function
                        transf_tlm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                        transf_elm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                        transf_blm = gauss_beam(df.a2r(da.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(da.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
                    dl.ttebl = {'t': transf_tlm, 'e': transf_elm, 'b':transf_blm}

                    # Isotropic approximation to the filtering (used eg for response calculations)
                    ftl_len = cli(dl.cls_len['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_len = cli(dl.cls_len['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_len = cli(dl.cls_len['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(da.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_len = {'t': ftl_len, 'e': fel_len, 'b':fbl_len}

                    # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                    ftl_unl = cli(dl.cls_unl['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
                    fel_unl = cli(dl.cls_unl['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
                    fbl_unl = cli(dl.cls_unl['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
                    dl.ftebl_unl = {'t': ftl_unl, 'e': fel_unl, 'b':fbl_unl}


                @log_on_start(logging.DEBUG, "_process_Qerec() started")
                @log_on_end(logging.DEBUG, "_process_Qerec() finished")
                def _process_Qerec(dl, qe):

                    dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                    dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                    # blt_pert
                    dl.blt_pert = qe.blt_pert

                    # QE_subtract_meanfield
                    dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True

                    # lmax_qlm
                    dl.lm_max_qlm = qe.lm_max_qlm
                    dl.qlm_type = qe.qlm_type

                    # filter
                    dl.qe_filter_directional = qe.filter_directional


                @log_on_start(logging.DEBUG, "_process_Itrec() started")
                @log_on_end(logging.DEBUG, "_process_Itrec() finished")
                def _process_Itrec(dl, it):
                    # tasks

                    dl.lm_max_unl = it.lm_max_unl
                    dl.lm_max_qlm = it.lm_max_qlm
                    # chain

                    # cg_tol
                    dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1
                    # filter
                    dl.it_filter_directional = it.filter_directional
                    # itmax
                    dl.itmax = it.itmax
                    # iterator_typ
                    dl.iterator_typ = it.iterator_typ
                    # soltn_cond
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
                        dl.base_mask = np.ones(shape=hp.nside2npix(cf.data.nside))
                    noisemodel_rhits_map = df.get_nlev_mask(np.inf, dl.base_mask)
                    if ma.nlevels == None or ma.nlevels == [] or ma.nlevels == False:
                        dl.nlevels = np.array([np.inf])
                    else:
                        dl.nlevels = ma.nlevels
                    dl.masks = {nlevel: [] for nlevel in dl.nlevels} 
                    dl.binmasks = {nlevel: [] for nlevel in dl.nlevels}

                    for nlevel, value in dl.masks.items():
                            dl.masks[nlevel] = df.get_nlev_mask(nlevel, noisemodel_rhits_map)
                            dl.binmasks[nlevel] = np.where(dl.masks[nlevel]>0,1,0)

                    ## TODO if config file contains masks_fn, create new masks
                    # if ma.masks_fn == 'masks':
                    #     dl.mask_ids = np.zeros(shape=len(ma.masks[1]))
                    #     for fni, fn in enumerate(ma.masks[1]):
                    #         if fn == None:
                    #             buffer = np.ones(shape=hp.nside2npix(dl.sims_nside))
                    #             dl.mask_ids[fni] = 1.00
                    #         elif fn.endswith('.fits'):
                    #             buffer = hp.read_map(fn)
                    #         else:
                    #             buffer = np.load(fn)
                    #         _fsky = float("{:0.3f}".format(np.sum(buffer)/len(buffer)))
                    #         dl.mask_ids[fni] = _fsky
                    #         dl.masks[ma.masks[0]].update({_fsky:buffer})
                    #         dl.binmasks[ma.masks[0]].update({_fsky: np.where(dl.masks[ma.masks[0]][_fsky]>0,1,0)})

        
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
                        dl.lmax = 200 #ma.lmax
                        dl.lmax_mask = 3*ma.lmax-1
                        dl.edges = ma.edges
                        dl.edges_center = (dl.edges[1:]+dl.edges[:-1])/2.
                        dl.sha_edges = hashlib.sha256()
                        dl.sha_edges.update((str(dl.edges)).encode())
                        dl.dirid = dl.sha_edges.hexdigest()[:4]
                        dl.ct = dl.clc_templ[np.array(dl.edges, dtype=int)] # TODO marginalising over binrange would probably be better

                    elif dl.binning == 'unbinned':
                        dl.lmax = 200
                        dl.lmax_mask = 3*ma.lmax-1
                        dl.edges = np.arange(0,dl.lmax+2)
                        dl.edges_center = dl.edges[1:]
                        dl.ct = np.ones(shape=len(dl.edges_center))
                        dl.sha_edges = hashlib.sha256()
                        dl.sha_edges.update(('unbinned').encode())
                        dl.dirid = dl.sha_edges.hexdigest()[:4]

                    dl.vers_str = '/{}'.format(dl.version) if dl.version != '' else 'base'
                    dl.TEMP_DELENSED_SPECTRUM = transform(dl, l2T_Transformer())


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


                dl = DLENSALOT_Concept()

                dl.blt_pert = cf.qerec.blt_pert
                _process_Meta(dl, cf.meta)
                _process_Computing(dl, cf.computing)
                _process_Noisemodel(dl, cf.noisemodel)
                _process_Analysis(dl, cf.analysis)
                _process_Data(dl, cf.data)
                _process_Madel(dl, cf.madel)
                _process_Config(dl, cf.config)
                _check_powspeccalculator(dl.cl_calc)

                # Need a few attributes for predictions (like ftebl, lm_max_qlm, ..)
                _process_Qerec(dl, cf.qerec)
                _process_Itrec(dl, cf.itrec)

                dl.cpp = camb_clfile(cf.analysis.cpp)['pp'][:dl.lm_max_qlm[0] + 1] ## TODO could be added via 'fiducial' parameter in dlensalot config for user
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