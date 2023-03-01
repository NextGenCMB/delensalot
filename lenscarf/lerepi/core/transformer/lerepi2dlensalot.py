#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build dlensalot model from configuation file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
import copy
from os.path import join as opj
import importlib


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import numpy as np
import healpy as hp
import hashlib

import plancklens
from plancklens.sims import maps, phas
from plancklens.qcinv import opfilt_pp
from plancklens import qest, qecl, utils
from plancklens.filt import filt_util, filt_cinv, filt_simple
from plancklens.qcinv import cd_solve

from lenspyx.remapping import utils_geom

from lenscarf.core.mpi import check_MPI
from lenscarf.core import mpi
from lenscarf.sims import sims_ffp10
from lenscarf import utils_scarf, utils_sims
from lenscarf.utils import cli, read_map
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt import utils_cinv_p as cinv_p_OBD
import lenscarf.core.handler as lenscarf_handler
from lenscarf.opfilt.bmodes_ninv import template_dense
from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.config.config_helper import data_functions as df, LEREPI_Constants as lc
from lenscarf.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm, DLENSALOT_Concept, DLENSALOT_Chaindescriptor


# TODO swap rhits with ninv


class l2T_Transformer:
    # TODO this needs a big refactoring. Suggest working via cachers
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
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
            TEMP =  opj(os.environ['SCRATCH'], 'dlensalot', cf.data.package_, cf.data.module_.split('.')[-1], _suffix)

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


class l2lensrec_Transformer:
    """_summary_
    """

    def mapper(self, cf):
        return self.build(cf)
        # TODO implement if needed
        # if cf.meta.version == '0.2b':
        #     return self.build(cf)
        
        # elif cf.meta.version == '0.9':
        #     return self.build_v3(cf)

    @check_MPI
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):


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
            # dlm_mod
            dl.dlm_mod_bool = cf.madel.dlm_mod
            # mask
            dl.mask_fn = an.mask
            # key -> k
            dl.k = an.key
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
            dl.cls_unl = utils.camb_clfile(an.cls_unl)
            # if 
            dl.cls_len = utils.camb_clfile(an.cls_len)
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


        @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
        @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
        def _process_Noisemodel(dl, nm):
            # sky_coverage
            dl.sky_coverage = nm.sky_coverage
            # TODO assuming that masked sky comes with a hits-count map. If not, take mask
            if dl.sky_coverage == 'masked':
                # rhits_normalised
                dl.rhits_normalised = dl.masks if nm.rhits_normalised is None else nm.rhits_normalised
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
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(cf.data.nside, zbounds=dl.zbounds)
            dl.tpl = template_dense(dl.lmin_teb[2], dl.ninvjob_geometry, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)
  

        @log_on_start(logging.DEBUG, "_process_Data() started")
        @log_on_end(logging.DEBUG, "_process_Data() finished")
        def _process_Data(dl, da):
            # package_
            _package = da.package_
            # module_
            _module = da.module_
            # class_
            _class = da.class_
            # class_parameters -> sims
            ## TODO what if user wants to add own sims_module outside of dlensalot?
            _dataclass_parameters = da.class_parameters
            dl.class_parameters = da.class_parameters
            if 'fg' in _dataclass_parameters:
                dl.fg = _dataclass_parameters['fg']
            _sims_full_name = '{}.{}'.format(_package, _module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl._sims = getattr(_sims_module, _class)(**_dataclass_parameters)
            ## get_sim_pmap comes from sims module directly
            # -> nothing to do here
            ## sims parameter come from configuration file
            dl._sims.beam = da.beam
            dl._sims.lmax_transf = da.lmax_transf
            dl._sims.nlev_t = da.nlev_t
            dl._sims.nlev_p = da.nlev_p
            dl._sims.nside = da.nside
            ## get_sim_pmap comes from plancklens.maps wrapper
            if 'lib_dir' in da.class_parameters:
                pix_phas = phas.pix_lib_phas(da.class_parameters['lib_dir'], 3, (hp.nside2npix(dl._sims.nside),))
            else:
                pix_phas = phas.pix_lib_phas(da.class_parameters['cacher'].lib_dir, 3, (hp.nside2npix(dl._sims.nside),))
            transf_dat = gauss_beam(dl._sims.beam / 180 / 60 * np.pi, lmax=dl._sims.lmax_transf)
            dl.sims = maps.cmb_maps_nlev(dl._sims, transf_dat, dl._sims.nlev_t, dl._sims.nlev_p, dl._sims.nside, pix_lib_phas=pix_phas)

            # transferfunction
            dl.transferfunction = da.transferfunction
            if dl.transferfunction == 'gauss_no_pixwin':
                # Fiducial model of the transfer function
                transf_tlm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                transf_elm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                transf_blm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
            elif dl.transferfunction == 'gauss_with_pixwin':
                # Fiducial model of the transfer function
                transf_tlm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl._sims.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[0])
                transf_elm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl._sims.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[1])
                transf_blm = gauss_beam(df.a2r(dl._sims.beam), lmax=dl.lm_max_ivf[0]) * hp.pixwin(dl._sims.nside, lmax=dl.lm_max_ivf[0]) * (np.arange(dl.lm_max_ivf[0] + 1) >= dl.lmin_teb[2])
            dl.ttebl = {'t': transf_tlm, 'e': transf_elm, 'b':transf_blm}

            # Isotropic approximation to the filtering (used eg for response calculations)
            ftl_len = cli(dl.cls_len['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl._sims.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
            fel_len = cli(dl.cls_len['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl._sims.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
            fbl_len = cli(dl.cls_len['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl._sims.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
            dl.ftebl_len = {'t': ftl_len, 'e': fel_len, 'b':fbl_len}

            # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
            ftl_unl = cli(dl.cls_unl['tt'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.ttebl['t'] ** 2)) * (dl.ttebl['t'] > 0)
            fel_unl = cli(dl.cls_unl['ee'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['e'] ** 2)) * (dl.ttebl['e'] > 0)
            fbl_unl = cli(dl.cls_unl['bb'][:dl.lm_max_ivf[0] + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.ttebl['b'] ** 2)) * (dl.ttebl['b'] > 0)
            dl.ftebl_unl = {'t': ftl_unl, 'e': fel_unl, 'b':fbl_unl}


        @log_on_start(logging.DEBUG, "_process_Qerec() started")
        @log_on_end(logging.DEBUG, "_process_Qerec() finished")
        def _process_Qerec(dl, qe):
            # blt_pert
            dl.blt_pert = qe.blt_pert
            # qe_tasks
            dl.qe_tasks = qe.tasks
            # QE_subtract_meanfield
            dl.QE_subtract_meanfield = False if dl.version == 'noMF' else True
            ## if QE_subtract_meanfield is True, mean-field needs to be calculated either way.
            ## also move calc_meanfield to the front, so it is calculated first. The following lines assume that all other tasks are in the right order...
            ## TODO allow user to provide task-list unordered
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
            dl.qe_lm_max_qlm = qe.lm_max_qlm

            # ninvjob_qe_geometry
            if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl._sims.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl._sims.nside, zbounds=dl.zbounds)
            # cg_tol
            dl.cg_tol = qe.cg_tol

            # chain
            if qe.chain == None:
                dl.chain_descr = lambda a,b: None
                dl.chain_model = dl.chain_descr
            else:
                dl.chain_model = qe.chain
                dl.chain_model.p3 = dl._sims.nside
                
                if dl.chain_model.p6 == 'tr_cg':
                    _p6 = cd_solve.tr_cg
                if dl.chain_model.p7 == 'cache_mem':
                    _p7 = cd_solve.cache_mem()
                dl.chain_descr = lambda p2, p5 : [
                    [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

            # filter
            dl.qe_filter_directional = qe.filter_directional
            if dl.qe_filter_directional == 'anisotropic':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                lmax_plm = qe.lm_max_qlm[0]
                # TODO filters can be initialised with both, ninvX_desc and ninv_X. But Plancklens' hashcheck will complain if it changed since shapes are different. Not sure which one I want to use in the future..
                # TODO using ninv_X possibly causes hashcheck to fail, as v1 == v2 won't work on arrays.
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), lmax_plm, dl._sims.nside, dl.cls_len, dl.ttebl['t'], dl.ninvt_desc,
                    marge_monopole=True, marge_dipole=True, marge_maps=[])
                if dl.OBD:
                    transf_elm_loc = gauss_beam(dl._sims.beam / 180 / 60 * np.pi, lmax=lmax_plm)
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), lmax_plm, dl._sims.nside, dl.cls_len, transf_elm_loc[:lmax_plm+1], dl.ninvp_desc, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(lmax_plm, dl.cg_tol), bmarg_lmax=dl.lmin_teb[2], zbounds=dl.zbounds, _bmarg_lib_dir=dl.obd_libdir, _bmarg_rescal=dl.obd_rescale, sht_threads=dl.tr)
                else:
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), lmax_plm, dl._sims.nside, dl.cls_len, dl.ttebl['e'], dl.ninvp_desc,
                        chain_descr=dl.chain_descr(lmax_plm, dl.cg_tol), transf_blm=dl.ttebl['b'], marge_qmaps=(), marge_umaps=())

                _filter_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                _ftl_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_teb[0])
                _fel_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_teb[1])
                _fbl_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_teb[2])
                dl.ivfs = filt_util.library_ftl(_filter_raw, lmax_plm, _ftl_rs, _fel_rs, _fbl_rs)
            elif dl.qe_filter_directional == 'isotropic':
                dl.ivfs = filt_simple.library_fullsky_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl._sims.nside, dl.ttebl, dl.cls_len, dl.ftebl_len['t'], dl.ftebl_len['e'], dl.ftebl_len['b'], cache=True)
                # elif dl._sims.data_type == 'alm':
                    # dl.ivfs = filt_simple.library_fullsky_alms_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.ttebl, dl.cls_len, dl.ftl, dl.fel, dl.fbl, cache=True)
                
            # qlms
            if qe.qlm_type == 'sepTP':
                dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl._sims.nside, lmax_qlm=dl.qe_lm_max_qlm[0])
            # qe_cl_analysis
            dl.cl_analysis = qe.cl_analysis
            if qe.cl_analysis == True:
                # TODO fix numbers for mc ocrrection and total nsims
                dl.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                        np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
                dl.ds_dict = { k : -1 for k in range(300)}

                dl.ivfs_d = filt_util.library_shuffle(dl.ivfs, dl.ds_dict)
                dl.ivfs_s = filt_util.library_shuffle(dl.ivfs, dl.ss_dict)

                dl.qlms_ds = qest.library_sepTP(opj(dl.TEMP, 'qlms_ds'), dl.ivfs, dl.ivfs_d, dl.cls_len['te'], dl._sims.nside, lmax_qlm=dl.qe_lm_max_qlm[0])
                dl.qlms_ss = qest.library_sepTP(opj(dl.TEMP, 'qlms_ss'), dl.ivfs, dl.ivfs_s, dl.cls_len['te'], dl._sims.nside, lmax_qlm=dl.qe_lm_max_qlm[0])

                dl.mc_sims_bias = np.arange(60, dtype=int)
                dl.mc_sims_var  = np.arange(60, 300, dtype=int)

                dl.qcls_ds = qecl.library(opj(dl.TEMP, 'qcls_ds'), dl.qlms_ds, dl.qlms_ds, np.array([]))  # for QE RDN0 calculations
                dl.qcls_ss = qecl.library(opj(dl.TEMP, 'qcls_ss'), dl.qlms_ss, dl.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
                dl.qcls_dd = qecl.library(opj(dl.TEMP, 'qcls_dd'), dl.qlms_dd, dl.qlms_dd, dl.mc_sims_bias)


        @log_on_start(logging.DEBUG, "_process_Itrec() started")
        @log_on_end(logging.DEBUG, "_process_Itrec() finished")
        def _process_Itrec(dl, it):
            # tasks
            dl.it_tasks = it.tasks
            # lmaxunl
            dl.lm_max_unl = it.lm_max_unl
            dl.it_lm_max_qlm = it.lm_max_qlm
            # chain
            dl.it_chain_model = DLENSALOT_Chaindescriptor()
            dl.it_chain_model.p3 = dl._sims.nside
            if dl.it_chain_model.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if dl.it_chain_model.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.it_chain_descr = lambda p2, p5 : [
                [dl.it_chain_model.p0, dl.it_chain_model.p1, p2, dl.it_chain_model.p3, dl.it_chain_model.p4, p5, _p6, _p7]]
            # lenjob_geometry
            # TODO lm_max_unl should be a bit larger here for geometry, perhaps add + X (~500)
            # dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2, zbounds=dl.zbounds_len) if it.lenjob_geometry == 'thin_gauss' else None
            dl.lenjob_geometry = utils_geom.Geom.get_thingauss_geometry(dl.lm_max_unl[0], 2)
            # lenjob_pbgeometry
            dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent)) if it.lenjob_pbgeometry == 'pbdGeometry' else None
            
            ## tasks -> mf_dirname
            if "calc_meanfield" in dl.it_tasks or 'calc_blt' in dl.it_tasks:
                if dl.version == '' or dl.version == None:
                    dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                else:
                    dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                if not os.path.isdir(dl.mf_dirname) and mpi.rank == 0:
                    os.makedirs(dl.mf_dirname)
            # cg_tol
            dl.it_cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1
            # filter
            dl.it_filter_directional = it.filter_directional
            # sims -> sims_MAP
            if it.filter_directional == 'anisotropic':
                dl.sims_MAP = utils_sims.ztrunc_sims(dl.sims, dl._sims.nside, [dl.zbounds])
            elif it.filter_directional == 'isotropic':
                dl.sims_MAP = dl.sims
            # itmax
            dl.itmax = it.itmax
            # iterator_typ
            dl.iterator_typ = it.iterator_typ
            # LENSRES
            dl.lensres = it.lensres

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
                # TODO undo hardcoding, make it userchoice
                # TODO this should be checked via validator accordingly
                if dl.stepper_model.lmax_qlm == -1 and dl.stepper_model.mmax_qlm == -1:
                    dl.stepper_model.lmax_qlm = dl.it_lm_max_qlm[0]
                    dl.stepper_model.mmax_qlm = dl.it_lm_max_qlm[1]
                dl.stepper = steps.harmonicbump(dl.stepper_model.lmax_qlm, dl.stepper_model.mmax_qlm, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
                # dl.stepper = steps.nrstep(dl.it_lm_max_qlm[0], dl.it_lm_max_qlm[1], val=0.5) # handler of the size steps in the MAP BFGS iterative search
            

        dl = DLENSALOT_Concept()    
        _process_Meta(dl, cf.meta)
        _process_Computing(dl, cf.computing)
        _process_Analysis(dl, cf.analysis)
        _process_Noisemodel(dl, cf.noisemodel)
        if dl.OBD:
            _process_OBD(dl, cf.obd)
        else:
            dl.tpl = None
        _process_Data(dl, cf.data)
        _process_Qerec(dl, cf.qerec)
        _process_Itrec(dl, cf.itrec)


        # TODO here goes anything that needs info from different classes

        # fiducial

        dl.cpp = utils.camb_clfile(cf.analysis.cpp)['pp'][:dl.qe_lm_max_qlm[0] + 1]  ## TODO could be added via 'fiducial' parameter in dlensalot config for user
        dl.cpp[:dl.Lmin] *= 0.

        if dl.it_filter_directional == 'anisotropic':
            # ninvjob_geometry
            if cf.noisemodel.ninvjob_geometry == 'healpix_geometry':
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(dl._sims.nside, zbounds=dl.zbounds)


        # if mpi.rank == 0:
        #     _str = "\nConfiguration:"+3*'\n---------------------------------------------------\n'
        #     for key, val in dl.__dict__.items():
        #         keylen = len(str(key))
        #         if type(val) in [list, np.ndarray, np.array, dict]:
        #             _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(type(val))
        #         else:
        #             _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(val)
        #         _str += '\n'
        #     _str += 3*'---------------------------------------------------\n'
        #     log.info(_str)

        return dl


class l2OBD_Transformer:
    """Extracts all parameters needed for building consistent OBD
    """

    @check_MPI
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):


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

            if os.path.isfile(opj(dl.libdir,'tniti.npy')):
                # TODO need to test if it is the right tniti.npy
                log.warning("tniti.npy in destination dir {} already exists.".format(dl.libdir))
                log.warning("Exiting. Please check your settings.")


        @log_on_start(logging.DEBUG, "_process_Noisemodel() started")
        @log_on_end(logging.DEBUG, "_process_Noisemodel() finished")
        def _process_Noisemodel(dl, nm):
            dl.lmin_b = dl.lmin_teb[2]
            dl.geom = utils_scarf.Geom.get_healpix_geometry(dl.nside)
            dl.masks, dl.rhits_map = l2OBD_Transformer.get_masks(cf)
            dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
            dl.ninv_p_desc = l2OBD_Transformer.get_ninvp(cf, dl.nside)

            b_transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax) # TODO ninv_p doesn't depend on this anyway, right?
            dl.ninv_p = np.array(opfilt_pp.alm_filter_ninv(dl.ninv_p_desc, b_transf, marge_qmaps=(), marge_umaps=()).get_ninv())
            

        dl = DLENSALOT_Concept()

        dl.TEMP = transform(cf, l2T_Transformer())
        # dl.TEMP = dl.libdir

        _process_Computing(dl, cf.computing)
        _process_Analysis(dl, cf.analysis)
        _process_OBD(dl, cf.obd)
        _process_Noisemodel(dl, cf.noisemodel)
        
        return dl


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


class l2d_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @check_MPI
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):


        def _process_Madel(dl, ma):

            dl.data_from_CFS = ma.data_from_CFS
            dl.k = cf.analysis.K
            dl.version = cf.analysis.V

            dl.its = [0] if ma.iterations == [] else ma.iterations
            dl.simidxs_mf = cf.analysis.simidxs_mf if cf.analysis.simidxs_mf != [] else dl.simidxs
            dl.Nmf = len(dl.simidxs_mf)
            dl.Nblt = len(cf.madel.simidxs_mblt)
            if 'fg' in cf.data.class_parameters:
                dl.fg = cf.data.class_parameters['fg']
            dl._package = cf.data.package_
            dl._module = cf.data.module_
            dl._class = cf.data.class_
            dl.class_parameters = cf.data.class_parameters
            _sims_full_name = '{}.{}'.format(dl._package, dl._module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl._sims = getattr(_sims_module, dl._class)(**dl.class_parameters)
            dl.sims = dl._sims.sims

            dl.ec = getattr(_sims_module, 'experiment_config')()
            dl.data_type = cf.data.data_type
            dl.data_field = cf.data.data_field


            # TODO hack. this is only needed to access old s08b data
            # Remove and think of a better way of including old data without existing config file
            dl.TEMP = transform(cf, l2T_Transformer())

            if ma.libdir_it is None:
                dl.libdir_iterators = lambda qe_key, simidx, version: opj(dl.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
            else:
                dl.libdir_iterators = 'overwrite'
            dl.analysis_path = dl.TEMP.split('/')[-1]

            if cf.noisemodel.rhits_normalised is not None:
                _mask_path = cf.noisemodel.rhits_normalised[0]
                dl.base_mask = np.nan_to_num(hp.read_map(_mask_path))
            else:
                dl.base_mask = np.ones(shape=hp.nside2npix(cf.data.nside))
            noisemodel_rhits_map = df.get_nlev_mask(np.inf, dl.base_mask)
            noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf

            if ma.masks != None:
                innermaskid = 10
                if ma.ringmask:
                    _innermask = df.get_nlev_mask(innermaskid, noisemodel_rhits_map)
                else:
                    _innermask = 0
                dl.masks = dict({ma.masks[0]:{}})
                dl.binmasks = dict({ma.masks[0]:{}})
                dl.mask_ids = ma.masks[1]
                if ma.masks[0] == 'nlevels': 
                    for mask_id in dl.mask_ids:
                        buffer = df.get_nlev_mask(mask_id, noisemodel_rhits_map)
                        if mask_id > innermaskid:
                            innermask = np.copy(_innermask)
                        else:
                            innermask = 0
                        dl.masks[ma.masks[0]].update({mask_id:buffer-innermask})
                        dl.binmasks[ma.masks[0]].update({mask_id: np.where(dl.masks[ma.masks[0]][mask_id]>0,1,0)})
                elif ma.masks[0] == 'masks':
                    dl.mask_ids = np.zeros(shape=len(ma.masks[1]))
                    for fni, fn in enumerate(ma.masks[1]):
                        if fn == None:
                            buffer = np.ones(shape=hp.nside2npix(dl._sims.nside))
                            dl.mask_ids[fni] = 1.00
                        elif fn.endswith('.fits'):
                            buffer = hp.read_map(fn)
                        else:
                            buffer = np.load(fn)
                        _fsky = float("{:0.3f}".format(np.sum(buffer)/len(buffer)))
                        dl.mask_ids[fni] = _fsky
                        dl.masks[ma.masks[0]].update({_fsky:buffer})
                        dl.binmasks[ma.masks[0]].update({_fsky: np.where(dl.masks[ma.masks[0]][_fsky]>0,1,0)})
            else:
                dl.masks = {"no":{1.00:np.ones(shape=hp.nside2npix(dl._sims.nside))}}
                dl.mask_ids = np.array([1.00])

            dl.beam = cf.data.beam
            dl.lmax_transf = cf._sims.lmax_transf
            if cf.analysis.STANDARD_TRANSFERFUNCTION == True:
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf)
            elif cf.analysis.STANDARD_TRANSFERFUNCTION == 'with_pixwin':
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf) * hp.pixwin(cf.data.nside, lmax=dl.lmax_transf)
            else:
                log.info("Don't understand your STANDARD_TRANSFERFUNCTION: {}".format(cf.analysis.STANDARD_TRANSFERFUNCTION))
            
            if ma.Cl_fid == 'ffp10':
                dl.cls_unl = utils.camb_clfile(cf.analysis.cls_unl)
                dl.cls_len = utils.camb_clfile(cf.analysis.cls_len)
                dl.clg_templ = dl.cls_len['ee']
                dl.clc_templ = dl.cls_len['bb']
                dl.clg_templ[0] = 1e-32
                dl.clg_templ[1] = 1e-32
            pert_mod_string = ''
            dl.blt_pert = ma.blt_pert
            if dl.blt_pert == True:
                pert_mod_string = 'pertblens'
            dl.binning = ma.binning
            if dl.binning == 'binned':
                dl.lmax = ma.lmax
                dl.lmax_mask = 3*dl.lmax-1
                dl.edges = []
                dl.edges_id = []
                if ma.edges != -1:
                    if 'cmbs4' in ma.edges:
                        dl.edges.append(lc.cmbs4_edges)
                        dl.edges_id.append('cmbs4')
                    if 'ioreco' in ma.edges:
                        dl.edges.append(lc.ioreco_edges) 
                        dl.edges_id.append('ioreco')
                    if 'lowell' in ma.edges:
                        dl.edges.append(lc.lowell_edges) 
                        dl.edges_id.append('lowell')
                    elif 'fs' in ma.edges:
                        dl.edges.append(lc.fs_edges)
                        dl.edges_id.append('fs')
                dl.edges = np.array(dl.edges)
                dl.sha_edges = [hashlib.sha256() for n in range(len(dl.edges))]
                for n in range(len(dl.edges)):
                    dl.sha_edges[n].update((str(dl.edges[n]) + pert_mod_string + cf.madel.ringmask*'ringmask').encode())
                dl.dirid = [dl.sha_edges[n].hexdigest()[:4] for n in range(len(dl.edges))]
                dl.edges_center = np.array([(e[1:]+e[:-1])/2 for e in dl.edges])
                dl.ct = np.array([[dl.clc_templ[np.array(ec,dtype=int)]for ec in edge] for edge in dl.edges_center])
            elif dl.binning == 'unbinned':
                dl.lmax = 200
                dl.lmax_mask = 6*dl.lmax-1
                dl.edges = np.array([np.arange(0,dl.lmax+2)])
                dl.edges_id = [dl.binning]
                dl.edges_center = dl.edges[:,1:]
                dl.ct = np.ones(shape=len(dl.edges_center))
                dl.sha_edges = [hashlib.sha256()]
                dl.sha_edges[0].update(('unbinned'+pert_mod_string + cf.madel.ringmask*'ringmask').encode())
                dl.dirid = [dl.sha_edges[0].hexdigest()[:4]]
            else:
                log.info("Don't understand your spectrum type")
                sys.exit()

            dl.vers_str = '/{}'.format(dl.version) if dl.version != '' else 'base'
            dl.TEMP_DELENSED_SPECTRUM = transform(dl, l2T_Transformer())
            for dir_id in dl.dirid:
                if mpi.rank == 0:
                    if not(os.path.isdir(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dir_id))):
                        os.makedirs(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dir_id))

            # TODO II
            # TODO fn needs changing
            dl.dlm_mod_bool = ma.dlm_mod[0]
            dl.dlm_mod_fnsuffix = ma.dlm_mod[1]
            dl.calc_via_MFsplitset = bool(ma.dlm_mod[1])

            dl.subtract_mblt = ma.subtract_mblt[0]
            dl.simidxs_mblt = ma.simidxs_mblt
            dl.Nmblt = len(dl.simidxs_mblt)
            dl.calc_via_mbltsplitset = bool(ma.subtract_mblt[1])
            
            if dl.binning == 'binned':
                if dl.dlm_mod_bool:
                    if dl.subtract_mblt or dl.calc_via_MFsplitset or dl.calc_via_mbltsplitset:
                        splitset_fnsuffix = dl.subtract_mblt * '_mblt' + dl.calc_via_MFsplitset * '_MFsplit'  + dl.calc_via_mbltsplitset * '_mbltsplit'
                    else:
                        splitset_fnsuffix = ''
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod{}{}'.format(dl.dlm_mod_fnsuffix, splitset_fnsuffix), fg)
                else:
                    if dl.subtract_mblt or dl.calc_via_MFsplitset or dl.calc_via_mbltsplitset:
                        splitset_fnsuffix = dl.subtract_mblt * '_mblt' + dl.calc_via_MFsplitset * '_MFsplit'  + dl.calc_via_mbltsplitset * '_mbltsplit'
                    else:
                        splitset_fnsuffix = ''
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_fg%s_res2b3acm%s.npy'%(idx, fg, splitset_fnsuffix)
            else:
                if dl.subtract_mblt or dl.calc_via_MFsplitset or dl.calc_via_mbltsplitset:
                    log.error("Implement for unbinned if needed")
                    sys.exit()
                if dl.dlm_mod_bool:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod{}'.format(dl.dlm_mod_fnsuffix), fg)
                else:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_fg%s_res2b3acm.npy'%(idx, fg)

            if ma.spectrum_calculator == None:
                log.info("Using Healpy as powerspectrum calculator")
                dl.cl_calc = hp
            else:
                dl.cl_calc = ma.spectrum_calculator       


        def _process_Config(dl, co):
            if co.outdir_plot_rel:
                dl.outdir_plot_rel = co.outdir_plot_rel
            else:
                dl.outdir_plot_rel = '{}/{}'.format(cf.data.module_.split('.')[2],cf.data.module_.split('.')[-1])
                    
            if co.outdir_plot_root:
                dl.outdir_plot_root = co.outdir_plot_root
            else:
                dl.outdir_plot_root = os.environ['HOME']
            
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

        dl.blt_pert = cf.itrec.blt_pert

        _process_Madel(dl, cf.madel)
        _process_Config(dl, cf.config)
        _check_powspeccalculator(dl.cl_calc)

        dl.prediction = dict()
        for key0 in ['N0', 'N1', 'cl_del']:
            if key0 not in dl.prediction:
                dl.prediction[key0] = dict()
            for key4 in dl.mask_ids + ['fs']:
                if key4 not in dl.prediction[key0]:
                    dl.prediction[key0][key4] = dict()
                for key1 in ['QE', "MAP"]:
                    if key1 not in dl.prediction[key0][key4]:
                        dl.prediction[key0][key4][key1] = dict()
                    for key2 in ['N', 'N_eff']:
                        if key2 not in dl.prediction[key0][key4][key1]:
                                dl.prediction[key0][key4][key1][key2] = np.array([], dtype=np.complex128)
        return dl

      
class l2i_Transformer:

    @check_MPI
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):


        def _process_X(dl):
            dl.data = dict()
            for key0 in ['cs-cmb', 'cs-cmb-noise', 'fg', 'noise', 'noise_eff', 'mean-field', 'cmb_len', 'BLT', 'cs', 'pred', 'pred_eff', 'BLT_QE', 'BLT_MAP', 'BLT_QE_avg', 'BLT_MAP_avg']:
                if key0 not in dl.data:
                    dl.data[key0] = dict()
                for key4 in dl.mask_ids + ['fs']:
                    if key4 not in dl.data[key0]:
                        dl.data[key0][key4] = dict()
                    for key1 in ['map', 'alm', 'cl', 'cl_patch', 'cl_masked', 'cl_template', 'cl_template_binned', 'tot']:
                        if key1 not in dl.data[key0][key4]:
                            dl.data[key0][key4][key1] = dict()
                        for key2 in dl.ec.freqs + ['comb']:
                            if key2 not in dl.data[key0][key4][key1]:
                                dl.data[key0][key4][key1][key2] = dict()
                            for key6 in ['TEB', 'IQU', 'EB', 'QU', 'EB_bp', 'QU_bp', 'T', 'E', 'B', 'Q', 'U', 'E_bp', 'B_bp']:
                                if key6 not in dl.data[key0][key4][key1][key2]:
                                    dl.data[key0][key4][key1][key2][key6] = np.array([], dtype=np.complex128)

# dl.data[component]['nlevel']['fs']['cl_template'][freq]['EB']
            
            dl.prediction = dict()
            for key0 in ['N0', 'N1', 'cl_del']:
                if key0 not in dl.prediction:
                    dl.prediction[key0] = dict()
                for key4 in dl.mask_ids + ['fs']:
                    if key4 not in dl.prediction[key0]:
                        dl.prediction[key0][key4] = dict()
                    for key1 in ['QE', "MAP"]:
                        if key1 not in dl.prediction[key0][key4]:
                            dl.prediction[key0][key4][key1] = dict()
                        for key2 in ['N', 'N_eff']:
                            if key2 not in dl.prediction[key0][key4][key1]:
                                dl.prediction[key0][key4][key1][key2] = np.array([], dtype=np.complex128)

            dl.data['weight'] = np.zeros(shape=(2,*(np.loadtxt(dl.ic.weights_fns.format(dl.fg, 'E')).shape)))
            for i, flavour in enumerate(['E', 'B']):
                dl.data['weight'][int(i%len(['E', 'B']))] = np.loadtxt(dl.ic.weights_fns.format(dl.fg, flavour))


        def _process_Madel(dl, ma):
            dl.data_from_CFS = True
            dl.its = [0] if ma.iterations == [] else ma.iterations
            dl.edges = []
            dl.edges_id = []
            if ma.edges != -1:
                dl.edges.append(lc.SPDP_edges)
                dl.edges_id.append('SPDP')
            dl.edges = np.array(dl.edges)
            dl.edges_center = np.array([(e[1:]+e[:-1])/2 for e in dl.edges])

            dl.ll = np.arange(0,dl.edges[0][-1]+1,1)
            dl.scale_ps = dl.ll*(dl.ll+1)/(2*np.pi)
            dl.scale_ps_binned = np.take(dl.scale_ps, [int(a) for a in dl.edges_center[0]])

            if cf.noisemodel.rhits_normalised is not None:
                _mask_path = cf.noisemodel.rhits_normalised[0]
                dl.base_mask = np.nan_to_num(hp.read_map(_mask_path))
            else:
                dl.base_mask = np.ones(shape=hp.nside2npix(cf.data.nside))
            noisemodel_rhits_map = df.get_nlev_mask(np.inf, dl.base_mask)
            noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf

            if ma.masks != None:
                dl.masks = dict({ma.masks[0]:{}})
                dl.binmasks = dict({ma.masks[0]:{}})
                dl.mask_ids = ma.masks[1]
                if ma.masks[0] == 'nlevels': 
                    for mask_id in dl.mask_ids:
                        buffer = df.get_nlev_mask(mask_id, noisemodel_rhits_map)
                        dl.masks[ma.masks[0]].update({mask_id:buffer})
                        dl.binmasks[ma.masks[0]].update({mask_id: np.where(dl.masks[ma.masks[0]][mask_id]>0,1,0)})
                elif ma.masks[0] == 'masks':
                    dl.mask_ids = np.zeros(shape=len(ma.masks[1]))
                    for fni, fn in enumerate(ma.masks[1]):
                        if fn == None:
                            buffer = np.ones(shape=hp.nside2npix(dl._sims.nside))
                            dl.mask_ids[fni] = 1.00
                        elif fn.endswith('.fits'):
                            buffer = hp.read_map(fn)
                        else:
                            buffer = np.load(fn)
                        _fsky = float("{:0.3f}".format(np.sum(buffer)/len(buffer)))
                        dl.mask_ids[fni] = _fsky
                        dl.masks[ma.masks[0]].update({_fsky:buffer})
                        dl.binmasks[ma.masks[0]].update({_fsky: np.where(dl.masks[ma.masks[0]][_fsky]>0,1,0)})
            else:
                dl.masks = {"no":{1.00:np.ones(shape=hp.nside2npix(dl._sims.nside))}}
                dl.mask_ids = np.array([1.00])


        def _process_Config(dl, co):
            if co.outdir_plot_rel:
                dl.outdir_plot_rel = co.outdir_plot_rel
            else:
                dl.outdir_plot_rel = '{}/{}'.format(cf.data.module_.split('.')[2],cf.data.module_.split('.')[-1])
                    
            if co.outdir_plot_root:
                dl.outdir_plot_root = co.outdir_plot_root
            else:
                dl.outdir_plot_root = opj(os.environ['HOME'],'plots')
            
            dl.outdir_plot_abs = opj(dl.outdir_plot_root, dl.outdir_plot_rel)
            if not os.path.isdir(dl.outdir_plot_abs):
                os.makedirs(dl.outdir_plot_abs)
            log.info('Plots will be stored at {}'.format(dl.outdir_plot_abs))

        
        def _process_Data(dl, da):
            dl.simidxs = da.simidxs

            if 'fg' in da.class_parameters:
                dl.fg = da.class_parameters['fg']
            dl._package = da.package_
            dl._module = da.module_
            dl._class = da.class_
            dl.class_parameters = da.class_parameters
            _sims_full_name = '{}.{}'.format(dl._package, dl._module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl._sims = getattr(_sims_module, dl._class)(**dl.class_parameters)
            dl.sims = dl._sims.sims

            if 'experiment_config' in _sims_module.__dict__:
                dl.ec = getattr(_sims_module, 'experiment_config')()
            if 'ILC_config' in _sims_module.__dict__:
                dl.ic = getattr(_sims_module, 'ILC_config')()
            if 'foreground' in _sims_module.__dict__:
                dl.fc = getattr(_sims_module, 'foreground')(dl.fg)
            dl._sims.nside = dl._sims.nside

            dl.beam = dl._sims.beam
            dl.lmax_transf = dl._sims.lmax_transf
            dl.transf = hp.gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf)
            

        dl = DLENSALOT_Concept()
  
        _process_Data(dl, cf.data)
        _process_Madel(dl, cf.madel)
        _process_X(dl)
        _process_Config(dl, cf.config)

        return dl
            

class l2ji_Transformer:
    """Extracts parameters needed for the interactive D.Lensalot job
    """
    @check_MPI
    def build(self, cf):
        
        def _process_Jobs(jobs):
            jobs.append({"interactive":((cf, l2i_Transformer()), lenscarf_handler.Notebook_interactor)})

        jobs = []
        _process_Jobs(jobs)

        return jobs      
        

class l2j_Transformer:
    """Extracts parameters needed for the specific D.Lensalot jobs
    """
    @check_MPI
    def build(self, cf):
        
        # TODO if the pf.X objects were distinguishable by X2X_Transformer, could replace the seemingly redundant checks here.
        def _process_Jobs(jobs, jb):
            if "build_OBD" in jb.jobs:
                jobs.append({"build_OBD":((cf, l2OBD_Transformer()), lenscarf_handler.OBD_builder)})
            if "QE_lensrec" in jb.jobs:
                jobs.append({"QE_lensrec":((cf, l2lensrec_Transformer()), lenscarf_handler.QE_lr)})
            if "MAP_lensrec" in jb.jobs:
                jobs.append({"MAP_lensrec":((cf, l2lensrec_Transformer()), lenscarf_handler.MAP_lr)})
            if "map_delensing" in jb.jobs:
                jobs.append({"map_delensing":((cf, l2d_Transformer()), lenscarf_handler.Map_delenser)})

        jobs = []
        _process_Jobs(jobs, cf.job)
        return jobs


@transform.case(DLENSALOT_Model_mm, l2i_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2ji_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Concept, l2T_Transformer)
def f2b(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_delsuffix(expr)

@transform.case(DLENSALOT_Model_mm, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_mm, l2lensrec_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.mapper(expr)