#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build dlensalot model from configuation file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
from os.path import join as opj
import importlib


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import numpy as np
import healpy as hp
import hashlib

import plancklens
from lenscarf.core import mpi
from plancklens import qest, qecl, utils
from plancklens.sims import planck2018_sims
from plancklens.filt import filt_util, filt_cinv, filt_simple
from plancklens.qcinv import cd_solve

from lenscarf import utils_scarf, utils_sims, remapping
from lenscarf.utils import cli, read_map
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt import utils_cinv_p as cinv_p_OBD
import lenscarf.core.handler as lenscarf_handler

from lenscarf.opfilt import opfilt_ee_wl
from lenscarf.opfilt import opfilt_iso_ee_wl
from lenscarf.opfilt.bmodes_ninv import template_dense

from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Model, DLENSALOT_Concept
from lenscarf.lerepi.config.config_helper import data_functions as df, LEREPI_Constants as lc

class l2T_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    # @log_on_start(logging.INFO, "build() started")
    # @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        _suffix = cf.data.class_
        if 'fg' in cf.data.class_parameters:
            _suffix +='_%s'%(cf.data.class_parameters['fg'])
        if cf.noisemodel.lowell_treat == 'OBD':
            _suffix += '_OBD'
        elif cf.noisemodel.lowell_treat == 'trunc':
            _suffix += '_OBDtrunc'+str(cf.noisemodel.lmin_blm)
        elif cf.noisemodel.lowell_treat == 'None' or cf.noisemodel.typ == None:
            _suffix += '_noOBD'

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


    # @log_on_start(logging.INFO, "build_OBD() started")
    # @log_on_end(logging.INFO, "build_OBD() finished")
    def build_OBD(self, TEMP):

        return os.path.join(TEMP, 'OBD_matrix')


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
        if cf.meta.version == '0.9':
            return self.build(cf)

  
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):


        @log_on_start(logging.INFO, "_process_Meta() started")
        @log_on_end(logging.INFO, "_process_Meta() finished")
        def _process_Meta(dl, me):
            dl.dversion = me.version


        @log_on_start(logging.INFO, "_process_Computing() started")
        @log_on_end(logging.INFO, "_process_Computing() finished")
        def _process_Computing(dl, co):
             dl.tr = int(os.environ.get('OMP_NUM_THREADS', co.OMP_NUM_THREADS))


        @log_on_start(logging.INFO, "_process_Analysis() started")
        @log_on_end(logging.INFO, "_process_Analysis() finished")
        def _process_Analysis(dl, an):
            # key -> k
            dl.key = an.key


            # version -> version
            dl.version = an.version


            # TEMP_suffix -> TEMP_suffix
            dl.TEMP_suffix = an.TEMP_suffix
            dl.TEMP = transform(cf, l2T_Transformer())


            # LENSRES
            dl.lens_res = an.lens_res


            # zbounds -> zbounds
            if an.zbounds[0] == 'nmr_relative':
                dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), an.zbounds[1])
            elif an.zbounds[0] == 'mr_relative':
                _zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.mask), np.inf)
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


        @log_on_start(logging.INFO, "_process_Data() started")
        @log_on_end(logging.INFO, "_process_Data() finished")
        def _process_Data(dl, da):
            # package_
            _package = da.package_


            # module_
            _module = da.module_


            # class_
            _class = da.class_


            # class_parameters -> sims
            _dataclass_parameters = da.class_parameters
            if 'fg' in _dataclass_parameters:
                dl.fg = _dataclass_parameters['fg']
            _sims_full_name = '{}.{}'.format(_package, _module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, _class)(**_dataclass_parameters)


            # data_type
            dl.data_type = da.data_type


            # data_field
            dl.data_field = da.data_field


            # beam
            dl.beam = da.beam


            # nside
            dl.nside = da.nside


            # transferfunction
            dl.transferfunction = da.transferfunction
            data_lmax = da.lmax
            if dl.transferfunction == 'gauss':
                _cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
                dl.cls_unl = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
                dl.cls_len = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lensedCls.dat'))
                dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
                dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)

                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_tlm)
                dl.transf_elm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_elm)
                dl.transf_blm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl = cli(dl.cls_len['tt'][:data_lmax + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:data_lmax + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)
            elif dl.transferfunction == 'gauss_with_pixwin':
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * hp.pixwin(2048, lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_tlm)
                dl.transf_elm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * hp.pixwin(2048, lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_elm)
                dl.transf_blm = gauss_beam(df.a2r(cf.data.beam), lmax=data_lmax) * hp.pixwin(2048, lmax=data_lmax) * (np.arange(data_lmax + 1) >= cf.noisemodel.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl = cli(dl.cls_len['tt'][:data_lmax + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:data_lmax + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:data_lmax + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)


        @log_on_start(logging.INFO, "_process_Noisemodel() started")
        @log_on_end(logging.INFO, "_process_Noisemodel() finished")
        def _process_Noisemodel(dl, nm):
            # lmin_tlm
            dl.lmin_tlm = nm.lmin_tlm
                
                
            # lmin_elm
            dl.lmin_elm = nm.lmin_elm


            # lmin_blm
            dl.lmin_blm = nm.lmin_blm


            # lowell_treat
            dl.lowell_treat = nm.lowell_treat
            if dl.lowell_treat == 'OBD':
                dl.obd_libdir = nm.OBD.libdir
                dl.obd_rescale = nm.rescale
                dl.obd_tpl = nm.OBD.tpl
                dl.obd_nlevdep = nm.OBD.nlevdep
                dl.tpl = template_dense(nm.lmin_blm, dl.ninvjob_geometry, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)
            elif dl.lowell_treat == 'trunc':
                dl.tpl = None
                dl.tpl_kwargs = dict()
                dl.lmin_tlm = nm.lmin_tlm
                dl.lmin_elm = nm.lmin_elm
                dl.lmin_blm = nm.lmin_blm
            elif dl.lowell_treat == None or dl.lowell_treat == 'None':
                dl.tpl = None
                dl.tpl_kwargs = dict()
                # TODO are 0s a good value? 
                dl.lmin_tlm = dl.Lmin
                dl.lmin_elm = dl.Lmin
                dl.lmin_blm = dl.Lmin


            # nlev_t
            dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)


            # nlev_p
            dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)


            # rhits_normalised
            dl.rhits_normalised = nm.rhits_normalised


            # mask
            dl.masks = l2OBD_Transformer.get_masks(cf)


            # ninvjob_geometry
            if nm.ninvjob_geometry == 'healpix_geometry':
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)
      

        @log_on_start(logging.INFO, "_process_Qerec() started")
        @log_on_end(logging.INFO, "_process_Qerec() finished")
        def _process_Qerec(dl, qe):
            # simidxs
            dl.QE_simidxs = qe.simidxs
            dl.QE_simidxs_mf = qe.simidxs_mf
            dl.Nmf = 0 if cf.analysis.version == 'noMF' else len(dl.QE_simidxs_mf)


            # cg_tol
            dl.cg_tol = qe.cg_tol


            # lmax
            dl.qe_filter_lmax = qe.filter.lmax
            dl.qe_filter_mmax = qe.filter.mmax
            dl.lmax_qlm = qe.lmax_qlm


            # chain
            dl.chain_model = qe.chain
            if dl.chain_model.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if dl.chain_model.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]


            # filter
            dl.qe_filter_directional = qe.filter.directional
            dl.qe_filter_data_type = qe.filter.data_type
            if dl.qe_filter_directional == 'aniso':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                # TODO filters can be initialised with both, ninvX_desc and ninv_X. But Plancklens' hashcheck will complain if it changed since shapes are different. Not sure which one I want to use in the future..
                # TODO using ninv_X possibly causes hashcheck to fail, as v1 == v2 won't work on arrays.
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), qe.lmax_qlm, dl.nside, dl.cls_len, dl.transf_tlm, dl.ninvt_desc,
                    marge_monopole=True, marge_dipole=True, marge_maps=[])
                if dl.lowell_treat == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=qe.lmax_qlm)
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), lmax_qlm, dl.nside, dl.cls_len, transf_elm_loc[:qe.lmax_qlm+1], dl.ninvp_desc, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(qe.lmax_qlm, dl.cg_tol), bmarg_lmax=dl.lmin_blm, zbounds=dl.zbounds, _bmarg_lib_dir=dl.obd_libdir, _bmarg_rescal=dl.obd_rescale, sht_threads=dl.tr)
                elif dl.lowell_treat == 'trunc' or dl.lowell_treat == None or dl.lowell_treat == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), qe.lmax_qlm, dl.nside, dl.cls_len, dl.transf_elm, dl.ninvp_desc,
                        chain_descr=dl.chain_descr(qe.lmax_qlm, dl.cg_tol), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())

                _filter_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                _ftl_rs = np.ones(qe.lmax_qlm + 1, dtype=float) * (np.arange(qe.lmax_qlm + 1) >= dl.lmin_tlm)
                _fel_rs = np.ones(qe.lmax_qlm + 1, dtype=float) * (np.arange(qe.lmax_qlm + 1) >= dl.lmin_elm)
                _fbl_rs = np.ones(qe.lmax_qlm + 1, dtype=float) * (np.arange(qe.lmax_qlm + 1) >= dl.lmin_blm)
                dl.filter = filt_util.library_ftl(_filter_raw, qe.lmax_qlm, _ftl_rs, _fel_rs, _fbl_rs)
            elif dl.qe_filter_directional == 'iso':
                dl.filter = filt_simple.library_fullsky_alms_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, {'t':dl.transf_tlm, 'e':dl.transf_elm, 'b':dl.transf_blm}, dl.cls_len, dl.ftl, dl.fel, dl.fbl, cache=True)


            # qlms
            dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.filter, dl.filter, dl.cls_len['te'], dl.nside, lmax_qlm=qe.lmax_qlm)


            # ninvjob_qe_geometry
            if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)


            # qe_cl_analysis
            dl.cl_analysis = qe.cl_analysis
            if qe.cl_analysis == True:
                # TODO fix numbers for mc correction and total nsims
                dl.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                        np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
                dl.ds_dict = { k : -1 for k in range(300)}

                dl.ivfs_d = filt_util.library_shuffle(dl.ivfs, dl.ds_dict)
                dl.ivfs_s = filt_util.library_shuffle(dl.ivfs, dl.ss_dict)

                dl.qlms_ds = qest.library_sepTP(opj(dl.TEMP, 'qlms_ds'), dl.ivfs, dl.ivfs_d, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)
                dl.qlms_ss = qest.library_sepTP(opj(dl.TEMP, 'qlms_ss'), dl.ivfs, dl.ivfs_s, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)

                dl.mc_sims_bias = np.arange(60, dtype=int)
                dl.mc_sims_var  = np.arange(60, 300, dtype=int)

                dl.qcls_ds = qecl.library(opj(dl.TEMP, 'qcls_ds'), dl.qlms_ds, dl.qlms_ds, np.array([]))  # for QE RDN0 calculations
                dl.qcls_ss = qecl.library(opj(dl.TEMP, 'qcls_ss'), dl.qlms_ss, dl.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
                dl.qcls_dd = qecl.library(opj(dl.TEMP, 'qcls_dd'), dl.qlms_dd, dl.qlms_dd, dl.mc_sims_bias)


            # Lmin -> cpp
            dl.Lmin = qe.Lmin
            dl.cpp = np.copy(dl.cls_unl['pp'][:qe.lmax_qlm + 1])
            dl.cpp[:dl.Lmin] *= 0.


        @log_on_start(logging.INFO, "_process_Itrec() started")
        @log_on_end(logging.INFO, "_process_Itrec() finished")
        def _process_Itrec(dl, it):
            # tasks
            dl.tasks = it.tasks
            ## tasks -> mf_dirname
            if "calc_meanfield" in dl.tasks:
                if dl.version == '' or dl.version == None:
                    dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'Nmf': dl.Nmf}))
                else:
                    dl.mf_dirname = opj(dl.TEMP, l2T_Transformer.ofj('mf', {'version': dl.version, 'Nmf': dl.Nmf}))
                if not os.path.isdir(dl.mf_dirname) and mpi.rank == 0:
                    os.makedirs(dl.mf_dirname)


            # cg_tol
            dl.cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1


            # simidxs
            dl.it_simidxs = it.simidxs
            dl.it_simidxs_mf = it.simidxs_mf


            # sims -> sims_MAP
            if it.filter.directional == 'aniso':
                dl.sims_MAP = utils_sims.ztrunc_sims(dl.sims, dl.nside, [dl.zbounds])
            elif it.filter.directional == 'iso':
                dl.sims_MAP = dl.sims


            # itmax
            dl.itmax = it.itmax


            # iterator_typ
            dl.iterator_typ = it.iterator_typ


            # lmax
            dl.it_filter_lmax = it.filter.lmax
            dl.it_filter_mmax = it.filter.mmax
            dl.it_lmax_unl, dl.it_mmax_unl = it.filter.lmax_unl, it.filter.mmax_unl
            dl.it_lmax_len, dl.it_mmax_len = it.filter.lmax_len, it.filter.mmax_len


            # lenjob_geometry
            if it.lenjob_geometry == 'thin_gauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(dl.it_lmax_unl, 2, zbounds=dl.zbounds_len)


            # lenjob_pbgeometry
            if it.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent))


            # filter
            dl.filter_directional = it.filter.directional
            dl.filter_data_type = it.filter.data_type
            wee = dl.key == 'p_p'
            dl.ffi = remapping.deflection(dl.lenjob_pbgeometry, dl.lens_res, np.zeros(shape=hp.Alm.getsize(it.lmax_plm)), it.mmax_plm, dl.tr, dl.tr)
            if dl.filter_directional == 'iso':
                dl.filter = opfilt_iso_ee_wl.alm_filter_nlev_wl(dl.nlev_p, dl.ffi, dl.transf_elm, (dl.it_lmax_unl, it.it_mmax_unl), (dl.it_lmax_len, dl.it_mmax_len), wee=wee, transf_b=dl.transf_blm, nlev_b=dl.nlev_p)
                dl.k_geom = filter.ffi.geom
            elif dl.filter_directional == 'aniso':
                ninv = [dl.sims_MAP.ztruncify(read_map(ni)) for ni in dl.ninvp_desc]
                dl.filter = opfilt_ee_wl.alm_filter_ninv_wl(
                    dl.ninvjob_geometry, ninv, dl.ffi, dl.transf_elm,
                    (dl.it_lmax_unl, dl.it_mmax_unl), (dl.it_lmax_len, dl.it_mmax_len),
                    dl.tr, dl.tpl, wee=wee, lmin_dotop=min(dl.lmin_elm, dl.lmin_blm), transf_blm=dl.transf_blm)
                dl.k_geom = dl.filter.ffi.geom


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
                dl.stepper = steps.harmonicbump(it.lmax_plm, it.mmax_plm, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
            

        dl = DLENSALOT_Concept()    
        _process_Meta(dl, cf.meta)
        _process_Computing(dl, cf.computing)
        _process_Analysis(dl, cf.analysis)
        _process_Data(dl, cf.data)
        _process_Noisemodel(dl, cf.noisemodel)
        _process_Qerec(dl, cf.qerec)
        _process_Itrec(dl, cf.itrec)


        if mpi.rank == 0:
            log.info("I am going to work with the following values:")
            _str = '---------------------------------------------------\n'
            for key, val in dl.__dict__.items():
                _str += '{}:\t{}'.format(key, val)
                _str += '\n'
            _str += '---------------------------------------------------\n'
            log.info(_str)

        return dl


class l2OBD_Transformer:
    """Extracts all parameters needed for building consistent OBD
    """

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        @log_on_start(logging.INFO, "() started")
        @log_on_end(logging.INFO, "_process_builOBDparams() finished")
        def _process_Noisemodel(dl, nm):
            _TEMP = transform(cf, l2T_Transformer())
            dl.TEMP = transform(_TEMP, l2T_Transformer())
            if os.path.isfile(opj(nm.BMARG_LIBDIR,'tniti.npy')):
                # TODO need to test if it is the right tniti.npy
                log.warning("tniti.npy in destination dir {} already exists.".format(nm.BMARG_LIBDIR))
            if os.path.isfile(opj(dl.TEMP,'tniti.npy')):
                # TODO need to test if it is the right tniti.npy
                log.warning("tniti.npy in buildpath dir {} already exists.".format(dl.TEMP))
                log.warning("Exiting. Please check your settings.")
                sys.exit()
            else:
                dl.BMARG_LCUT = nm.BMARG_LCUT
                dl.nside = cf.data.nside
                dl.nlev_dep = nm.nlev_dep
                dl.geom = utils_scarf.Geom.get_healpix_geometry(dl.nside)
                dl.masks, dl.rhits_map = l2OBD_Transformer.get_masks(cf)
                dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                dl.ninv_p_desc = l2OBD_Transformer.get_ninvp(cf)


        dl = DLENSALOT_Concept()
        _process_Noisemodel(dl, cf.noisemodel)

        return dl


    # @log_on_start(logging.INFO, "get_nlrh_map() started")
    # @log_on_end(logging.INFO, "get_nlrh_map() finished")
    def get_nlrh_map(cf):
        noisemodel_rhits_map = df.get_nlev_mask(cf.noisemodel.rhits_normalised[1], hp.read_map(cf.noisemodel.rhits_normalised[0]))
        noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf

        return noisemodel_rhits_map


    # @log_on_start(logging.INFO, "get_nlevt() started")
    # @log_on_end(logging.INFO, "get_nlevt() finished")
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


    # @log_on_start(logging.INFO, "get_nlevp() started")
    # @log_on_end(logging.INFO, "get_nlevp() finished")
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


    @log_on_start(logging.INFO, "get_ninvt() started")
    @log_on_end(logging.INFO, "get_ninvt() finished")
    def get_ninvt(cf):
        nlev_t = l2OBD_Transformer.get_nlevt(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        ninv_desc = [np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_t ** 2])/noisemodel_norm] + masks

        return ninv_desc


    @log_on_start(logging.INFO, "get_ninvp() started")
    @log_on_end(logging.INFO, "get_ninvp() finished")
    def get_ninvp(cf):
        nlev_p = l2OBD_Transformer.get_nlevp(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        ninv_desc = [[np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_p ** 2])/noisemodel_norm] + masks]

        return ninv_desc


    # @log_on_start(logging.INFO, "get_masks() started")
    # @log_on_end(logging.INFO, "get_masks() finished")
    def get_masks(cf):
        # TODO refactor
        masks = []
        if cf.noisemodel.rhits_normalised is not None:
            msk = l2OBD_Transformer.get_nlrh_map(cf)
        else:
            msk = np.ones(shape=hp.nside2npix(cf.data.nside))
        masks.append(msk)
        if cf.noisemodel.mask is not None:
            if type(cf.noisemodel.mask) == str:
                _mask = cf.noisemodel.mask
            elif cf.noisemodel.mask[0] == 'nlev':
                noisemodel_rhits_map = msk.copy()
                _mask = df.get_nlev_mask(cf.noisemodel.mask[1], noisemodel_rhits_map)
                _mask = np.where(_mask>0., 1., 0.)
        else:
            _mask = np.ones(shape=hp.nside2npix(cf.data.nside))
        masks.append(_mask)

        return masks, msk


class l2d_Transformer:
    """Directory is built upon runtime, so accessing it here
    """

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build()) finished")
    def build(self, cf):
        def _process_Madel(dl, ma):
            dl.data_from_CFS = ma.data_from_CFS
            dl.k = cf.analysis.K
            dl.version = cf.analysis.V

            dl.imin = cf.data.IMIN
            dl.imax = cf.data.IMAX
            dl.simidxs = cf.data.simidxs if cf.data.simidxs != [] else np.arange(dl.imin, dl.imax+1)
            dl.its = [0] if ma.iterations == [] else ma.iterations

            dl.Nmf = len(cf.analysis.simidxs_mf)
            if 'fg' in cf.data.class_parameters:
                dl.fg = cf.data.class_parameters['fg']
            dl._package = cf.data.package_
            dl._module = cf.data.module_
            dl._class = cf.data.class_
            dl.class_parameters = cf.data.class_parameters
            _sims_full_name = '{}.{}'.format(dl._package, dl._module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, dl._class)(**dl.class_parameters)

            dl.ec = getattr(_sims_module, 'experiment_config')()
            dl.nside = cf.data.nside

            dl.data_type = cf.data.data_type
            dl.data_field = cf.data.data_field

            dl.TEMP = transform(cf, l2T_Transformer())

            # TODO II
            # could put btempl paths similar to sim path handling. If D.lensalot handles it, use D.lensalot internal class for it
            # dl.libdir_iterators = lambda qe_key, simidx, version: de.libdir_it%()
            # if it==12:
            #     rootstr = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/lt_recons/')
            #     if self.fg == '00':
            #         return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            #     elif self.fg == '07':
            #         return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            #     elif self.fg == '09':
            #         return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            # elif it==0:
            #     return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(self.fg, simidx, simidx)    
          
            if ma.libdir_it is None:
                dl.libdir_iterators = lambda qe_key, simidx, version: opj(dl.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
                def fn_btempl_intern(simidx, fg, it):
                    if it == 0:
                        return self.libdir_iterators(self.key, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%03d.npy'%(it, it, self.Nmf)
                    if self.dlm_mod_bool:
                        return self.libdir_iterators(self.key, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod%03d.npy'%(it, it, self.Nmf)
                    else:
                        return self.libdir_iterators(self.key, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%03d.npy'%(it, it, self.Nmf)
                dl.getfn_btemplm = fn_btempl_intern
            elif ma.dir_btempl == '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/lt_recons/':
                # if btemplates come from cdirs, need to use 'external' dirstructure.
                if cf.da.module_.splot('.').endswith('08b'):
                    # 08b structure is especially difficult
                    def fn_btempl_08b(simidx, fg, it):
                        fn = 'blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                        if it==0:
                            return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(fg, simidx, simidx)    
                        elif it==12:
                            if fg == '00':
                                return opj(ma.dir_btempl, '08b.%02d_sebibel_210708_ilc_iter', fn)
                            elif fg in ['07', '09']:
                                return opj(ma.dir_btempl, '08b.%02d_sebibel_210910_ilc_iter', fn)
                    dl.getfn_btemplm = fn_btempl_08b
                elif cf.da.module_.splot('.').endswith('08d'):
                    # 
                    assert 0, 'Implement if needed'
            
            dl.analysis_path = dl.TEMP.split('/')[-1]
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
                            buffer = np.ones(shape=hp.nside2npix(dl.nside))
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
                dl.masks = {"no":{1.00:np.ones(shape=hp.nside2npix(dl.nside))}}
                dl.mask_ids = np.array([1.00])

            dl.beam = cf.data.beam
            dl.lmax_transf = cf.data.lmax_transf
            if cf.analysis.STANDARD_TRANSFERFUNCTION == True:
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf)
            elif cf.analysis.STANDARD_TRANSFERFUNCTION == 'with_pixwin':
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf) * hp.pixwin(cf.data.nside, lmax=dl.lmax_transf)
            else:
                log.info("Don't understand your STANDARD_TRANSFERFUNCTION: {}".format(cf.analysis.STANDARD_TRANSFERFUNCTION))
            
            if ma.Cl_fid == 'ffp10':
                dl.cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
                dl.cls_len = utils.camb_clfile(opj(dl.cls_path, 'FFP10_wdipole_lensedCls.dat'))
                dl.clg_templ = dl.cls_len['ee']
                dl.clc_templ = dl.cls_len['bb']
                dl.clg_templ[0] = 1e-32
                dl.clg_templ[1] = 1e-32
            pert_mod_string = ''
            dl.btemplate_perturbative_lensremap = ma.btemplate_perturbative_lensremap
            if dl.btemplate_perturbative_lensremap == True:
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
                    dl.sha_edges[n].update((str(dl.edges[n]) + pert_mod_string).encode())
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
                dl.sha_edges[0].update(('unbinned'+pert_mod_string).encode())
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
            dl.dlm_mod_bool = ma.dlm_mod
            if dl.binning == 'binned':
                if dl.dlm_mod_bool:
                    dl.file_op = lambda simidx, fg, edges_idx: opj(
                        dl.TEMP_DELENSED_SPECTRUM, '{}'.format(dl.dirid[edges_idx]), l2T_Transformer.ofj('ClBB', {'simidx': simidx, 'dlmmod': '', 'fg': fg}))
                else:
                    dl.file_op = lambda simidx, fg, edges_idx: opj(
                        dl.TEMP_DELENSED_SPECTRUM, '{}'.format(dl.dirid[edges_idx]), l2T_Transformer.ofj('ClBB', {'simidx': simidx, 'fg': fg}))
            else:
                if dl.dlm_mod_bool:
                    dl.file_op = lambda simidx, fg, x: opj(
                        dl.TEMP_DELENSED_SPECTRUM, dl.dirid[0], l2T_Transformer.ofj('ClBB', {'simidx': simidx, 'dlmmod': '', 'fg': fg}))
                else:
                    dl.file_op = lambda simidx, fg, x: opj(
                        dl.TEMP_DELENSED_SPECTRUM, dl.dirid[0], l2T_Transformer.ofj('ClBB', {'simidx': simidx, 'fg': fg}))

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

        dl.btemplate_perturbative_lensremap = cf.itrec.btemplate_perturbative_lensremap

        _process_Madel(dl, cf.madel)
        _process_Config(dl, cf.config)
        _check_powspeccalculator(dl.cl_calc)


        return dl


class l2j_Transformer:
    """Extracts parameters needed for the specific D.Lensalot jobs
    """
    def build(self, cf):
        
        # TODO if the pf.X objects were distinguishable by X2X_Transformer, could replace the seemingly redundant checks here.
        def _process_Jobs(jobs, jb):
            if jb.build_OBD:
                jobs.append({"build_OBD":((cf, l2OBD_Transformer()), lenscarf_handler.OBD_builder)})
            if jb.QE_lensrec:
                jobs.append({"QE_lensrec":((cf, l2lensrec_Transformer()), lenscarf_handler.QE_lr)})
            if jb.MAP_lensrec:
                jobs.append({"MAP_lensrec":((cf, l2lensrec_Transformer()), lenscarf_handler.MAP_lr)})
            if jb.map_delensing:
                jobs.append({"map_delensing":((cf, l2d_Transformer()), lenscarf_handler.Map_delenser)})
            if jb.inspect_result:
                assert 0, "Implement if needed"

        jobs = []
        _process_Jobs(jobs, cf.job)

        return jobs


@transform.case(DLENSALOT_Concept, l2T_Transformer)
def f2b(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_delsuffix(expr)

@transform.case(str, l2T_Transformer)
def f2c(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_OBD(expr)

@transform.case(DLENSALOT_Model, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2lensrec_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.mapper(expr)
