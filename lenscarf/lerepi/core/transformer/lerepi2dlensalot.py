#!/usr/bin/env python

"""lerepi2dlensalot.py: transformer module to build dlensalot model from configuation file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
from os.path import join as opj
import importlib
import traceback


import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import numpy as np
import healpy as hp
import hashlib

import plancklens
from lenscarf.core import mpi
from plancklens import qest, qecl, utils
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

from lenscarf.lerepi.config.config_helper import data_functions as df, LEREPI_Constants as lc
from lenscarf.lerepi.core.metamodel.dlensalot_v2 import DLENSALOT_Model as DLENSALOT_Model_v2, DLENSALOT_Concept
from lenscarf.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model as DLENSALOT_Model_mm


class l2T_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """


    # @log_on_start(logging.INFO, "build_v2() started")
    # @log_on_end(logging.INFO, "build_v2() finished")
    def build_v2(self, cf):
        _suffix = cf.data.class_
        if 'fg' in cf.data.class_parameters:
            _suffix +='_%s'%(cf.data.class_parameters['fg'])
        if cf.noisemodel.typ == 'OBD':
            _suffix += '_OBD'
        elif cf.noisemodel.typ == 'trunc':
            _suffix += '_OBDtrunc'+str(cf.noisemodel.lmin_blm)
        elif cf.noisemodel.typ == 'None' or cf.noisemodel.typ == None:
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
        if cf.meta.version == '0.2a':
            return self.build(cf)

        elif cf.meta.version == '0.2b':
            return self.build_v2(cf)
        
        elif cf.meta.version == '0.9':
            return self.build_v3(cf)


    @log_on_start(logging.INFO, "build_v2() started")
    @log_on_end(logging.INFO, "build_v2() finished")
    def build_v2(self, cf):

        @log_on_start(logging.INFO, "_process_Analysis() started")
        @log_on_end(logging.INFO, "_process_Analysis() finished")
        def _process_Analysis(dl, an):
            dl.temp_suffix = an.TEMP_suffix
            dl.TEMP = transform(cf, l2T_Transformer())
            # TODO unclear what this actually does
            if cf.qerec.overwrite_libdir != '' and cf.qerec.overwrite_libdir != -1 and cf.qerec.overwrite_libdir != None:
                dl.TEMP = cf.qerec.overwrite_libdir
                dl.overwrite_libdir = cf.qerec.overwrite_libdir
            else:
                dl.overwrite_libdir = None
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', cf.job.OMP_NUM_THREADS))
            dl.version = an.V
            dl.k = an.K
            dl.itmax = an.ITMAX
            dl.simidxs_mf = cf.analysis.simidxs_mf
            dl.Nmf = 0 if cf.analysis.V == 'noMF' else len(dl.simidxs_mf)
            if an.zbounds[0] == 'nmr_relative':
                dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), an.zbounds[1])
            elif type(an.zbounds[0]) in [float, int, np.float64]:
                dl.zbounds = an.zbounds
            else:
                log.error('Not sure what to do with this zbounds: {}'.format(an.zbounds))
                traceback.print_stack()
                sys.exit()

            if an.zbounds_len[0] == 'extend':
                dl.zbounds_len = df.extend_zbounds(dl.zbounds, degrees=an.zbounds_len[1])
            elif an.zbounds_len[0] == 'max':
                dl.zbounds_len = [-1, 1]
            elif type(an.zbounds_len[0]) in [float, int, np.float64]:
                dl.zbounds_len = an.zbounds_len
            else:
                log.error('Not sure what to do with this zbounds_len: {}'.format(an.zbounds_len))
                traceback.print_stack()
                sys.exit()

            dl.pb_ctr, dl.pb_extent = an.pbounds

            dl.lensres = an.LENSRES
            dl.Lmin = an.Lmin

            dl.lmax_filt = an.lmax_filt
            dl.lmax_ivf = an.lmax_ivf
            dl.lmin_ivf = an.lmin_ivf
            dl.mmax_ivf = an.mmax_ivf
            dl.mmin_ivf = an.mmin_ivf

            dl.lmax_unl = an.lmax_unl
            dl.mmax_unl = an.mmax_unl

            dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
            dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)

            _cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
            dl.cls_unl = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            dl.cls_len = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lensedCls.dat'))

            dl.STANDARD_TRANSFERFUNCTION  = an.STANDARD_TRANSFERFUNCTION 
            if dl.STANDARD_TRANSFERFUNCTION == True:
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_tlm)
                dl.transf_elm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_elm)
                dl.transf_blm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl = cli(dl.cls_len['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

            elif dl.STANDARD_TRANSFERFUNCTION == 'with_pixwin':
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * hp.pixwin(2048, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_tlm)
                dl.transf_elm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * hp.pixwin(2048, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_elm)
                dl.transf_blm = gauss_beam(df.a2r(cf.data.beam), lmax=an.lmax_ivf) * hp.pixwin(2048, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl = cli(dl.cls_len['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t[:an.lmax_ivf + 1])**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p[:an.lmax_ivf + 1])**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p[:an.lmax_ivf + 1])**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t[:an.lmax_ivf + 1])**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p[:an.lmax_ivf + 1])**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p[:an.lmax_ivf + 1])**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)
            else:
                log.info("Don't understand your input.")
                sys.exit()

        @log_on_start(logging.INFO, "_process_Data() started")
        @log_on_end(logging.INFO, "_process_Data() finished")
        def _process_Data(dl, da):
            dl.imin = da.IMIN
            dl.imax = da.IMAX

            dl.simidxs = da.simidxs if da.simidxs != [] else np.arange(dl.imin, dl.imax+1)
            _package = da.package_
            if da.package_.startswith('lerepi'):
                _package = 'lenscarf.'+da.package_

            _module = da.module_
            _class = da.class_
            dl.dataclass_parameters = da.class_parameters
            dl.nside = da.nside

            _sims_full_name = '{}.{}'.format(_package, _module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, _class)(**dl.dataclass_parameters)

            if 'fg' in dl.dataclass_parameters:
                dl.fg = dl.dataclass_parameters['fg']

            if da.data_type is None:
                log.info("must specify data_type")
                sys.exit()
            elif da.data_type in ['map', 'alm']:
                dl.data_type = da.data_type
            else:
                log.info("Don't understand your data_type: {}".format(da.data_type))
                sys.exit()

            if da.data_field is None:
                log.info("must specify data_type")
                sys.exit()
            elif da.data_field in ['eb', 'qu']:
                dl.data_field = da.data_field
            else:
                log.info("Don't understand your data_field: {}".format(da.data_field))
                sys.exit()

            dl.beam = da.beam
            dl.lmax_transf = da.lmax_transf
            # dl.transf_data = gauss_beam(df.a2r(cf.data.beam), lmax=dl.lmax_transf)


        @log_on_start(logging.INFO, "_process_Noisemodel() started")
        @log_on_end(logging.INFO, "_process_Noisemodel() finished")
        def _process_Noisemodel(dl, nm):
            if nm.ninvjob_geometry == 'healpix_geometry':
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)
            dl.OBD_type = nm.typ
            dl.BMARG_LIBDIR = nm.BMARG_LIBDIR
            dl.BMARG_LCUT = nm.BMARG_LCUT
            dl.BMARG_RESCALE = nm.BMARG_RESCALE

            if dl.OBD_type == 'OBD':
                if nm.tpl == 'template_dense':
                    def tpl_kwargs(lmax_marg, geom, sht_threads, _lib_dir=None, rescal=1.):
                        return locals()
                    dl.tpl = template_dense
                    dl.tpl_kwargs = tpl_kwargs(nm.BMARG_LCUT, dl.ninvjob_geometry, dl.tr, _lib_dir=dl.BMARG_LIBDIR, rescal=dl.BMARG_RESCALE) 
                else:
                    assert 0, "Implement if needed"
                dl.lmin_tlm = nm.lmin_tlm
                dl.lmin_elm = nm.lmin_elm
                dl.lmin_blm = nm.lmin_blm
            elif dl.OBD_type == 'trunc':
                dl.tpl = None
                dl.tpl_kwargs = dict()
                dl.lmin_tlm = nm.lmin_tlm
                dl.lmin_elm = nm.lmin_elm
                dl.lmin_blm = nm.lmin_blm
            elif dl.OBD_type == None or dl.OBD_type == 'None':
                dl.tpl = None
                dl.tpl_kwargs = dict()
                dl.lmin_tlm = 0
                dl.lmin_elm = 0
                dl.lmin_blm = 0
            else:
                log.error("Don't understand your OBD_type input. Exiting..")
                traceback.print_stack()
                sys.exit()

            dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
            dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
            dl.nlev_dep = nm.nlev_dep
            dl.inf = nm.inf
            dl.masks = l2OBD_Transformer.get_masks(cf)
            dl.rhits_normalised = nm.rhits_normalised
            

        @log_on_start(logging.INFO, "_process_Qerec() started")
        @log_on_end(logging.INFO, "_process_Qerec() finished")
        def _process_Qerec(dl, qe):
            dl.lmax_qlm = qe.lmax_qlm
            dl.mmax_qlm = qe.mmax_qlm
            dl.cg_tol = qe.cg_tol
            dl.qe_tasks = qe.tasks

            dl.chain_model = qe.chain
            if dl.chain_model.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if dl.chain_model.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

            if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)

            if qe.ivfs == 'sepTP':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_tlm, dl.ninvt_desc,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])  
                if dl.OBD_type == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=dl.lmax_ivf)
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, transf_elm_loc[:dl.lmax_ivf+1], dl.ninvp_desc, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(dl.lmax_ivf, dl.cg_tol), bmarg_lmax=dl.BMARG_LCUT, zbounds=dl.zbounds, _bmarg_lib_dir=dl.BMARG_LIBDIR, _bmarg_rescal=dl.BMARG_RESCALE, sht_threads=dl.tr)
                elif dl.OBD_type == 'trunc' or dl.OBD_type == None or dl.OBD_type == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_elm, dl.ninvp_desc,
                        chain_descr=dl.chain_descr(dl.lmax_ivf, dl.cg_tol), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())
                else:
                    log.error("Don't understand your OBD_typ input. Exiting..")
                    traceback.print_stack()
                    sys.exit()
                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)

                dl.ftl_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.fel_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_elm)
                dl.fbl_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_blm)
                dl.ivfs   = filt_util.library_ftl(dl.ivfs_raw, dl.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
            elif qe.ivfs == 'simple':
                dl.ivfs = filt_simple.library_fullsky_alms_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, {'t':dl.transf_tlm, 'e':dl.transf_elm, 'b':dl.transf_blm}, dl.cls_len, dl.ftl, dl.fel, dl.fbl, cache=True)
            else:
                assert 0, 'Implement if needed'
            dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)

            dl.QE_LENSING_CL_ANALYSIS = qe.QE_LENSING_CL_ANALYSIS
            if qe.QE_LENSING_CL_ANALYSIS == True:
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

            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:dl.Lmin] *= 0.


        @log_on_start(logging.INFO, "_process_Itrec() started")
        @log_on_end(logging.INFO, "_process_Itrec() finished")
        def _process_Itrec(dl, it):
            assert it.filter in ['opfilt_ee_wl.alm_filter_ninv_wl', 'opfilt_iso_ee_wl.alm_filter_nlev_wl'] , 'Implement if needed, MAP filter needs to move to l2d'
            dl.filter = it.filter
            dl.ivfs_qe = cf.qerec.ivfs
            dl.btemplate_perturbative_lensremap = it.btemplate_perturbative_lensremap

            # TODO hack. We always want to subtract it atm. But possibly not in the future.
            if "QE_subtract_meanfield" in it.__dict__:
                # dl.subtract_meanfield = iteration.QE_subtract_meanfield
                dl.subtract_meanfield = True
            else:
                dl.subtract_meanfield  = True

            dl.it_tasks = it.tasks
            if it.cg_tol < 1.:
                if 'tol5e5' in cf.analysis.TEMP_suffix:
                    dl.cg_tol = lambda itr : it.cg_tol
                else:
                    dl.cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1
            else:
                if 'rinf_tol4' in cf.analysis.TEMP_suffix:
                    log.warning('tol_iter increased for this run. This is hardcoded.')
                    dl.cg_tol = lambda itr : 2*10 ** (- it.cg_tol) if itr <= 10 else 2*10 ** (-(it.cg_tol+1))
                elif 'tol5e5' in cf.analysis.TEMP_suffix:
                    dl.cg_tol = lambda itr : 1*10 ** (- it.cg_tol) 
                else:
                    dl.cg_tol = lambda itr : 1*10 ** (- it.cg_tol) if itr <= 10 else 1*10 ** (-(it.cg_tol+1))
            dl.soltn_cond = it.soltn_cond

            if it.lenjob_geometry == 'thin_gauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(dl.lmax_unl, 2, zbounds=dl.zbounds_len)
            if it.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent))

            dl.iterator_typ = it.iterator_typ
            if it.mfvar == 'same' or it.mfvar == '':
                dl.mfvar = None
            elif it.mfvar.startswith('/'):
                if os.path.isfile(it.mfvar):
                    dl.mfvar = it.mfvar
                else:
                    log.error('Not sure what to do with this meanfield: {}'.format(it.mfvar))
            if dl.mfvar:
                # TODO this is a terrible way of replacing foreground..
                # the following lines access analysis Y while being an analysis X, 
                # as realization dependent-mf calculation in core/handler needs qlms_dd_mfvar to remove the correct simulation
                __dataclass_parameters = cf.data.class_parameters
                __dataclass_parameters['fg'] = __dataclass_parameters['fg'].replace(dl.fg, dl.version[2:4])
                if 'fg' in __dataclass_parameters:
                    _fg = __dataclass_parameters['fg']
                _package = cf.data.package_
                _module = cf.data.module_
                if cf.data.package_.startswith('lerepi'):
                    _package = 'lenscarf.'+cf.data.package_
                __sims_full_name = '{}.{}'.format(_package, _module)
                __sims_module = importlib.import_module(__sims_full_name)
                _class = cf.data.class_
                _sims = getattr(__sims_module, _class)(**__dataclass_parameters)
                TEMPmfvar = dl.TEMP.replace('_{}_'.format(dl.fg), "_{}_".format(dl.version[2:4]))
                _ivfs_raw = filt_cinv.library_cinv_sepTP(opj(TEMPmfvar, 'ivfs'), _sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                _ivfs = filt_util.library_ftl(_ivfs_raw, dl.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
                dl.qlms_dd_mfvar = qest.library_sepTP(opj(TEMPmfvar, 'qlms_dd'), _ivfs, _ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)
            
            dl.stepper_model = it.stepper
            if dl.stepper_model.typ == 'harmonicbump':
                dl.stepper = steps.harmonicbump(dl.lmax_qlm, dl.mmax_qlm, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)


        dl = DLENSALOT_Concept()
        
        dl.dlm_mod_bool = cf.madel.dlm_mod[0]
        dl.dlm_mod_fnsuffix = cf.madel.dlm_mod[1]


        _process_Analysis(dl, cf.analysis)
        _process_Data(dl, cf.data)
        _process_Noisemodel(dl, cf.noisemodel)
        _process_Qerec(dl, cf.qerec)
        _process_Itrec(dl, cf.itrec)

        if "calc_meanfield" in dl.it_tasks or 'calc_btemplate' in dl.it_tasks:
            if dl.version == '' or dl.version == None:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{:03d}_{}'.format(dl.Nmf, dl.dlm_mod_fnsuffix))
            else:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{}_{:03d}_{}'.format(dl.version, dl.Nmf, dl.dlm_mod_fnsuffix))
            if not os.path.isdir(dl.mf_dirname) and mpi.rank == 0:
                os.makedirs(dl.mf_dirname)

        if mpi.rank == 0:
            log.info("I am going to work with the following values:")
            _str = '---------------------------------------------------\n'
            for key, val in dl.__dict__.items():
                _str += '{}:\t{}'.format(key, val)
                _str += '\n'
            _str += '---------------------------------------------------\n'
            log.info(_str)

        return dl

      
    @log_on_start(logging.INFO, "build_v3() started")
    @log_on_end(logging.INFO, "build_v3() finished")
    def build_v3(self, cf):


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
            dl.k = an.key


            # version -> version
            dl.version = an.version


            # simidxs_mf
            dl.simidxs_mf = cf.analysis.simidxs_mf
            dl.Nmf = 0 if cf.analysis.V == 'noMF' else len(dl.simidxs_mf)


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
                if nm.obd_tpl == 'template_dense':
                    # TODO need to check if tniti exists, and if tniti is the correct one
                    dl.tpl = template_dense(nm.lmin_blm, dl.ninvjob_geometry, dl.tr, _lib_dir=dl.obd_libdir, rescal=dl.obd_rescale)
                else:
                    assert 0, "Implement if needed"
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


            # filter
            dl.qe_filter_directional = qe.filter.directional
            dl.qe_filter_data_type = qe.filter.data_type
            if dl.qe_filter_directional == 'aniso':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                lmax_plm = qe.lmax_plm
                # TODO filters can be initialised with both, ninvX_desc and ninv_X. But Plancklens' hashcheck will complain if it changed since shapes are different. Not sure which one I want to use in the future..
                # TODO using ninv_X possibly causes hashcheck to fail, as v1 == v2 won't work on arrays.
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), lmax_plm, dl.nside, dl.cls_len, dl.transf_tlm, dl.ninvt_desc,
                    marge_monopole=True, marge_dipole=True, marge_maps=[])
                if dl.lowell_treat == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=lmax_plm)
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), lmax_plm, dl.nside, dl.cls_len, transf_elm_loc[:lmax_plm+1], dl.ninvp_desc, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(lmax_plm, dl.cg_tol), bmarg_lmax=dl.lmin_blm, zbounds=dl.zbounds, _bmarg_lib_dir=dl.obd_libdir, _bmarg_rescal=dl.obd_rescale, sht_threads=dl.tr)
                elif dl.lowell_treat == 'trunc' or dl.lowell_treat == None or dl.lowell_treat == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), lmax_plm, dl.nside, dl.cls_len, dl.transf_elm, dl.ninvp_desc,
                        chain_descr=dl.chain_descr(lmax_plm, dl.cg_tol), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())

                _filter_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                _ftl_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_tlm)
                _fel_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_elm)
                _fbl_rs = np.ones(lmax_plm + 1, dtype=float) * (np.arange(lmax_plm + 1) >= dl.lmin_blm)
                dl.filter = filt_util.library_ftl(_filter_raw, lmax_plm, _ftl_rs, _fel_rs, _fbl_rs)
            elif dl.qe_filter_directional == 'iso':
                dl.filter = filt_simple.library_fullsky_alms_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, {'t':dl.transf_tlm, 'e':dl.transf_elm, 'b':dl.transf_blm}, dl.cls_len, dl.ftl, dl.fel, dl.fbl, cache=True)


            # qlms
            dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)


            # cg_tol
            dl.cg_tol = qe.cg_tol


            # ninvjob_qe_geometry
            if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)


            # chain
            dl.chain_model = qe.chain
            if dl.chain_model.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if dl.chain_model.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

            hp.alm2cl
            # qe_cl_analysis
            dl.cl_analysis = qe.cl_analysis
            if qe.cl_analysis == True:
                # TODO fix numbers for mc ocrrection and total nsims
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
            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:dl.Lmin] *= 0.


        @log_on_start(logging.INFO, "_process_Itrec() started")
        @log_on_end(logging.INFO, "_process_Itrec() finished")
        def _process_Itrec(dl, it):
            # tasks
            dl.tasks = it.tasks
            ## tasks -> mf_dirname
            if "calc_meanfield" in dl.tasks or 'calc_btemplate' in dl.tasks:
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


            # sims -> sims_MAP
            if it.filter_directional == 'aniso':
                dl.sims_MAP = utils_sims.ztrunc_sims(dl.sims, self.nside, [dl.zbounds])
            elif it.filter_directional == 'iso':
                dl.sims_MAP = self.sims


            # itmax
            dl.itmax = it.itmax


            # iterator_typ
            dl.iterator_typ = it.iterator_typ


            # filter
            dl.filter_directional = it.filter.directional
            dl.filter_data_type = it.filter.data_type
            wee = self.k == 'p_p'
            dl.ffi = remapping.deflection(dl.lenjob_pbgeometry, self.lensres, np.zeros_like(hp.Alm.getsize(4000)), it.mmax_qlm, self.tr, self.tr)
            if dl.filter_directional == 'iso':
                dl.filter = opfilt_iso_ee_wl.alm_filter_nlev_wl(dl.nlev_p, dl.ffi, dl.transf_elm, (it.filter.lmax_unl, it.filter.mmax_unl), (it.filter.lmax_len, it.filter.mmax_len), wee=wee, transf_b=dl.transf_blm, nlev_b=dl.nlev_p)
                self.k_geom = filter.ffi.geom
            elif dl.filter_directional == 'aniso':
                self.get_filter_aniso(dl.sims_MAP, dl.ffi, dl.tpl)
                ninv = [dl.sims_MAP.ztruncify(read_map(ni)) for ni in self.ninvp_desc]
                dl.filter = opfilt_ee_wl.alm_filter_ninv_wl(
                    self.ninvjob_geometry, ninv, dl.ffi, self.transf_elm,
                    (self.lmax_unl, self.mmax_unl), (self.lmax_len, self.mmax_len),
                    self.tr, dl.tpl, wee=wee, lmin_dotop=min(self.lmin_elm, self.lmin_blm), transf_blm=self.transf_blm)
                self.k_geom = filter.ffi.geom


            # lenjob_geometry
            if it.lenjob_geometry == 'thin_gauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(dl.lmax_unl, 2, zbounds=dl.zbounds_len)


            # lenjob_pbgeometry
            if it.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent))


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
                dl.stepper = steps.harmonicbump(dl.lmax_qlm, dl.mmax_qlm, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)
            

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


    @log_on_start(logging.INFO, "build_v2() started")
    @log_on_end(logging.INFO, "build_v2() finished")
    def build_v2(self, cf):
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

    Returns:
        _type_: _description_
    """

    @log_on_start(logging.INFO, "build_v2() started")
    @log_on_end(logging.INFO, "build_v2() finished")
    def build_v2(self, cf):
        def _process_Madel(dl, ma):

            dl.data_from_CFS = ma.data_from_CFS
            dl.k = cf.analysis.K
            dl.version = cf.analysis.V

            dl.imin = cf.data.IMIN
            dl.imax = cf.data.IMAX
            dl.simidxs = cf.data.simidxs if cf.data.simidxs != [] else np.arange(dl.imin, dl.imax+1)
            dl.its = [0] if ma.iterations == [] else ma.iterations

            dl.Nmf = len(cf.analysis.simidxs_mf)
            dl.Nblt = len(cf.madel.simidxs_mblt)
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

            if cf.data.data_type is None:
                log.info("must specify data_type")
                sys.exit()
            elif cf.data.data_type in ['map', 'alm']:
                dl.data_type = cf.data.data_type
            else:
                log.info("Don't understand your data_type: {}".format(cf.data.data_type))
                sys.exit()

            if cf.data.data_field is None:
                log.info("must specify data_type")
                sys.exit()
            elif cf.data.data_field in ['eb', 'qu']:
                dl.data_field = cf.data.data_field
            else:
                log.info("Don't understand your data_field: {}".format(cf.data.data_field))
                sys.exit()

            # TODO hack. this is only needed to access old s08b data
            # Remove and think of a better way of including old data without existing config file
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
                if ma.ringmask:
                    _innermask = df.get_nlev_mask(2, noisemodel_rhits_map)
                else:
                    _innermask = 0
                dl.masks = dict({ma.masks[0]:{}})
                dl.binmasks = dict({ma.masks[0]:{}})
                dl.mask_ids = ma.masks[1]
                if ma.masks[0] == 'nlevels': 
                    for mask_id in dl.mask_ids:
                        buffer = df.get_nlev_mask(mask_id, noisemodel_rhits_map)
                        if mask_id > 2:
                            innermask = np.copy(_innermask)
                        else:
                            innermask = 0
                        dl.masks[ma.masks[0]].update({mask_id:buffer-innermask})
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

        dl.btemplate_perturbative_lensremap = cf.itrec.btemplate_perturbative_lensremap

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

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build_v2(self, cf):

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

# self.data[component]['nlevel']['fs']['cl_template'][freq]['EB']
            
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
                # if 'cmbs4' in ma.edges:
                #     dl.edges.append(lc.cmbs4_edges)
                #     dl.edges_id.append('cmbs4')
                # if 'ioreco' in ma.edges:
                #     dl.edges.append(lc.ioreco_edges) 
                #     dl.edges_id.append('ioreco')
                # if 'lowell' in ma.edges:
                #     dl.edges.append(lc.lowell_edges) 
                #     dl.edges_id.append('lowell')
                # elif 'fs' in ma.edges:
                #     dl.edges.append(lc.fs_edges)
                #     dl.edges_id.append('fs')
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
            dl.imin = da.IMIN
            dl.imax = da.IMAX
            dl.simidxs = da.simidxs if da.simidxs is not None else np.arange(dl.imin, dl.imax+1)

            if 'fg' in da.class_parameters:
                dl.fg = da.class_parameters['fg']
            dl._package = da.package_
            dl._module = da.module_
            dl._class = da.class_
            dl.class_parameters = da.class_parameters
            _sims_full_name = '{}.{}'.format(dl._package, dl._module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, dl._class)(**dl.class_parameters)

            dl.ec = getattr(_sims_module, 'experiment_config')()
            dl.ic = getattr(_sims_module, 'ILC_config')()
            dl.fc = getattr(_sims_module, 'foreground')(dl.fg)
            dl.nside = cf.data.nside

            dl.beam = da.beam
            dl.lmax_transf = da.lmax_transf
            dl.transf = hp.gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf)
            

        dl = DLENSALOT_Concept()
        dl.lmax = cf.analysis.lmax_filt

        _process_Data(dl, cf.data)
        _process_Madel(dl, cf.madel)
        _process_X(dl)
        _process_Config(dl, cf.config)

        return dl
            

class l2ji_Transformer:
    """Extracts parameters needed for the interactive D.Lensalot job
    """
    def build(self, cf):
        
        def _process_Jobs(jobs):
            jobs.append({"interactive":((cf, l2i_Transformer()), lenscarf_handler.Notebook_interactor)})

        jobs = []
        _process_Jobs(jobs)

        return jobs      
        

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


@transform.case(DLENSALOT_Model_v2, l2i_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, l2ji_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Concept, l2T_Transformer)
def f2b(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_delsuffix(expr)

@transform.case(str, l2T_Transformer)
def f2c(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_OBD(expr)

@transform.case(DLENSALOT_Model_v2, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, l2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, l2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_v2, l2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, l2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_mm, l2lensrec_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.mapper(expr)
