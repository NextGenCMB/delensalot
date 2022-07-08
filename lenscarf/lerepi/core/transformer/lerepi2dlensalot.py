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
from plancklens.qcinv import opfilt_pp

from lenscarf import utils_scarf
from lenscarf.utils import cli
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt import utils_cinv_p as cinv_p_OBD
import lenscarf.core.handler as lenscarf_handler
from lenscarf.opfilt.bmodes_ninv import template_dense

from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.config.config_helper import data_functions as df, LEREPI_Constants as lc
from lenscarf.lerepi.core.metamodel.dlensalot import DLENSALOT_Concept, DLENSALOT_Model
from lenscarf.lerepi.core.metamodel.dlensalot_v2 import DLENSALOT_Model as DLENSALOT_Model_v2

class l2T_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    # @log_on_start(logging.INFO, "build() started")
    # @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        _nsims_mf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
        ovw = _nsims_mf
        _suffix = cf.data.sims.split('/')[1]+'_%s'%(cf.data.fg)
        if cf.noisemodel.typ == 'OBD':
            _suffix += '_OBD'
        elif cf.noisemodel.typ == 'trunc':
            _suffix += '_OBDtrunc'+str(cf.noisemodel.lmin_blm)
        elif cf.noisemodel.typ == 'None' or cf.noisemodel.typ == None:
            _suffix += '_noOBD'

        _suffix += '_MF%s'%(_nsims_mf) if _nsims_mf > 0 else ''
        if cf.data.TEMP_suffix != '':
            _suffix += '_'+cf.data.TEMP_suffix
        TEMP =  opj(os.environ['SCRATCH'], cf.data.sims.split('/')[0], _suffix)

        return TEMP


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


class l2lensrec_Transformer:
    """_summary_
    """

    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        @log_on_start(logging.INFO, "_process_dataparams() started")
        @log_on_end(logging.INFO, "_process_dataparams() finished")
        def _process_dataparams(dl, data):
            dl.TEMP = transform(cf, l2T_Transformer())
            dl.nside = data.nside
            # TODO simplify the following two attributes
            dl.Nmf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
            dl.fg = data.fg

            _ui = data.sims.split('/')
            _sims_module_name = 'lenscarf.lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg)

            dl.masks = l2OBD_Transformer.get_masks(cf)

            dl.beam = data.beam
            dl.lmax_transf = data.lmax_transf
            dl.transf = data.transf(df.a2r(dl.beam), lmax=dl.lmax_transf)

            cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
            dl.cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            dl.cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))


        @log_on_start(logging.INFO, "_process_iterationparams() started")
        @log_on_end(logging.INFO, "_process_iterationparams() finished")
        def _process_iterationparams(dl, iteration):
            dl.ivfs_qe = iteration.ivfs
            dl.filter = iteration.filter
            # TODO hack. We always want to subtract it atm. But possibly not in the future.
            if "QE_subtract_meanfield" in iteration.__dict__:
                # dl.subtract_meanfield = iteration.QE_subtract_meanfield
                dl.subtract_meanfield = True
            else:
                dl.subtract_meanfield = True
            # TODO hack. Think of a better way of including mfvar
            if iteration.mfvar == 'same' or iteration.mfvar == '':
                dl.mfvar = None
            elif iteration.mfvar.startswith('/'):
                if os.path.isfile(iteration.mfvar):
                    dl.mfvar = iteration.mfvar
                else:
                    log.error('Not sure what to do with this meanfield: {}'.format(iteration.mfvar))

            dl.version = iteration.V
            dl.k = iteration.K  
            dl.itmax = iteration.ITMAX
            dl.imin = iteration.IMIN
            dl.imax = iteration.IMAX
            dl.simidxs = np.arange(dl.imin,dl.imax+1)
            dl.simidxs_mf = np.arange(dl.imin, dl.imax+1)
            dl.lmax_filt = iteration.lmax_filt
            
            dl.lmax_qlm = iteration.lmax_qlm
            dl.mmax_qlm = iteration.mmax_qlm
            
            dl.lmax_ivf = iteration.lmax_ivf
            dl.lmin_ivf = iteration.lmin_ivf
            dl.mmax_ivf = iteration.mmax_ivf

            dl.mmin_ivf = iteration.mmin_ivf
            dl.lmax_unl = iteration.lmax_unl
            dl.mmax_unl = iteration.mmax_unl

            dl.TEMP_suffix = cf.data.TEMP_suffix

            dl.tol = iteration.TOL
            dl.soltn_cond = iteration.soltn_cond # Uses (or not) previous E-mode solution as input to search for current iteration one
            if iteration.cg_tol < 1.:
                # TODO hack. For cases where TOL is not only the exponent. Remove exponent-only version.
                if 'tol5e5' in cf.data.TEMP_suffix:
                    dl.cg_tol = lambda itr : iteration.cg_tol
                else:
                    dl.cg_tol = lambda itr : iteration.cg_tol if itr <= 10 else iteration.cg_tol*0.1
            else:
                if 'rinf_tol4' in cf.data.TEMP_suffix:
                    log.warning('tol_iter increased for this run. This is hardcoded.')
                    dl.cg_tol = lambda itr : 2*10 ** (- dl.cg_tol) if itr <= 10 else 2*10 ** (-(dl.cg_tol+1))
                elif 'tol5e5' in cf.data.TEMP_suffix:
                    dl.cg_tol = lambda itr : 1*10 ** (- dl.cg_tol) 
                else:
                    dl.cg_tol = lambda itr : 1*10 ** (- dl.cg_tol) if itr <= 10 else 1*10 ** (-(dl.cg_tol+1))

            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:iteration.Lmin] *= 0.

            dl.lensres = iteration.LENSRES
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', iteration.OMP_NUM_THREADS))
            dl.iterator_typ = iteration.iterator_typ

            if iteration.STANDARD_TRANSFERFUNCTION == True:
                dl.nlev_t = l2OBD_Transformer.get_nlevt(cf)
                dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(df.a2r(dl.beam), lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.transf_elm = gauss_beam(df.a2r(dl.beam), lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_elm)
                dl.transf_blm = gauss_beam(df.a2r(dl.beam), lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl = cli(dl.cls_len['tt'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm**2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm**2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm**2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm**2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm**2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:iteration.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm**2)) * (dl.transf_blm > 0)

            if iteration.filter == 'cinv_sepTP':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                # TODO filters can be initialised with both, ninvX_desc and ninv_X. But Plancklens' hashcheck will complain if it changed since shapes are different.
                # TODO using ninv_X causes hashcheck to fail, as these ninv are List[np.array] and Plancklens checks for List[np.ndarray]

                # TODO cinv_t and cinv_p trigger computation. Perhaps move this to the lerepi job-level. Could be done via introducing a DLENSALOT_Filter model component
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), iteration.lmax_ivf,dl.nside, dl.cls_len, dl.transf_tlm, dl.ninvt_desc,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])
                if dl.OBD_type == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf)
                    dl.cinv_p = cinv_p_OBD.cinv_p(
                        opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, transf_elm_loc[:dl.lmax_ivf+1], 
                        dl.ninvp_desc, geom=dl.ninvjob_qe_geometry, chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.cg_tol),
                        bmarg_lmax=dl.BMARG_LCUT, zbounds=dl.zbounds, _bmarg_lib_dir=dl.BMARG_LIBDIR, _bmarg_rescal=dl.BMARG_RESCALE,
                        sht_threads=cf.iteration.OMP_NUM_THREADS)
                elif dl.OBD_type == 'trunc' or dl.OBD_type == None or dl.OBD_type == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(
                        opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len,
                        dl.transf_elm, dl.ninvp_desc, chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.cg_tol), transf_blm=dl.transf_blm,
                        marge_qmaps=(), marge_umaps=())
                else:
                    log.error("Don't understand your OBD_typ input. Exiting..")
                    traceback.print_stack()
                    sys.exit()
                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                dl.ftl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.fel_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_elm)
                dl.fbl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_blm)
                dl.ivfs   = filt_util.library_ftl(dl.ivfs_raw, iteration.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
                dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)

                if dl.mfvar:
                    # TODO this is a terrible way of replacing foreground..
                    TEMPmfvar = dl.TEMP.replace('_00_', "_{}_".format(dl.version[2:4]))
                    _ivfs_raw = filt_cinv.library_cinv_sepTP(opj(TEMPmfvar, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                    _ivfs = filt_util.library_ftl(_ivfs_raw, iteration.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
                    dl.qlms_dd_mfvar = qest.library_sepTP(opj(TEMPmfvar, 'qlms_dd'), _ivfs, _ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)
    
            if iteration.QE_LENSING_CL_ANALYSIS == True:
                dl.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                        np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
                dl.ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

                dl.ivfs_d = filt_util.library_shuffle(dl.ivfs, iteration.ds_dict)
                dl.ivfs_s = filt_util.library_shuffle(dl.ivfs, iteration.ss_dict)

                dl.qlms_ds = qest.library_sepTP(opj(dl.TEMP, 'qlms_ds'), iteration.ivfs, iteration.ivfs_d, dl.cls_len['te'], dl.nside, lmax_qlm=iteration.lmax_qlm)
                dl.qlms_ss = qest.library_sepTP(opj(dl.TEMP, 'qlms_ss'), iteration.ivfs, iteration.ivfs_s, dl.cls_len['te'], dl.nside, lmax_qlm=iteration.lmax_qlm)

                dl.mc_sims_bias = np.arange(60, dtype=int)
                dl.mc_sims_var  = np.arange(60, 300, dtype=int)

                dl.qcls_ds = qecl.library(opj(dl.TEMP, 'qcls_ds'), dl.qlms_ds, dl.qlms_ds, np.array([]))  # for QE RDN0 calculations
                dl.qcls_ss = qecl.library(opj(dl.TEMP, 'qcls_ss'), dl.qlms_ss, dl.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
                dl.qcls_dd = qecl.library(opj(dl.TEMP, 'qcls_dd'), dl.qlms_dd, dl.qlms_dd, dl.mc_sims_bias)


        @log_on_start(logging.INFO, "_process_geometryparams() started")
        @log_on_end(logging.INFO, "_process_geometryparams() finished")
        def _process_geometryparams(dl, geometry):
            dl.pb_ctr, dl.pb_extent = geometry.pbounds
            if geometry.zbounds[0] == 'nmr_relative':
                dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), geometry.zbounds[1])
            elif geometry.zbounds[0] == float or geometry.zbounds[0] == int:
                dl.zbounds = geometry.zbounds
            else:
                log.error('Not sure what to do with this zbounds: {}'.format(geometry.zbounds))
                traceback.print_stack()
                sys.exit()
            if geometry.zbounds_len[0] == 'extend':
                dl.zbounds_len = df.extend_zbounds(dl.zbounds, degrees=geometry.zbounds_len[1])
            elif geometry.zbounds_len[0] == 'max':
                  dl.zbounds_len = [-1, 1]
            elif geometry.zbounds_len[0] == float or geometry.zbounds_len[0] == int:
                dl.zbounds_len = geometry.zbounds_len
            else:
                log.error('Not sure what to do with this zbounds_len: {}'.format(geometry.zbounds_len))
                traceback.print_stack()
                sys.exit()

            if geometry.lenjob_geometry == 'thin_gauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(geometry.lmax_unl, 2, zbounds=dl.zbounds_len)
            if geometry.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent))
            if geometry.ninvjob_geometry == 'healpix_geometry':
                # ninv MAP geometry. Could be merged with QE, if next comment resolved
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=dl.zbounds)
            if geometry.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduced new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=(-1,1))
            elif geometry.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=dl.zbounds)


        @log_on_start(logging.INFO, "_process_chaindescparams() started")
        @log_on_end(logging.INFO, "_process_chaindescparams() finished")
        def _process_chaindescparams(dl, cd):
            # TODO hacky solution. Redo if needed
            if cd.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if cd.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [cd.p0, cd.p1, p2, cd.p3, cd.p4, p5, _p6, _p7]]


        @log_on_start(logging.INFO, "_process_stepperparams() started")
        @log_on_end(logging.INFO, "_process_stepperparams() finished")
        def _process_stepperparams(dl, st):
            if st.typ == 'harmonicbump':
                dl.stepper = steps.harmonicbump(st.lmax_qlm, st.mmax_qlm, xa=st.xa, xb=st.xb)


        @log_on_start(logging.INFO, "_process_noisemodelparams() started")
        @log_on_end(logging.INFO, "_process_OBDparams() finished")
        def _process_noisemodelparams(dl, nm):
            dl.OBD_type = nm.typ
            dl.BMARG_LCUT = nm.BMARG_LCUT
            dl.BMARG_LIBDIR = nm.BMARG_LIBDIR
            dl.BMARG_RESCALE = nm.BMARG_RESCALE
            if dl.OBD_type == 'OBD':
                # TODO need to check if tniti exists, and if tniti is the correct one
                if cf.data.tpl == 'template_dense':
                    def tpl_kwargs(lmax_marg, geom, sht_threads, _lib_dir=None, rescal=1.):
                        return locals()
                    dl.tpl = template_dense
                    dl.tpl_kwargs = tpl_kwargs(nm.BMARG_LCUT, dl.ninvjob_geometry, cf.iteration.OMP_NUM_THREADS, _lib_dir=dl.BMARG_LIBDIR, rescal=dl.BMARG_RESCALE) 
                else:
                    assert 0, "Implement if needed"
                # TODO need to initialise as function expect it, but do I want this? Shouldn't be needed
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
                # TODO are 0s a good value? 
                dl.lmin_tlm = 0
                dl.lmin_elm = 0
                dl.lmin_blm = 0
            else:
                log.error("Don't understand your OBD_type input. Exiting..")
                traceback.print_stack()
                sys.exit()


        dl = DLENSALOT_Concept()

        dl.dlm_mod_bool = cf.map_delensing.dlm_mod
        _process_geometryparams(dl, cf.geometry)
        _process_noisemodelparams(dl, cf.noisemodel)
        _process_dataparams(dl, cf.data)
        _process_chaindescparams(dl, cf.chain_descriptor)
        _process_iterationparams(dl, cf.iteration)
        _process_stepperparams(dl, cf.stepper)


        dl.tasks = cf.iteration.tasks
        # TODO hack. Refactor
        if "calc_meanfield" in dl.tasks:
            if dl.version == '' or dl.version == None:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{:03d}'.format(dl.Nmf))
            else:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{}_{:03d}'.format(dl.version, dl.Nmf))
            if not os.path.isdir(dl.mf_dirname) and mpi.rank == 0:
                os.makedirs(dl.mf_dirname)
        if mpi.rank == 0:
            # TODO possibly don't want to show this when in interactive mode
            log.info("I am going to work with the following values:")
            _str = '---------------------------------------------------\n'
            for key, val in dl.__dict__.items():
                _str += '{}:\t{}'.format(key, val)
                _str += '\n'
            _str += '---------------------------------------------------\n'
            log.info(_str)

        return dl


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
                dl.ftl = cli(dl.cls_len['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel = cli(dl.cls_len['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl = cli(dl.cls_len['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl = cli(dl.cls_unl['tt'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_t)**2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl = cli(dl.cls_unl['ee'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl = cli(dl.cls_unl['bb'][:an.lmax_ivf + 1] + df.a2r(dl.nlev_p)**2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)
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
                # TODO need to check if tniti exists, and if tniti is the correct one
                # TODO this is a weird lazy loading solution of template_dense 
                if nm.tpl == 'template_dense':
                    def tpl_kwargs(lmax_marg, geom, sht_threads, _lib_dir=None, rescal=1.):
                        return locals()
                    dl.tpl = template_dense
                    dl.tpl_kwargs = tpl_kwargs(nm.BMARG_LCUT, dl.ninvjob_geometry, dl.tr, _lib_dir=dl.BMARG_LIBDIR, rescal=dl.BMARG_RESCALE) 
                else:
                    assert 0, "Implement if needed"
                # TODO need to initialise as function expect it, but do I want this? Shouldn't be needed
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
                # TODO are 0s a good value? 
                dl.lmin_tlm = 0
                dl.lmin_elm = 0
                dl.lmin_blm = 0
            else:
                log.error("Don't understand your OBD_type input. Exiting..")
                traceback.print_stack()
                sys.exit()

            # TODO duplicate in process_analysis
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

            dl.chain_model = qe.chain
            # TODO hacky solution. Redo if needed
            if dl.chain_model.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if dl.chain_model.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [dl.chain_model.p0, dl.chain_model.p1, p2, dl.chain_model.p3, dl.chain_model.p4, p5, _p6, _p7]]

            if qe.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduce new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)

            if qe.ivfs == 'sepTP':
                dl.ninvt_desc = l2OBD_Transformer.get_ninvt(cf)
                dl.ninvp_desc = l2OBD_Transformer.get_ninvp(cf)
                # TODO filters can be initialised with both, ninvX_desc and ninv_X. But Plancklens' hashcheck will complain if it changed since shapes are different. Not sure which one I want to use in the future..
                # TODO using ninv_X possibly causes hashcheck to fail, as v1 == v2 won't work on arrays.
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

            # TODO hack. We always want to subtract it atm. But possibly not in the future.
            if "QE_subtract_meanfield" in it.__dict__:
                # dl.subtract_meanfield = iteration.QE_subtract_meanfield
                dl.subtract_meanfield = True
            else:
                dl.subtract_meanfield = True

            dl.tasks = it.tasks
            if it.cg_tol < 1.:
                # TODO hack. For cases where TOL is not only the exponent. Remove exponent-only version.
                if 'tol5e5' in cf.analysis.TEMP_suffix:
                    dl.cg_tol = lambda itr : it.cg_tol
                else:
                    dl.cg_tol = lambda itr : it.cg_tol if itr <= 10 else it.cg_tol*0.1
            else:
                if 'rinf_tol4' in cf.analysis.TEMP_suffix:
                    log.warning('tol_iter increased for this run. This is hardcoded.')
                    dl.cg_tol = lambda itr : 2*10 ** (- dl.cg_tol) if itr <= 10 else 2*10 ** (-(dl.cg_tol+1))
                elif 'tol5e5' in cf.analysis.TEMP_suffix:
                    dl.cg_tol = lambda itr : 1*10 ** (- dl.cg_tol) 
                else:
                    dl.cg_tol = lambda itr : 1*10 ** (- dl.cg_tol) if itr <= 10 else 1*10 ** (-(dl.cg_tol+1))
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
            
            dl.stepper_model = it.stepper
            if dl.stepper_model.typ == 'harmonicbump':
                # TODO stepper is needed for iterator, but depends on qe settings?
                dl.stepper = steps.harmonicbump(dl.lmax_qlm, dl.mmax_qlm, xa=dl.stepper_model.xa, xb=dl.stepper_model.xb)


        dl = DLENSALOT_Concept()
        
        dl.dlm_mod_bool = cf.madel.dlm_mod
        _process_Analysis(dl, cf.analysis)
        _process_Data(dl, cf.data)
        _process_Noisemodel(dl, cf.noisemodel)
        _process_Qerec(dl, cf.qerec)
        _process_Itrec(dl, cf.itrec)

        # TODO belongs to l2T
        if "calc_meanfield" in dl.tasks:
            if dl.version == '' or dl.version == None:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{:03d}'.format(dl.Nmf))
            else:
                dl.mf_dirname = opj(dl.TEMP, 'mf_{}_{:03d}'.format(dl.version, dl.Nmf))
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


class l2OBD_Transformer:
    """Extracts all parameters needed for building consistent OBD
    """
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
        # TODO hack, needed for v1 and v2 compatibility
        if isinstance(cf, DLENSALOT_Model):
            t_transf = gauss_beam(df.a2r(cf.data.beam), lmax=cf.iteration.lmax_ivf)
        else:
            t_transf = gauss_beam(df.a2r(cf.data.beam), lmax=cf.analysis.lmax_ivf)
        ninv_desc = [np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_t ** 2])/noisemodel_norm] + masks
        # ninv_t = opfilt_pp.alm_filter_ninv([ninv_desc], t_transf, marge_qmaps=(), marge_umaps=()).get_ninv()
        # return ninv_t, ninv_desc
        return ninv_desc


    @log_on_start(logging.INFO, "get_ninvp() started")
    @log_on_end(logging.INFO, "get_ninvp() finished")
    def get_ninvp(cf):
        nlev_p = l2OBD_Transformer.get_nlevp(cf)
        masks, noisemodel_rhits_map =  l2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        # TODO hack, needed for v1 and v2 compatibility
        if isinstance(cf, DLENSALOT_Model):
            b_transf = gauss_beam(df.a2r(cf.data.beam), lmax=cf.iteration.lmax_ivf) # TODO ninv_p doesn't depend on this anyway, right?
        else:
            b_transf = gauss_beam(df.a2r(cf.data.beam), lmax=cf.analysis.lmax_ivf) # TODO ninv_p doesn't depend on this anyway, right?
        ninv_desc = [[np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_p ** 2])/noisemodel_norm] + masks]
        # ninv_p = opfilt_pp.alm_filter_ninv(ninv_desc, b_transf, marge_qmaps=(), marge_umaps=()).get_ninv()
        # return ninv_p, ninv_desc
        return ninv_desc


    # @log_on_start(logging.INFO, "get_masks() started")
    # @log_on_end(logging.INFO, "get_masks() finished")
    def get_masks(cf):
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


    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        @log_on_start(logging.INFO, "() started")
        @log_on_end(logging.INFO, "_process_builOBDparams() finished")
        def _process_builOBDparams(dl, nm):
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
                dl.CENTRALNLEV_UKAMIN = nm.CENTRALNLEV_UKAMIN
                dl.geom = utils_scarf.Geom.get_healpix_geometry(dl.nside)
                dl.masks, dl.rhits_map = l2OBD_Transformer.get_masks(cf)
                dl.nlev_p = l2OBD_Transformer.get_nlevp(cf)
                dl.ninv_p_desc = l2OBD_Transformer.get_ninvp(cf)


        dl = DLENSALOT_Concept()
        _process_builOBDparams(dl, cf.noisemodel)

        return dl


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


class l2d_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):

        def _process_data(dl, da):
            dl.fg = da.fg
            dl.class_parameters = {'fg': da.fg}
           
            _ui = cf.data.sims.split('/')
            _sims_module_name = 'lenscarf.lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            dl._module = _sims_module_name[9:]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg)
            dl.nside = da.nside

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


        def _process_iteration(dl, it):
            dl.k = it.K
            dl.version = it.V
            dl.Nmf = it.nsims_mf
            dl.imin = it.IMIN
            dl.imax = it.IMAX
            dl.simidxs = np.arange(dl.imin, dl.imax+1)
            dl.Nmf = it.nsims_mf

            if it.STANDARD_TRANSFERFUNCTION == True:
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf)
            elif it.STANDARD_TRANSFERFUNCTION == 'with_pixwin':
                dl.transf = gauss_beam(df.a2r(dl.beam), lmax=dl.lmax_transf) * hp.pixwin(dl.nside, lmax=dl.lmax_transf)
            else:
                log.info("Don't understand your STANDARD_TRANSFERFUNCTION: {}".format(it.STANDARD_TRANSFERFUNCTION))


        def _process_TEMP(dl):
            dl.TEMP = transform(cf, l2T_Transformer())
            dl.analysis_path = dl.TEMP.split('/')[-1]
            dl.TEMP_DELENSED_SPECTRUM = transform(dl, l2T_Transformer())

            dl.vers_str = '/{}'.format(dl.version) if dl.version != '' else 'base'
            if mpi.rank == 0:
                for edgesi, edges in enumerate(dl.edges):
                    if not(os.path.isdir(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edgesi]))):
                        os.makedirs(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edgesi]))
                        log.info("dir created: {}".format(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edgesi])))


        def _process_Madel(dl, ma):
            dl.its = ma.iterations 
            if ma.libdir_it is None:
                dl.libdir_iterators = lambda qe_key, simidx, version: opj(dl.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
            else:
                dl.libdir_iterators = 'overwrite'
            if cf.noisemodel.rhits_normalised is not None:
                _mask_path = cf.noisemodel.rhits_normalised[0]
                dl.base_mask = np.nan_to_num(hp.read_map(_mask_path))
            else:
                dl.base_mask = np.ones(shape=hp.nside2npix(cf.data.nside))
            noisemodel_rhits_map = df.get_nlev_mask(np.inf, dl.base_mask)
            noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf
            if ma.masks != None:
                dl.masks = dict({ma.masks[0]:{}})
                dl.mask_ids = ma.masks[1]
                if ma.masks[0] == 'nlevels': 
                    for mask_id in dl.mask_ids:
                        buffer = df.get_nlev_mask(mask_id, noisemodel_rhits_map)
                        dl.masks[ma.masks[0]].update({mask_id:buffer})
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
                        _fsky = float("{:0.2f}".format(np.sum(buffer)/len(buffer)))
                        dl.mask_ids[fni] = _fsky
                        dl.masks[ma.masks[0]].update({_fsky:buffer})
            else:
                dl.masks = {"no":{1.00:np.ones(shape=hp.nside2npix(dl.nside))}}
                dl.mask_ids = np.array([1.00])
            
            if ma.Cl_fid == 'ffp10':
                dl.cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
                dl.cls_len = utils.camb_clfile(opj(dl.cls_path, 'FFP10_wdipole_lensedCls.dat'))
                dl.clg_templ = dl.cls_len['ee']
                dl.clc_templ = dl.cls_len['bb']
                dl.clg_templ[0] = 1e-32
                dl.clg_templ[1] = 1e-32

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
                    elif 'fs' in ma.edges:
                        dl.edges.append(lc.fs_edges)
                        dl.edges_id.append('fs')
                dl.edges = np.array(dl.edges)
                dl.sha_edges = [hashlib.sha256() for n in range(len(dl.edges))]
                for n in range(len(dl.edges)):
                    dl.sha_edges[n].update(str(dl.edges[n]).encode())
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
                dl.sha_edges[0].update('unbinned'.encode())
                dl.dirid = [dl.sha_edges[0].hexdigest()[:4]]
            else:
                log.info("Don't understand your spectrum type")
                sys.exit()

            dl.dlm_mod_bool = ma.dlm_mod
            if dl.binning == 'binned':
                if dl.dlm_mod_bool:
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod', fg)
                else:
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_fg%s_res2b3acm.npy'%(idx, fg)
            else:
                if dl.dlm_mod_bool:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod', fg)
                else:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_fg%s_res2b3acm.npy'%(idx, fg)

            if ma.spectrum_calculator == None:
                log.info("Using Healpy as powerspectrum calculator")
                dl.cl_calc = hp
            else:
                dl.cl_calc = ma.spectrum_calculator       


        dl = DLENSALOT_Concept()
        _process_data(dl, cf.data)
        _process_iteration(dl, cf.iteration)
        
        _process_Madel(dl, cf.map_delensing)
        _process_TEMP(dl)
        
        return dl


    @log_on_start(logging.INFO, "build_v2() started")
    @log_on_end(logging.INFO, "build_v2() finished")
    def build_v2(self, cf):
        def _process_Madel(dl, ma):
            dl.k = cf.analysis.K
            dl.version = cf.analysis.V

            dl.imin = cf.data.IMIN
            dl.imax = cf.data.IMAX
            dl.simidxs = cf.data.simidxs if cf.data.simidxs != [] else np.arange(dl.imin, dl.imax+1)
            dl.its = ma.iterations

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
            #     rootstr = '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/lt_recons/'
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
                dl.masks = dict({ma.masks[0]:{}})
                dl.mask_ids = ma.masks[1]
                if ma.masks[0] == 'nlevels': 
                    for mask_id in dl.mask_ids:
                        buffer = df.get_nlev_mask(mask_id, noisemodel_rhits_map)
                        dl.masks[ma.masks[0]].update({mask_id:buffer})
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
                        _fsky = float("{:0.2f}".format(np.sum(buffer)/len(buffer)))
                        dl.mask_ids[fni] = _fsky
                        dl.masks[ma.masks[0]].update({_fsky:buffer})
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
                    elif 'fs' in ma.edges:
                        dl.edges.append(lc.fs_edges)
                        dl.edges_id.append('fs')
                dl.edges = np.array(dl.edges)
                dl.sha_edges = [hashlib.sha256() for n in range(len(dl.edges))]
                for n in range(len(dl.edges)):
                    dl.sha_edges[n].update(str(dl.edges[n]).encode())
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
                dl.sha_edges[0].update('unbinned'.encode())
                dl.dirid = [dl.sha_edges[0].hexdigest()[:4]]
            else:
                log.info("Don't understand your spectrum type")
                sys.exit()

            dl.vers_str = '/{}'.format(dl.version) if dl.version != '' else 'base'
            dl.TEMP_DELENSED_SPECTRUM = transform(dl, l2T_Transformer())
            for dir_id in dl.dirid:
                if not(os.path.isdir(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dir_id))):
                    os.makedirs(dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dir_id))

            # TODO II
            # TODO fn needs changing
            dl.dlm_mod_bool = ma.dlm_mod
            if dl.binning == 'binned':
                if dl.dlm_mod_bool:
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod', fg)
                else:
                    dl.file_op = lambda idx, fg, edges_idx: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[edges_idx]) + '/ClBBwf_sim%04d_fg%s_res2b3acm.npy'%(idx, fg)
            else:
                if dl.dlm_mod_bool:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_%s_fg%s_res2b3acm.npy'%(idx, 'dlmmod', fg)
                else:
                    dl.file_op = lambda idx, fg, x: dl.TEMP_DELENSED_SPECTRUM + '/{}'.format(dl.dirid[0]) + '/ClBBwf_sim%04d_fg%s_res2b3acm.npy'%(idx, fg)

            if ma.spectrum_calculator == None:
                log.info("Using Healpy as powerspectrum calculator")
                dl.cl_calc = hp
            else:
                dl.cl_calc = ma.spectrum_calculator       
                
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

        _process_Madel(dl, cf.madel)
        _check_powspeccalculator(dl.cl_calc)

        return dl


class l2m_Transformer:
    """lerepi2meanfield transformation

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "build() started")
    @log_on_end(logging.INFO, "build() finished")
    def build(self, cf):
        assert 0, "Implement if needed"


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
                # TODO maybe use this to return something interactive
                assert 0, "Implement if needed"

        jobs = []
        _process_Jobs(jobs, cf.job)

        return jobs


@transform.case(DLENSALOT_Model, l2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2T_Transformer)
def f2a(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Concept, l2T_Transformer)
def f2b(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_delsuffix(expr)

@transform.case(str, l2T_Transformer)
def f2c(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_OBD(expr)

@transform.case(DLENSALOT_Model, l2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, l2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

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
