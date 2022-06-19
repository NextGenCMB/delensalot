#!/usr/bin/env python

"""param2dlensalot.py: transformer module to build dlensalot model from parameter file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os, sys
import cffi
import psutil
from os.path import join as opj
import importlib
import traceback

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
import healpy as hp
import hashlib

# TODO Only want initialisation at this level for lenscarf and plancklens objects, so queries work (lazy loading)
import plancklens
from plancklens.helpers import mpi
from plancklens import qest, qecl, utils
from plancklens.filt import filt_util, filt_cinv
from plancklens.qcinv import cd_solve
from plancklens.qcinv import opfilt_pp

from lenscarf import utils_scarf
import lenscarf.core.handler as lenscarf_handler
from lenscarf.utils import cli
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt import utils_cinv_p as cinv_p_OBD
from lenscarf.opfilt.bmodes_ninv import template_dense

from lerepi.config.config_helper import data_functions as df
from lerepi.core.visitor import transform
from lerepi.core.metamodel.dlensalot import DLENSALOT_Concept, DLENSALOT_Model
from lerepi.core.metamodel.dlensalot_v2 import DLENSALOT_Model as DLENSALOT_Model_v2

class p2T_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        _nsims_mf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
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

    @log_on_start(logging.INFO, "Start of build_nomf()")
    @log_on_end(logging.INFO, "Finished build_nomf()")
    def build_nomf(self, cf):
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


    @log_on_start(logging.INFO, "Start of build_del_suffix()")
    @log_on_end(logging.INFO, "Finished build_del_suffix()")
    def build_del_suffix(self, dl):
        if dl.version == '':
            return os.path.join(dl.TEMP, 'plotdata')
        else:
            return os.path.join(dl.TEMP, 'plotdata', dl.version)


    @log_on_start(logging.INFO, "Start of build_OBD()")
    @log_on_end(logging.INFO, "Finished build_OBD()")
    def build_OBD(self, TEMP):

        return os.path.join(TEMP, 'OBD_matrix')


class p2lensrec_Transformer:
    """_summary_
    """

    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        @log_on_start(logging.INFO, "Start of _process_dataparams()")
        @log_on_end(logging.INFO, "Finished _process_dataparams()")
        def _process_dataparams(dl, data):
            dl.TEMP = transform(cf, p2T_Transformer())
            dl.nside = data.nside
            # TODO simplify the following two attributes
            dl.nsims_mf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
            dl.mc_sims_mf_it0 = np.arange(dl.nsims_mf)
            dl.fg = data.fg

            _ui = data.sims.split('/')
            _sims_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg)

            dl.masks = p2OBD_Transformer.get_masks(cf)

            dl.beam = data.BEAM
            dl.lmax_transf = data.lmax_transf
            dl.transf = data.transf(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

            cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
            dl.cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            dl.cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))


        @log_on_start(logging.INFO, "Start of _process_iterationparams()")
        @log_on_end(logging.INFO, "Finished _process_iterationparams()")
        def _process_iterationparams(dl, iteration):
            dl.version = iteration.V
            dl.k = iteration.K  
            dl.itmax = iteration.ITMAX
            dl.imin = iteration.IMIN
            dl.imax = iteration.IMAX
            dl.lmax_filt = iteration.lmax_filt
            
            dl.lmax_qlm = iteration.lmax_qlm
            dl.mmax_qlm = iteration.mmax_qlm
            
            dl.lmax_ivf = iteration.lmax_ivf
            dl.lmin_ivf = iteration.lmin_ivf
            dl.mmax_ivf = iteration.mmax_ivf

            dl.mmin_ivf = iteration.mmin_ivf
            dl.lmax_unl = iteration.lmax_unl
            dl.mmax_unl = iteration.mmax_unl

            dl.tol = iteration.TOL
            if 'rinf_tol4' in cf.data.TEMP_suffix:
                log.warning('tol_iter increased for this run. This is hardcoded.')
                dl.tol_iter = lambda itr : 2*10 ** (- dl.tol) if itr <= 10 else 2*10 ** (-(dl.tol+1))
            else:
                dl.tol_iter = lambda itr : 1*10 ** (- dl.tol) if itr <= 10 else 1*10 ** (-(dl.tol+1))
            dl.soltn_cond = iteration.soltn_cond # Uses (or not) previous E-mode solution as input to search for current iteration one
            dl.cg_tol = iteration.CG_TOL

            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:iteration.Lmin] *= 0.

            dl.lensres = iteration.LENSRES
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', iteration.OMP_NUM_THREADS))
            dl.iterator = iteration.ITERATOR

            if iteration.STANDARD_TRANSFERFUNCTION == True:
                dl.nlev_t = p2OBD_Transformer.get_nlevt(cf)
                dl.nlev_p = p2OBD_Transformer.get_nlevp(cf)
                
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.transf_elm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_elm)
                dl.transf_blm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl =  cli(dl.cls_len['tt'][:iteration.lmax_ivf + 1] + (dl.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel =  cli(dl.cls_len['ee'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl =  cli(dl.cls_len['bb'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl =  cli(dl.cls_unl['tt'][:iteration.lmax_ivf + 1] + (dl.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl =  cli(dl.cls_unl['ee'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl =  cli(dl.cls_unl['bb'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

            if iteration.FILTER == 'cinv_sepTP':
                dl.ninv_t = p2OBD_Transformer.get_ninvt(cf)
                dl.ninv_p = p2OBD_Transformer.get_ninvp(cf)
                # TODO cinv_t and cinv_p trigger computation. Perhaps move this to the lerepi job-level. Could be done via introducing a DLENSALOT_Filter model component
                log.info('{} starting filt_cinv.cinv_t()'.format(mpi.rank))
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), iteration.lmax_ivf,dl.nside, dl.cls_len, dl.transf_tlm, dl.ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])
                log.info('{} finished filt_cinv.cinv_t()'.format(mpi.rank))
                # TODO this could move to _OBDparams()
                if dl.OBD_type == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf)
                    log.info('{} start cinv_p_OBD.cinv_p()'.format(mpi.rank))
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, transf_elm_loc[:dl.lmax_ivf+1], dl.ninv_p, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.CG_TOL), bmarg_lmax=dl.BMARG_LCUT, zbounds=dl.zbounds, _bmarg_lib_dir=dl.BMARG_LIBDIR, _bmarg_rescal=dl.BMARG_RESCALE, sht_threads=cf.iteration.OMP_NUM_THREADS)
                    log.info('{} finished cinv_p_OBD.cinv_p()'.format(mpi.rank))
                elif dl.OBD_type == 'trunc' or dl.OBD_type == None or dl.OBD_type == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_elm, dl.ninv_p,
                        chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.CG_TOL), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())
                else:
                    log.error("Don't understand your OBD_typ input. Exiting..")
                    traceback.print_stack()
                    sys.exit()
                log.info('{} starting filt_cinv.library_cinv_sepTP()'.format(mpi.rank))
                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                log.info('{} finished filt_cinv.library_cinv_sepTP()'.format(mpi.rank))
                dl.ftl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.fel_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_elm)
                dl.fbl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= dl.lmin_blm)
                log.info('{} starting filt_util.library_ftl()'.format(mpi.rank))
                dl.ivfs   = filt_util.library_ftl(dl.ivfs_raw, iteration.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
                log.info('{} finished filt_util.library_ftl()'.format(mpi.rank))
                    
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


            if iteration.mfvar == 'same':
                dl.mfvar = None
            elif iteration.mfvar.startswith('/'):
                if os.path.isfile(iteration.mfvar):
                    dl.mfvar = iteration.mfvar
                else:
                    log.error('Not sure what to do with this meanfield: {}'.format(iteration.mfvar))

        @log_on_start(logging.INFO, "Start of _process_geometryparams()")
        @log_on_end(logging.INFO, "Finished _process_geometryparams()")
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
                # TODO zbounds_loc must be identical to data.zbounds
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=dl.zbounds)
            if geometry.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduced new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=(-1,1))
            elif geometry.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=dl.zbounds)


        @log_on_start(logging.INFO, "Start of _process_chaindescparams()")
        @log_on_end(logging.INFO, "Finished _process_chaindescparams()")
        def _process_chaindescparams(dl, cd):
            # TODO hacky solution. Redo if needed
            if cd.p6 == 'tr_cg':
                _p6 = cd_solve.tr_cg
            if cd.p7 == 'cache_mem':
                _p7 = cd_solve.cache_mem()
            dl.chain_descr = lambda p2, p5 : [
                [cd.p0, cd.p1, p2, cd.p3, cd.p4, p5, _p6, _p7]]


        @log_on_start(logging.INFO, "Start of _process_stepperparams()")
        @log_on_end(logging.INFO, "Finished _process_stepperparams()")
        def _process_stepperparams(dl, st):
            if st.typ == 'harmonicbump':
                dl.stepper = steps.harmonicbump(st.lmax_qlm, st.mmax_qlm, xa=st.xa, xb=st.xb)


        @log_on_start(logging.INFO, "Start of _process_OBDparams()")
        @log_on_end(logging.INFO, "Finished _process_OBDparams()")
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
        _process_geometryparams(dl, cf.geometry)
        _process_noisemodelparams(dl, cf.noisemodel)
        _process_dataparams(dl, cf.data)
        _process_chaindescparams(dl, cf.chain_descriptor)
        _process_iterationparams(dl, cf.iteration)
        _process_stepperparams(dl, cf.stepper)

        if mpi.rank == 0:
            log.info("I am going to work with the following values:")
            _str = '---------------------------------------------------\n'
            for key, val in dl.__dict__.items():
                _str += '{}:\t{}'.format(key, val)
                _str += '\n'
            _str += '---------------------------------------------------\n'
            log.info(_str)

        return dl


    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build_v2(self, cf):

        @log_on_start(logging.INFO, "Start of _process_Analysis()")
        @log_on_end(logging.INFO, "Finished _process_Analysis()")
        def _process_Analysis(dl, an):
            dl.temp_suffix = an.TEMP_suffix
            dl.TEMP = transform(cf, p2T_Transformer())
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', cf.job.OMP_NUM_THREADS))
            dl.version = an.V
            dl.k = an.K
            dl.itmax = an.ITMAX
            dl.nsims_mf = 0 if cf.analysis.V == 'noMF' else cf.analysis.nsims_mf
            dl.mc_sims_mf_it0 = np.arange(dl.nsims_mf)
            if an.zbounds[0] == 'nmr_relative':
                dl.zbounds = df.get_zbounds(hp.read_map(cf.noisemodel.rhits_normalised[0]), an.zbounds[1])
            elif an.zbounds[0] == float or an.zbounds[0] == int:
                dl.zbounds = an.zbounds
            else:
                log.error('Not sure what to do with this zbounds: {}'.format(an.zbounds))
                traceback.print_stack()
                sys.exit()
            if an.zbounds_len[0] == 'extend':
                dl.zbounds_len = df.extend_zbounds(dl.zbounds, degrees=an.zbounds_len[1])
            elif an.zbounds_len[0] == 'max':
                  dl.zbounds_len = [-1, 1]
            elif an.zbounds_len[0] == float or an.zbounds_len[0] == int:
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

            _cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
            dl.cls_unl = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            dl.cls_len = utils.camb_clfile(opj(_cls_path, 'FFP10_wdipole_lensedCls.dat'))

            dl.STANDARD_TRANSFERFUNCTION  = an.STANDARD_TRANSFERFUNCTION 
            if dl.STANDARD_TRANSFERFUNCTION == True:
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(cf.data.beam/180 / 60 * np.pi, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_tlm)
                dl.transf_elm = gauss_beam(cf.data.beam/180 / 60 * np.pi, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_elm)
                dl.transf_blm = gauss_beam(cf.data.beam/180 / 60 * np.pi, lmax=an.lmax_ivf) * (np.arange(an.lmax_ivf + 1) >= cf.noisemodel.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl =  cli(dl.cls_len['tt'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel =  cli(dl.cls_len['ee'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl =  cli(dl.cls_len['bb'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl =  cli(dl.cls_unl['tt'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl =  cli(dl.cls_unl['ee'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl =  cli(dl.cls_unl['bb'][:an.lmax_ivf + 1] + (cf.noisemodel.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

        @log_on_start(logging.INFO, "Start of _process_Data()")
        @log_on_end(logging.INFO, "Finished _process_Data()")
        def _process_Data(dl, da):
            dl.imin = da.IMIN
            dl.imax = da.IMAX

            _package = da.package_
            _module = da.module_
            _class = da.class_
            dl.dataclass_parameters = da.class_parameters
            dl.nside = da.nside

            _sims_full_name = '{}.{}'.format(_package, _module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, _class)(**dl.dataclass_parameters)

            if 'fg' in dl.dataclass_parameters:
                dl.fg = dl.dataclass_parameters['fg']
            dl.beam = da.beam
            dl.lmax_transf = da.lmax_transf
            dl.transf_data = hp.gauss_beam(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

        @log_on_start(logging.INFO, "Start of _process_Noisemodel()")
        @log_on_end(logging.INFO, "Finished _process_Noisemodel()")
        def _process_Noisemodel(dl, nm):
            if nm.ninvjob_geometry == 'healpix_geometry':
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)
            dl.OBD_type = nm.typ
            dl.BMARG_LIBDIR = nm.BMARG_LIBDIR
            dl.BMARG_LCUT = nm.BMARG_LCUT
            dl.BMARG_RESCALE = nm.BMARG_RESCALE

            if dl.OBD_type == 'OBD':
                # TODO need to check if tniti exists, and if tniti is the correct one
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

            dl.CENTRALNLEV_UKAMIN = nm.CENTRALNLEV_UKAMIN
            dl.nlev_t = p2OBD_Transformer.get_nlevt(cf)
            dl.nlev_p = p2OBD_Transformer.get_nlevp(cf)
            dl.nlev_dep = nm.nlev_dep
            dl.inf = nm.inf
            dl.masks = p2OBD_Transformer.get_masks(cf)

            dl.rhits_normalised = nm.rhits_normalised
            

        @log_on_start(logging.INFO, "Start of _process_Qerec()")
        @log_on_end(logging.INFO, "Finished _process_Qerec()")
        def _process_Qerec(dl, qe):
            dl.lmax_qlm = qe.lmax_qlm
            dl.mmax_qlm = qe.mmax_qlm
            dl.cg_tol = qe.CG_TOL

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
                # Introduced new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=(-1,1))
            elif qe.ninvjob_qe_geometry == 'healpix_geometry':
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)

            if qe.FILTER_QE == 'sepTP':
                dl.ninv_t = p2OBD_Transformer.get_ninvt(cf)
                dl.ninv_p = p2OBD_Transformer.get_ninvp(cf)
                log.info('{} starting filt_cinv.cinv_t()'.format(mpi.rank))
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_tlm, dl.ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])
                log.info('{} finished filt_cinv.cinv_t()'.format(mpi.rank))
                
                if dl.OBD_type == 'OBD':
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=dl.lmax_ivf)
                    log.info('{} start cinv_p_OBD.cinv_p()'.format(mpi.rank))
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, transf_elm_loc[:dl.lmax_ivf+1], dl.ninv_p, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(dl.lmax_ivf, dl.cg_tol), bmarg_lmax=dl.BMARG_LCUT, zbounds=dl.zbounds, _bmarg_lib_dir=dl.BMARG_LIBDIR, _bmarg_rescal=dl.BMARG_RESCALE, sht_threads=dl.tr)
                    log.info('{} finished cinv_p_OBD.cinv_p()'.format(mpi.rank))
                elif dl.OBD_type == 'trunc' or dl.OBD_type == None or dl.OBD_type == 'None':
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_elm, dl.ninv_p,
                        chain_descr=dl.chain_descr(dl.lmax_ivf, dl.CG_TOL), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())
                else:
                    log.error("Don't understand your OBD_typ input. Exiting..")
                    traceback.print_stack()
                    sys.exit()
                log.info('{} starting filt_cinv.library_cinv_sepTP()'.format(mpi.rank))
                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                log.info('{} finished filt_cinv.library_cinv_sepTP()'.format(mpi.rank))
                dl.ftl_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_tlm)
                dl.fel_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_elm)
                dl.fbl_rs = np.ones(dl.lmax_ivf + 1, dtype=float) * (np.arange(dl.lmax_ivf + 1) >= dl.lmin_blm)
                log.info('{} starting filt_util.library_ftl()'.format(mpi.rank))
                dl.ivfs   = filt_util.library_ftl(dl.ivfs_raw, dl.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)
                log.info('{} finished filt_util.library_ftl()'.format(mpi.rank))

                log.info('{} starting qest.library_sepTP()'.format(mpi.rank))
                dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=dl.lmax_qlm)
                log.info('{} finished qest.library_sepTP()'.format(mpi.rank))   
            else:
                assert 0, 'Implement if needed'

            dl.QE_LENSING_CL_ANALYSIS = qe.QE_LENSING_CL_ANALYSIS # Change only if a full, Planck-like QE lensing power spectrum analysis is desired
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


        @log_on_start(logging.INFO, "Start of _process_Itrec()")
        @log_on_end(logging.INFO, "Finished _process_Itrec()")
        def _process_Itrec(dl, it):
            assert it.FILTER == 'opfilt_ee_wl.alm_filter_ninv_wl', 'Implement if needed, MAP filter needs to move to p2d'
            dl.FILTER = it.FILTER

            dl.tol = it.TOL
            if 'rinf_tol4' in cf.analysis.TEMP_suffix:
                log.warning('tol_iter increased for this run. This is hardcoded.')
                dl.tol_iter = lambda itr : 2*10 ** (- dl.tol) if itr <= 10 else 2*10 ** (-(dl.tol+1))
            else:
                dl.tol_iter = lambda itr : 1*10 ** (- dl.tol) if itr <= 10 else 1*10 ** (-(dl.tol+1))
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


class p2OBD_Transformer:
    """Extracts all parameters needed for building consistent OBD
    """
    @log_on_start(logging.INFO, "Start of get_nlrh_map()")
    @log_on_end(logging.INFO, "Finished get_nlrh_map()")
    def get_nlrh_map(cf):
        noisemodel_rhits_map = df.get_nlev_mask(cf.noisemodel.rhits_normalised[1], hp.read_map(cf.noisemodel.rhits_normalised[0]))
        noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf

        return noisemodel_rhits_map


    # @log_on_start(logging.INFO, "Start of get_nlevt()")
    # @log_on_end(logging.INFO, "Finished get_nlevt()")
    def get_nlevt(cf):
        nlev_t = cf.data.CENTRALNLEV_UKAMIN/np.sqrt(2) if cf.noisemodel.nlev_t == None else cf.noisemodel.nlev_t

        return nlev_t


    # @log_on_start(logging.INFO, "Start of get_nlevp()")
    # @log_on_end(logging.INFO, "Finished get_nlevp()")
    def get_nlevp(cf):
        nlev_p = cf.noisemodel.CENTRALNLEV_UKAMIN if cf.noisemodel.nlev_p == None else cf.noisemodel.nlev_p

        return nlev_p


    @log_on_start(logging.INFO, "Start of get_ninvt()")
    @log_on_end(logging.INFO, "Finished get_ninvt()")
    def get_ninvt(cf):
        nlev_t = p2OBD_Transformer.get_nlevp(cf)
        masks, noisemodel_rhits_map =  p2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        t_transf = gauss_beam(cf.data.beam/180 / 60 * np.pi, lmax=cf.analysis.lmax_ivf)
        ninv_desc = [[np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_t ** 2])/noisemodel_norm] + masks]
        ninv_t = opfilt_pp.alm_filter_ninv(ninv_desc, t_transf, marge_qmaps=(), marge_umaps=()).get_ninv()

        return ninv_t


    @log_on_start(logging.INFO, "Start of get_ninvp()")
    @log_on_end(logging.INFO, "Finished get_ninvp()")
    def get_ninvp(cf):
        nlev_p = p2OBD_Transformer.get_nlevp(cf)
        masks, noisemodel_rhits_map =  p2OBD_Transformer.get_masks(cf)
        noisemodel_norm = np.max(noisemodel_rhits_map)
        b_transf = gauss_beam(cf.data.beam/180 / 60 * np.pi, lmax=cf.analysis.lmax_ivf) # TODO ninv_p doesn't depend on this anyway, right?
        ninv_desc = [[np.array([hp.nside2pixarea(cf.data.nside, degrees=True) * 60 ** 2 / nlev_p ** 2])/noisemodel_norm] + masks]
        ninv_p = opfilt_pp.alm_filter_ninv(ninv_desc, b_transf, marge_qmaps=(), marge_umaps=()).get_ninv()

        return ninv_p


    # @log_on_start(logging.INFO, "Start of get_masks()")
    # @log_on_end(logging.INFO, "Finished get_masks()")
    def get_masks(cf):
        masks = []
        if cf.noisemodel.rhits_normalised is not None:
            msk = p2OBD_Transformer.get_nlrh_map(cf)
            masks.append(msk)
        if cf.noisemodel.mask[0] == 'nlev':
            noisemodel_rhits_map = msk.copy()
            _mask = df.get_nlev_mask(cf.noisemodel.mask[1], noisemodel_rhits_map)
            _mask = np.where(_mask>0., 1., 0.)
            masks.append(_mask)

        return masks, msk


    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        @log_on_start(logging.INFO, "Start of _process_builOBDparams()")
        @log_on_end(logging.INFO, "Finished _process_builOBDparams()")
        def _process_builOBDparams(dl, nm):
            _TEMP = transform(cf, p2T_Transformer())
            dl.TEMP = transform(_TEMP, p2T_Transformer())
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
                dl.masks, dl.rhits_map = p2OBD_Transformer.get_masks(cf)
                dl.nlev_p = p2OBD_Transformer.get_nlevp(cf)
                dl.ninv_p = p2OBD_Transformer.get_ninvp(cf)


        dl = DLENSALOT_Concept()
        _process_builOBDparams(dl, cf.noisemodel)

        return dl


    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build_v2(self, cf):
        @log_on_start(logging.INFO, "Start of _process_builOBDparams()")
        @log_on_end(logging.INFO, "Finished _process_builOBDparams()")
        def _process_Noisemodel(dl, nm):
            _TEMP = transform(cf, p2T_Transformer())
            dl.TEMP = transform(_TEMP, p2T_Transformer())
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
                dl.masks, dl.rhits_map = p2OBD_Transformer.get_masks(cf)
                dl.nlev_p = p2OBD_Transformer.get_nlevp(cf)
                dl.ninv_p = p2OBD_Transformer.get_ninvp(cf)


        dl = DLENSALOT_Concept()
        _process_Noisemodel(dl, cf.noisemodel)

        return dl


class p2d_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        # TODO make this an option for the user. If needed, user can define their own edges via configfile.
        fs_edges = np.arange(2, 3000, 20)
        ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
        cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
        def _process_delensingparams(dl, de):
            dl.k = cf.iteration.K # Lensing key, either p_p, ptt, p_eb
            dl.version = cf.iteration.V # version, can be 'noMF'
            if de.edges == 'ioreco':
                dl.edges = ioreco_edges
            elif de.edges == 'cmbs4':
                dl.edges = cmbs4_edges
            elif de.edges == 'fs':
                dl.edges = fs_edges
            dl.edges_center = (dl.edges[1:]+dl.edges[:-1])/2.
            dl.imin = de.IMIN
            dl.imax = de.IMAX
            dl.iterations = de.ITMAX
            dl.droplist = de.droplist
            dl.fg = de.fg
 
            _ui = de.base_mask.split('/')
            _sims_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg)

            mask_path = cf.noisemodel.rhits_normalised[0] # dl.sims.p2mask
            dl.base_mask = np.nan_to_num(hp.read_map(mask_path))
            dl.TEMP = transform(cf, p2T_Transformer())
            dl.analysis_path = dl.TEMP.split('/')[-1]
            dl.TEMP_DELENSED_SPECTRUM = transform(dl, p2T_Transformer())
            dl.nlev_mask = dict()
            # TODO this can possibly be simplified
            noisemodel_rhits_map = df.get_nlev_mask(np.inf, hp.read_map(cf.noisemodel.rhits_normalised[0]))
            noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf
            for nlev in de.nlevels:
                buffer = df.get_nlev_mask(nlev, noisemodel_rhits_map)
                dl.nlev_mask.update({nlev:buffer})

            dl.nlevels = de.nlevels
            dl.nside = de.nside
            dl.lmax_cl = de.lmax_cl
            dl.lmax_lib = 3*dl.lmax_cl-1
            dl.beam = de.beam
            dl.lmax_transf = de.lmax_transf
            if de.transf == 'gauss':
                dl.transf = hp.gauss_beam(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

            if de.Cl_fid == 'ffp10':
                dl.cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
                dl.cls_len = utils.camb_clfile(opj(dl.cls_path, 'FFP10_wdipole_lensedCls.dat'))
                dl.clg_templ = dl.cls_len['ee']
                dl.clc_templ = dl.cls_len['bb']
                dl.clg_templ[0] = 1e-32
                dl.clg_templ[1] = 1e-32

            dl.sha_edges = hashlib.sha256()
            dl.sha_edges.update(str(dl.edges).encode())
            dl.dirid = dl.sha_edges.hexdigest()[:4] 


        dl = DLENSALOT_Concept()
        _process_delensingparams(dl, cf.map_delensing)

        return dl


    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build_v2(self, cf):
        # TODO make this an option for the user. If needed, user can define their own edges via configfile.
        fs_edges = np.arange(2, 3000, 20)
        ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
        cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
        def _process_Madel(dl, ma):
            dl.k = cf.analysis.K # Lensing key, either p_p, ptt, p_eb
            dl.version = cf.analysis.V # version, can be 'noMF'
            if ma.edges == 'ioreco':
                dl.edges = ioreco_edges
            elif ma.edges == 'cmbs4':
                dl.edges = cmbs4_edges
            elif ma.edges == 'fs':
                dl.edges = fs_edges
            dl.edges_center = (dl.edges[1:]+dl.edges[:-1])/2.
            dl.imin = cf.analysis.IMIN
            dl.imax = cf.analysis.IMAX
            dl.iterations = ma.iterations
            dl.droplist = ma.droplist
            if 'fg' in cf.data.dataclass_parameters:
                dl.fg = cf.data.dataclass_parameters['fg']
 
            _package = cf.data.package_
            _module = cf.data.module_
            _class = cf.data.class_
            dl.dataclass_parameters = cf.data.class_parameters
            _sims_full_name = '{}.{}'.format(_package, _module)
            _sims_module = importlib.import_module(_sims_full_name)
            dl.sims = getattr(_sims_module, _class)(**dl.dataclass_parameters)

            mask_path = cf.noisemodel.rhits_normalised[0] # dl.sims.p2mask
            dl.base_mask = np.nan_to_num(hp.read_map(mask_path))
            dl.TEMP = transform(cf, p2T_Transformer())
            dl.analysis_path = dl.TEMP.split('/')[-1]
            dl.TEMP_DELENSED_SPECTRUM = transform(dl, p2T_Transformer())
            dl.nlev_mask = dict()
            # TODO this can possibly be simplified
            noisemodel_rhits_map = df.get_nlev_mask(np.inf, hp.read_map(cf.noisemodel.rhits_normalised[0]))
            noisemodel_rhits_map[noisemodel_rhits_map == np.inf] = cf.noisemodel.inf
            dl.nlevels = ma.nlevels
            for nlev in ma.nlevels:
                buffer = df.get_nlev_mask(nlev, noisemodel_rhits_map)
                dl.nlev_mask.update({nlev:buffer})

            dl.lmax_cl = ma.lmax_cl
            dl.lmax_lib = 3*dl.lmax_cl-1
            dl.beam = cf.data.beam
            dl.lmax_transf = cf.data.lmax_transf
            dl.transf = hp.gauss_beam(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

            if ma.Cl_fid == 'ffp10':
                dl.cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
                dl.cls_len = utils.camb_clfile(opj(dl.cls_path, 'FFP10_wdipole_lensedCls.dat'))
                dl.clg_templ = dl.cls_len['ee']
                dl.clc_templ = dl.cls_len['bb']
                dl.clg_templ[0] = 1e-32
                dl.clg_templ[1] = 1e-32

            dl.sha_edges = hashlib.sha256()
            dl.sha_edges.update(str(dl.edges).encode())
            dl.dirid = dl.sha_edges.hexdigest()[:4] 


        dl = DLENSALOT_Concept()
        _process_Madel(dl, cf.madel)

        return dl


class p2j_Transformer:
    """Extracts parameters needed for the specific D.Lensalot jobs
    Implement if needed
    """
    def build(self, cf):
        
        # TODO if the pf.X objects were distinguishable by X2X_Transformer, could replace the seemingly redundant checks here.
        def _process_Jobs(jobs, jb):
            if jb.build_OBD:
                jobs.append(((cf, p2OBD_Transformer()), lenscarf_handler.OBD_builder))
            if jb.QE_lensrec:
                jobs.append(((cf, p2lensrec_Transformer()), lenscarf_handler.QE_lr))
            if jb.MAP_lensrec:
                jobs.append(((cf, p2lensrec_Transformer()), lenscarf_handler.MAP_lr))
            if jb.Btemplate_per_iteration:
                jobs.append(((cf, p2lensrec_Transformer()), lenscarf_handler.B_template_constructor))
            if jb.map_delensing:
                jobs.append(((cf, p2d_Transformer()), lenscarf_handler.Map_delenser))
            if jb.inspect_result:
                # TODO maybe use this to return something interactive? Like a webservice with all plots dynamic? Like a dashboard..
                assert 0, "Implement if needed"

        jobs = []
        _process_Jobs(jobs, cf.job)

        return jobs


@transform.case(DLENSALOT_Model, p2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2T_Transformer)
def f2a(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Concept, p2T_Transformer)
def f2b(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_del_suffix(expr)

@transform.case(str, p2T_Transformer)
def f2c(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_OBD(expr)

@transform.case(DLENSALOT_Model, p2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_v2, p2OBD_Transformer)
def f4(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, p2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)

@transform.case(DLENSALOT_Model_v2, p2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model_v2, p2T_Transformer)
def f2a2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_nomf(expr)

@transform.case(DLENSALOT_Model_v2, p2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build_v2(expr)
