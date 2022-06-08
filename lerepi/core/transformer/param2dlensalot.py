#!/usr/bin/env python

"""param2dlensalot.py: transformer module to build dlensalot model from parameter file
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

# TODO Only want initialisation at this level for lenscarf and plancklens objects, so queries work (lazy loading)
import plancklens
from plancklens import qest, qecl, utils
from plancklens.filt import filt_util, filt_cinv
from plancklens.qcinv import cd_solve

from lenscarf import utils_scarf
import lenscarf.core.handler as lenscarf_handler
from lenscarf.utils import cli
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt import utils_cinv_p as cinv_p_OBD
from lenscarf.opfilt.bmodes_ninv import template_dense

from lerepi.core.metamodel.dlensalot import DLENSALOT_Concept, DLENSALOT_Model
from lerepi.core.visitor import transform


class p2T_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        _nsims_mf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
        _suffix = cf.data.sims.split('/')[1]+'_%s_r%s'%(cf.data.fg, cf.data.mask_suffix)+'_isOBD'*cf.data.isOBD
        _suffix += '_MF%s'%(_nsims_mf) if _nsims_mf > 0 else ''
        if cf.data.TEMP_suffix != '':
            _suffix += '_'+cf.data.TEMP_suffix
        TEMP =  opj(os.environ['SCRATCH'], cf.data.sims.split('/')[0], _suffix)
        return TEMP


# TODO parameters are sometimes redundant, not general, and not descriptive
# remove redundancy, remove non-general parameters, change names 
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
            dl.mask_suffix = data.mask_suffix
            dl.nside = data.nside
            dl.isOBD = data.isOBD
            # TODO simplify the following two attributes
            dl.nsims_mf = 0 if cf.iteration.V == 'noMF' else cf.iteration.nsims_mf
            dl.mc_sims_mf_it0 = np.arange(dl.nsims_mf)
            dl.rhits = hp.read_map(data.rhits)
            dl.fg = data.fg

            _ui = data.sims.split('/')
            _sims_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg, mask_suffix=dl.mask_suffix)

            _ui = data.mask.split('/')
            _mask_class_name = _ui[-1]
            _mask_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _mask_module = importlib.import_module(_mask_module_name)
            dl.mask = getattr(_mask_module, _mask_class_name)(dl.fg, mask_suffix=dl.mask_suffix).get_mask_path()
            dl.masks = [dl.mask] # TODO [masks] doesn't work as intented

            dl.beam = data.BEAM
            dl.lmax_transf = data.lmax_transf
            dl.transf = data.transf(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

            if data.zbounds[0] ==  data.sims:
                dl.zbounds = dl.sims.get_zbounds(hp.read_map(dl.mask),data.zbounds[1])
            if data.zbounds_len[0] ==  data.sims:
                dl.zbounds_len = dl.sims.extend_zbounds(dl.zbounds, data.zbounds_len[1])
            dl.pb_ctr, dl.pb_extent = data.pbounds

            dl.DATA_libdir = data.DATA_LIBDIR # TODO I don't really need this..
            dl.BMARG_LIBDIR = data.BMARG_LIBDIR
            dl.BMARG_LCUT = data.BMARG_LCUT  # TODO if tnitit != bmarg_lcut size, error. Either extract from tniti, or make it an actual user-parameter. 
            dl.BMARG_RESCALE = data.BMARG_RESCALE

            dl.CENTRALNLEV_UKAMIN = data.CENTRALNLEV_UKAMIN
            dl.nlev_t = data.CENTRALNLEV_UKAMIN/np.sqrt(2) if data.nlev_t == None else data.nlev_t
            dl.nlev_p = data.CENTRALNLEV_UKAMIN if data.nlev_p == None else data.nlev_p
    
            if data.isOBD:
                if data.tpl == 'template_dense':
                    def tpl_kwargs(lmax_marg, geom, sht_threads, _lib_dir=None, rescal=1.):
                        return locals()
                    dl.tpl = template_dense
                    dl.tpl_kwargs = tpl_kwargs(data.BMARG_LCUT, dl.ninvjob_geometry, cf.iteration.OMP_NUM_THREADS, _lib_dir=data.BMARG_LIBDIR, rescal=data.BMARG_RESCALE)
                    
                else:
                    assert 0, "Implement if needed"
            else:
                dl.tpl = None

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
            
            dl.lmin_tlm = iteration.lmin_tlm
            dl.lmin_elm = iteration.lmin_elm
            dl.lmin_blm = iteration.lmin_blm
            dl.lmax_qlm = iteration.lmax_qlm
            dl.mmax_qlm = iteration.mmax_qlm
            
            dl.lmax_ivf = iteration.lmax_ivf
            dl.mmax_ivf = iteration.mmax_ivf
            dl.lmin_ivf = iteration.lmin_ivf
            dl.lmax_unl = iteration.lmax_unl
            dl.mmax_unl = iteration.mmax_unl

            dl.tol = iteration.TOL
            dl.tol_iter = lambda itr : 10 ** (- dl.tol) if itr <= 10 else 10 ** (-(dl.tol+1)) 
            dl.soltn_cond = iteration.soltn_cond # Uses (or not) previous E-mode solution as input to search for current iteration one
            dl.cg_tol = iteration.CG_TOL

            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:iteration.Lmin] *= 0. # TODO *0 or *1e-5?

            dl.lensres = iteration.LENSRES
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', iteration.OMP_NUM_THREADS))
            dl.iterator = iteration.ITERATOR

            dl.get_btemplate_per_iteration = iteration.get_btemplate_per_iteration

            if iteration.STANDARD_TRANSFERFUNCTION == True:
                # Fiducial model of the transfer function
                dl.transf_tlm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_tlm)
                dl.transf_elm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_elm)
                dl.transf_blm = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl =  cli(dl.cls_len['tt'][:iteration.lmax_ivf + 1] + (dl.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel =  cli(dl.cls_len['ee'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl =  cli(dl.cls_len['bb'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl =  cli(dl.cls_unl['tt'][:iteration.lmax_ivf + 1] + (dl.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl =  cli(dl.cls_unl['ee'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl =  cli(dl.cls_unl['bb'][:iteration.lmax_ivf + 1] + (dl.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)


            if iteration.FILTER == 'cinv_sepTP':
                mask_norm = cf.data.mask_norm
                dl.ninv_t = [np.array([hp.nside2pixarea(dl.nside, degrees=True) * 60 ** 2 / dl.nlev_t ** 2])/mask_norm] + dl.masks
                dl.ninv_p = [[np.array([hp.nside2pixarea(dl.nside, degrees=True) * 60 ** 2 / dl.nlev_p ** 2])/mask_norm] + dl.masks]
                # TODO cinv_t adn cinv_p trigger computation. Perhaps move this to the lerepi job-level. Could be done via introducing a DLENSALOT_Filter model component
                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), iteration.lmax_ivf,dl.nside, dl.cls_len, dl.transf_tlm, dl.ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])
                if dl.isOBD:
                    transf_elm_loc = gauss_beam(dl.beam/180 / 60 * np.pi, lmax=iteration.lmax_ivf)
                    dl.cinv_p = cinv_p_OBD.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, transf_elm_loc[:dl.lmax_ivf+1], dl.ninv_p, geom=dl.ninvjob_qe_geometry,
                        chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.CG_TOL), bmarg_lmax=dl.BMARG_LCUT, zbounds=dl.zbounds, _bmarg_lib_dir=dl.BMARG_LIBDIR, _bmarg_rescal=dl.BMARG_RESCALE, sht_threads=cf.iteration.OMP_NUM_THREADS)
                else:
                    dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, dl.nside, dl.cls_len, dl.transf_elm, dl.ninv_p,
                        chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.CG_TOL), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())

                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), dl.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
                dl.ftl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_tlm)
                dl.fel_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_elm)
                dl.fbl_rs = np.ones(iteration.lmax_ivf + 1, dtype=float) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_blm)
                dl.ivfs   = filt_util.library_ftl(dl.ivfs_raw, iteration.lmax_ivf, dl.ftl_rs, dl.fel_rs, dl.fbl_rs)


            if iteration.QE_LENSING_CL_ANALYSIS == True:
                dl.ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                        np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
                dl.ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

                dl.ivfs_d = filt_util.library_shuffle(dl.ivfs, iteration.ds_dict)
                dl.ivfs_s = filt_util.library_shuffle(dl.ivfs, iteration.ss_dict)

                dl.qlms_ds = qest.library_sepTP(opj(dl.TEMP, 'qlms_ds'), iteration.ivfs, iteration.ivfs_d, dl.cls_len['te'], dl.nside, lmax_qlm=iteration.lmax_qlm)
                dl.qlms_ss = qest.library_sepTP(opj(dl.TEMP, 'qlms_ss'), iteration.ivfs, iteration.ivfs_s, dl.cls_len['te'], dl.nside, lmax_qlm=iteration.lmax_qlm)

                dl.qcls_ds = qecl.library(opj(dl.TEMP, 'qcls_ds'), dl.qlms_ds, dl.qlms_ds, np.array([]))  # for QE RDN0 calculations
                dl.qcls_ss = qecl.library(opj(dl.TEMP, 'qcls_ss'), dl.qlms_ss, dl.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
                dl.qcls_dd = qecl.library(opj(dl.TEMP, 'qcls_dd'), dl.qlms_dd, dl.qlms_dd, dl.mc_sims_bias)


            if iteration.FILTER_QE == 'sepTP':
                # ---- QE libraries from plancklens to calculate unnormalized QE (qlms)
                dl.mc_sims_bias = np.arange(60, dtype=int)
                dl.mc_sims_var  = np.arange(60, 300, dtype=int)
                dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], dl.nside, lmax_qlm=iteration.lmax_qlm)
                

        @log_on_start(logging.INFO, "Start of _process_geometryparams()")
        @log_on_end(logging.INFO, "Finished _process_geometryparams()")
        def _process_geometryparams(dl, geometry):
            # TODO this is quite a hacky way for extracting zbounds independent of data object.. simplify..
            _ui = geometry.zbounds[0].split('/')
            _sims_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            sims_loc = getattr(_sims_module, _sims_class_name)('00', mask_suffix=cf.data.mask_suffix)
            zbounds_loc = sims_loc.get_zbounds(hp.read_map(sims_loc.get_mask_path()), geometry.zbounds[1])
            if geometry.zbounds_len[0] ==  geometry.zbounds[0]:
                zbounds_len_loc = sims_loc.extend_zbounds(zbounds_loc, geometry.zbounds_len[1])

            if geometry.lenjob_geometry == 'thin_gauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(geometry.lmax_unl, 2, zbounds=zbounds_len_loc)
            if geometry.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(geometry.pbounds[0], geometry.pbounds[1]))
            if geometry.ninvjob_geometry == 'healpix_geometry':
                    # ninv MAP geometry. Could be merged with QE, if next comment resolved
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=zbounds_loc)
            if geometry.ninvjob_qe_geometry == 'healpix_geometry_qe':
                # TODO for QE, isOBD only works with zbounds=(-1,1). Perhaps missing ztrunc on qumaps
                # Introduced new geometry for now, until either plancklens supports ztrunc, or ztrunced simlib (not sure if it already does)
                dl.ninvjob_qe_geometry = utils_scarf.Geom.get_healpix_geometry(geometry.nside, zbounds=(-1,1))


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


        dl = DLENSALOT_Concept()
        _process_geometryparams(dl, cf.geometry)
        _process_dataparams(dl, cf.data)
        _process_chaindescparams(dl, cf.chain_descriptor)
        _process_iterationparams(dl, cf.iteration)
        _process_stepperparams(dl, cf.stepper)

        return dl


class p2q_Transformer:
    """Extracts all parameters needed for querying results of D.lensalot
    """
    def build(self, cf):
        pass


class p2d_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def build(self, cf):
        # TODO make this an option for the user. If needed, user can define their own edges via configfile.
        fs_edges = np.arange(2,3000, 20)
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
            dl.itmax = de.ITMAX
            dl.fg = de.fg
 
            _ui = cf.data.sims.split('/')
            _sims_module_name = 'lerepi.config.'+_ui[0]+'.data.data_'+_ui[1]
            _sims_class_name = _ui[-1]
            _sims_module = importlib.import_module(_sims_module_name)
            dl.sims = getattr(_sims_module, _sims_class_name)(dl.fg, mask_suffix=dl.mask_suffix) # TODO this should be a *kwargs

            maskpath = dl.sims.get_mask_path()
            dl.base_mask = np.nan_to_num(hp.read_map(maskpath))
            dl.TEMP = transform(cf, p2T_Transformer())
            dl.analysis_path = dl.TEMP.split('/')[-1]
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


class p2i_Transformer:
    """Directory is built upon runtime, so accessing it here

    Returns:
        _type_: _description_
    """
    @log_on_start(logging.INFO, "Start of build()")
    @log_on_end(logging.INFO, "Finished build()")
    def buil(cf):
        pass


class p2j_Transformer:
    """Extracts all parameters needed for D.lensalot for QE and MAP delensing
    Implement if needed
    """
    def build(self, pf):
        jobs = []
        # TODO if the pf.X objects were distinguishable by X2X_Transformer, could replace the seemingly redundant if checks here.
        if pf.job.QE_lensrec:
            jobs.append(((pf, p2lensrec_Transformer()), lenscarf_handler.QE_lr))
        if pf.job.MAP_lensrec:
            jobs.append(((pf, p2lensrec_Transformer()), lenscarf_handler.MAP_lr))
        if pf.job.Btemplate_per_iteration:
            jobs.append(((pf, p2lensrec_Transformer()), lenscarf_handler.B_template_construction))
        if pf.job.map_delensing:
            jobs.append(((pf, p2d_Transformer()), lenscarf_handler.map_delensing))
        if pf.job.inspect_result:
            # TODO maybe use this to return something interactive? Like a webservice with all plots dynamic? Like a dashboard..
            jobs.append(((pf, p2i_Transformer()), lenscarf_handler.inspect_result))
        return jobs


@transform.case(DLENSALOT_Model, p2j_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2T_Transformer)
def f2(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2lensrec_Transformer)
def f3(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2d_Transformer)
def f5(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2i_Transformer)
def f6(expr, transformer): # pylint: disable=missing-function-docstring
    assert 0, "Implement if needed"
    return transformer.build(expr)

@transform.case(DLENSALOT_Model, p2q_Transformer)
def f7(expr, transformer): # pylint: disable=missing-function-docstring
    # TODO this could be a solution to connect to a future 'query' module. Transform into query language, and query.
    # But I am not entirely convinced it is the right way to use the same config file to define query specification.
    # Maybe if the configfile is the one copied to the TEMP dir it is ok..
    assert 0, "Implement if needed"
    return transformer.build(expr)