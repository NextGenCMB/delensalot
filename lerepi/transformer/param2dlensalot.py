#!/usr/bin/env python

"""param2dlensalot.py: transformer module to build dlensalot model from parameter file
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj

import numpy as np
import healpy as hp

import plancklens
from plancklens import qest, qecl
from plancklens import utils, qest, qecl
from plancklens.filt import filt_util
from plancklens.qcinv import cd_solve
from plancklens.filt import filt_cinv, filt_util

from lenscarf import utils_scarf
from lenscarf.utils import cli
from lenscarf.iterators import steps
from lenscarf.utils_hp import gauss_beam
from lenscarf.opfilt.bmodes_ninv import template_dense

from lerepi.metamodel.dlensalot import Dlensalot_Model

from lerepi.core.visitor import transform
from lerepi.core.delensing_interface import Dlensalot


class p2d_Transformer:
    """Extracts all parameters needed for Dlensalot and turns it into a dl._model
    """


    def build(cf):

        def _process_dataparams(dl, data):
            dl.mask_suffix = data.mask_suffix
            dl.nside = cf.nside
            dl.isOBD = data.isOBD
            dl.nsims_mf = 0 if data.V == 'noMF' else data.nsims_mf
            dl.mc_sims_mf_it0 = np.arange(dl.nsims_mf)
            dl.rhits = hp.read_map(data.rhits)
            dl.fg = data.fg
            dl.sims = data.sims(data.fg, mask_suffix=data.mask_suffix)
            dl.mask = data.mask.get_mask_path()
            dl.masks = [mask.get_mask_path() for mask in data.masks]

            dl.beam = data.BEAM
            dl.lmax_transf = data.lmax_transf
            dl.transf = data.transf(dl.beam / 180. / 60. * np.pi, lmax=dl.lmax_transf)

            _suffix = '08d_%s_r%s'%(data.fg, data.mask_suffix)+'_isOBD'*data.isOBD
            _suffix += '_MF%s'%(data.nsims_mf) if data.nsims_mf > 0 else ''
            dl.TEMP =  opj(os.environ['SCRATCH'], 'cmbs4', _suffix)

            dl.zbounds = data.zbounds[0].get_zbounds(hp.read_map(dl.mask),data.zbounds[1])
            dl.zbounds_len = data.zbounds_len[0].extend_zbounds(dl.zbounds, data.zbounds_len[1])
            dl.pb_ctr, dl.pb_extent = data.pbounds

            dl.DATA_libdir = data.DATA_LIBDIR
            dl.BMARG_LIBDIR = data.BMARG_LIBDIR
            dl.BMARG_LCUT = data.BMARG_LCUT

            dl.CENTRALNLEV_UKAMIN = data.CENTRALNLEV_UKAMIN
            dl.nlev_t = data.CENTRALNLEV_UKAMIN if data.nlev_t == None else data.nlev_t
            dl.nlev_p = data.CENTRALNLEV_UKAMIN/np.sqrt(2) if data.nlev_p == None else data.nlev_t

            if cf.isOBD:
                if data.tpl == 'template_dense':
                    dl.tpl = template_dense(data.BMARG_LCUT, dl.ninvjob_geometry, data.tr, _lib_dir=data.BMARG_LIBDIR)
                else:
                    assert 0, "Implement if needed"
            else:
                dl.tpl = None


            cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
            dl.cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            dl.cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))


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

            dl.tol = cf.TOL
            dl.tol_iter = lambda it : 10 ** (- dl.tol)
            dl.soltn_cond = cf.soltn_cond # Uses (or not) previous E-mode solution as input to search for current iteration one
            dl.cg_tol = cf.CG_TOL

            dl.cpp = np.copy(dl.cls_unl['pp'][:dl.lmax_qlm + 1])
            dl.cpp[:cf.Lmin] *= 0.


            dl.lensres = iteration.LENSRES
            dl.tr = int(os.environ.get('OMP_NUM_THREADS', iteration.OMP_NUM_THREADS)) #TODO hardcoded. what to do with it?
            dl.iterator = iteration.ITERATOR

            dl.stepper = iteration.stepper


            if iteration.STANDARD_TRANSFERFUNCTION == True:
                # Fiducial model of the transfer function
                dl.transf_tlm   =  gauss_beam(iteration.BEAM/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_tlm)
                dl.transf_elm   =  gauss_beam(iteration.BEAM/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_elm)
                dl.transf_blm   =  gauss_beam(iteration.BEAM/180 / 60 * np.pi, lmax=iteration.lmax_ivf) * (np.arange(iteration.lmax_ivf + 1) >= iteration.lmin_blm)

                # Isotropic approximation to the filtering (used eg for response calculations)
                dl.ftl =  cli(dl.cls_len['tt'][:iteration.lmax_ivf + 1] + (iteration.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel =  cli(dl.cls_len['ee'][:iteration.lmax_ivf + 1] + (iteration.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl =  cli(dl.cls_len['bb'][:iteration.lmax_ivf + 1] + (iteration.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)

                # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
                dl.ftl_unl =  cli(dl.cls_unl['tt'][:iteration.lmax_ivf + 1] + (iteration.nlev_t / 180 / 60 * np.pi) ** 2 * cli(dl.transf_tlm ** 2)) * (dl.transf_tlm > 0)
                dl.fel_unl =  cli(dl.cls_unl['ee'][:iteration.lmax_ivf + 1] + (iteration.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_elm ** 2)) * (dl.transf_elm > 0)
                dl.fbl_unl =  cli(dl.cls_unl['bb'][:iteration.lmax_ivf + 1] + (iteration.nlev_p / 180 / 60 * np.pi) ** 2 * cli(dl.transf_blm ** 2)) * (dl.transf_blm > 0)


            if iteration.FILTER == 'cinv_sepTP':
                dl.ninv_t = [np.array([hp.nside2pixarea(iteration.nside, degrees=True) * 60 ** 2 / iteration.nlev_t ** 2])] + iteration.masks
                dl.ninv_p = [[np.array([hp.nside2pixarea(iteration.nside, degrees=True) * 60 ** 2 / iteration.nlev_p ** 2])] + iteration.masks]

                dl.cinv_t = filt_cinv.cinv_t(opj(dl.TEMP, 'cinv_t'), iteration.lmax_ivf,iteration.nside, dl.cls_len, dl.transf_tlm, dl.ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])

                dl.cinv_p = filt_cinv.cinv_p(opj(dl.TEMP, 'cinv_p'), dl.lmax_ivf, iteration.nside, dl.cls_len, dl.transf_elm, dl.ninv_p,
                            chain_descr=dl.chain_descr(iteration.lmax_ivf, iteration.CG_TOL), transf_blm=dl.transf_blm, marge_qmaps=(), marge_umaps=())

                dl.ivfs_raw = filt_cinv.library_cinv_sepTP(opj(dl.TEMP, 'ivfs'), iteration.sims, dl.cinv_t, dl.cinv_p, dl.cls_len)
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

                dl.qlms_ds = qest.library_sepTP(opj(dl.TEMP, 'qlms_ds'), iteration.ivfs, iteration.ivfs_d, dl.cls_len['te'], iteration.nside, lmax_qlm=iteration.lmax_qlm)
                dl.qlms_ss = qest.library_sepTP(opj(dl.TEMP, 'qlms_ss'), iteration.ivfs, iteration.ivfs_s, dl.cls_len['te'], iteration.nside, lmax_qlm=iteration.lmax_qlm)

                dl.qcls_ds = qecl.library(opj(dl.TEMP, 'qcls_ds'), dl.qlms_ds, dl.qlms_ds, np.array([]))  # for QE RDN0 calculations
                dl.qcls_ss = qecl.library(opj(dl.TEMP, 'qcls_ss'), dl.qlms_ss, dl.qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations


            if iteration.FILTER_QE == 'sepTP':
                # ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
                dl.mc_sims_bias = np.arange(60, dtype=int)
                dl.mc_sims_var  = np.arange(60, 300, dtype=int)
                dl.qlms_dd = qest.library_sepTP(opj(dl.TEMP, 'qlms_dd'), dl.ivfs, dl.ivfs, dl.cls_len['te'], iteration.nside, lmax_qlm=iteration.lmax_qlm)
                dl.qcls_dd = qecl.library(opj(dl.TEMP, 'qcls_dd'), dl.qlms_dd, dl.qlms_dd, dl.mc_sims_bias)


        def _process_geometryparams(dl, geometry):
            if geometry.lenjob_geometry == 'thingauss':
                dl.lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(geometry.lmax_unl, 2, zbounds=dl.zbounds_len)
            if geometry.lenjob_pbgeometry == 'pbdGeometry':
                dl.lenjob_pbgeometry = utils_scarf.pbdGeometry(dl.lenjob_geometry, utils_scarf.pbounds(dl.pb_ctr, dl.pb_extent))
            if geometry.ninvjob_geometry == 'healpix_geometry':
                dl.ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(dl.nside, zbounds=dl.zbounds)
             

        def _process_chaindescparams(dl, cd):
            # TODO hacky solution. Redo if needed
            dl.chain_descr = lambda p2, p5 : [
                [cd.p0, cd.p1, p2, cd.p3, cd.p4, p5, cd.p6, cd.p7]]


        def _process_stepperparams(dl, st):
            if st.stepper == 'harmonicbump':
                dl.stepper = steps.harmonicbump(st.lmax_qlm, st.mmax_qlm, xa=400, xb=1500)


        dl = Dlensalot()
        _process_geometryparams(dl, cf.geometry)
        _process_dataparams(dl, cf.data)
        _process_iterationparams(dl, cf.iteration)
        _process_chaindescparams(dl, cf.chain_descriptor)
        _process_stepperparams(dl, cf.stepper)


class p2l_Transformer:
    """Extracts all parameters needed for lerepi
    Implement if needed
    """
    def build(cf):
        pass


@transform.case(Dlensalot_Model, p2d_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)


@transform.case(Dlensalot_Model, p2l_Transformer)
def f1(expr, transformer): # pylint: disable=missing-function-docstring
    return transformer.build(expr)