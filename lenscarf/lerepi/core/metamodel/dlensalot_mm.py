#!/usr/bin/env python

"""dlensalot_mm.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr
import os
from os.path import join as opj

import numpy as np
from lenscarf.lerepi.core.validator import analysis, chaindescriptor, computing, data, filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper
from attrs import validators

class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


    def __str__(self):
        ## overwrite print to summarize dlensalot model
        _str = ''
        for key, val in self.__dict__.items():
            keylen = len(str(key))
            if type(val) in [list, np.ndarray, np.array, dict]:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(type(val))
            else:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(val)
            _str += '\n'
        return _str

@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default=0, validator=chaindescriptor.p0)
    p1 = attr.ib(default=["diag_cl"], validator=chaindescriptor.p1)
    p2 = attr.ib(default=None, validator=chaindescriptor.p2)
    p3 = attr.ib(default=2048, validator=chaindescriptor.p3)
    p4 = attr.ib(default=np.inf, validator=chaindescriptor.p4)
    p5 = attr.ib(default=None, validator=chaindescriptor.p5)
    p6 = attr.ib(default='tr_cg', validator=chaindescriptor.p6)
    p7 = attr.ib(default='cache_mem', validator=chaindescriptor.p7)

@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default='harmonicbump', validator=stepper.typ)
    lmax_qlm = attr.ib(default=4000, validator=stepper.lmax_qlm)
    mmax_qlm = attr.ib(default=4000, validator=stepper.mmax_qlm)
    xa = attr.ib(default=400, validator=stepper.xa)
    xb = attr.ib(default=1500, validator=stepper.xb)

@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    jobs = attr.ib(default=['inspect_result'], validator=job.jobs)

@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    key = attr.ib(default='p_p', validator=[validators.instance_of(str), analysis.key], type=str)
    version = attr.ib(default='', validator=[validators.instance_of(str), analysis.version], type=str)
    simidxs = attr.ib(default=[], validator=data.simidxs)
    simidxs_mf = attr.ib(default=[], validator=analysis.simidxs_mf)
    TEMP_suffix = attr.ib(default='', validator=analysis.TEMP_suffix)
    Lmin = attr.ib(default=1, validator=analysis.Lmin)
    zbounds = attr.ib(default=(-1,1), validator=analysis.zbounds)
    zbounds_len = attr.ib(default=(-1,1), validator=analysis.zbounds_len)
    pbounds = attr.ib(default=(0., 2*np.pi), validator=analysis.pbounds)
    lm_max_len = attr.ib(default=(10,10), validator=filter.lm_max_len)
    lm_max_ivf = attr.ib(default=(10,10), validator=filter.lm_ivf)
    mask = attr.ib(default=None, validator=noisemodel.mask)

@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """

    class_parameters = attr.ib(default={}, validator=data.class_parameters)
    package_ = attr.ib(default=None, validator=data.package_)
    module_ = attr.ib(default=None, validator=data.module_)
    class_ = attr.ib(default=None, validator=data.class_)
    transferfunction = attr.ib(default='gauss_with_pixwin', validator=data.transferfunction)
    data_type = attr.ib(default='map', validator=data.data_type)
    data_field = attr.ib(default='qu', validator=data.data_field)
    beam = attr.ib(default=None)
    nside = attr.ib(default=None)
    nlev_t = attr.ib(default=None)
    nlev_p = attr.ib(default=None)
    transf_dat = attr.ib(default=None)
    lmax_transf = attr.ib(default=None)
    
@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    sky_coverage = attr.ib(default='isotropic', validator=noisemodel.sky_coverage)
    spectrum_type = attr.ib(default='white', validator=noisemodel.spectrum_type)
    OBD = attr.ib(default=False, validator=noisemodel.OBD)
    lmin_teb = attr.ib(default=(10,10,10), validator=noisemodel.lmin_teb)
    nlev_t = attr.ib(default=[], validator=noisemodel.nlev_t)
    nlev_p = attr.ib(default=[], validator=noisemodel.nlev_p)
    rhits_normalised = attr.ib(default=None, validator=noisemodel.rhits_normalised)
    ninvjob_geometry = attr.ib(default='healpix_geometry', validator=noisemodel.ninvjob_geometry)

@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    tasks = attr.ib(default=['calc_phi'], validator=qerec.tasks)
    qlm_type = attr.ib(default='sepTP', validator=qerec.qlms)
    cg_tol = attr.ib(default=1e-2, validator=qerec.cg_tol)
    filter_directional = attr.ib(default=np.nan, validator=qerec.filter_directional)
    ninvjob_qe_geometry = attr.ib(default='healpix_geometry_qe', validator=qerec.ninvjob_qe_geometry)
    lm_max_qlm = attr.ib(default=(10,10), validator=qerec.lm_max_qlm)
    chain = attr.ib(default=DLENSALOT_Chaindescriptor(), validator=qerec.chain)
    cl_analysis = attr.ib(default=False, validator=qerec.cl_analysis)
    blt_pert = attr.ib(default=True, validator=qerec.btemplate_perturbative_lensremap)

@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    tasks = attr.ib(default=['calc_phi'], validator=itrec.tasks)
    itmax = attr.ib(default=1, validator=itrec.itmax)
    cg_tol = attr.ib(default=np.nan, validator=itrec.cg_tol)
    iterator_typ = attr.ib(default='constmf', validator=itrec.iterator_type)
    lensres = attr.ib(default=1.7, validator=itrec.lensres)
    filter_directional = attr.ib(default=np.nan, validator=itrec.filter_directional)
    lenjob_geometry = attr.ib(default='thin_gauss', validator=itrec.lenjob_geometry)
    lenjob_pbgeometry = attr.ib(default='pbdGeometry', validator=itrec.lenjob_pbgeometry)
    lm_max_unl = attr.ib(default=(10,10), validator=itrec.lm_max_unl)
    lm_max_qlm = attr.ib(default=(10,10), validator=itrec.lm_max_qlm)
    mfvar = attr.ib(default='', validator=itrec.mfvar)
    soltn_cond = attr.ib(default=lambda it: True, validator=itrec.soltn_cond)
    stepper = attr.ib(default=DLENSALOT_Stepper(), validator=itrec.stepper)
    
@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges = attr.ib(default=-1, validator=mapdelensing.edges)
    dlm_mod = attr.ib(default=False, validator=mapdelensing.dlm_mod)
    iterations = attr.ib(default=-1, validator=mapdelensing.iterations)
    masks = attr.ib(default=None, validator=mapdelensing.masks)
    lmax = attr.ib(default=-1, validator=mapdelensing.lmax)
    Cl_fid = attr.ib(default=-1, validator=mapdelensing.Cl_fid)
    libdir_it = attr.ib(default=None, validator=mapdelensing.libdir_it)
    binning = attr.ib(default=-1, validator=mapdelensing.binning)
    spectrum_calculator = attr.ib(default=None, validator=mapdelensing.spectrum_calculator)

@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        BMARG_LIBDIR:
    """
    libdir = attr.ib(default='', validator=obd.libdir)
    rescale = attr.ib(default=1, validator=obd.rescale)
    tpl = attr.ib(default='template_dense', validator=obd.tpl)
    nlev_dep = attr.ib(default=np.nan, validator=obd.nlev_dep)
    nside = attr.ib(default=np.nan)
    lmax = attr.ib(default=np.nan)
    beam = attr.ib(default=np.nan)

@attr.s
class DLENSALOT_Config(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    outdir_plot_root = attr.ib(default=opj(os.environ['HOME'], 'plots'))
    outdir_plot_rel = attr.ib(default='')

@attr.s
class DLENSALOT_Meta(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        version:
    """
    version = attr.ib(default='', validator=attr.validators.instance_of(str))

@attr.s
class DLENSALOT_Computing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    OMP_NUM_THREADS = attr.ib(default=-1, validator=computing.OMP_NUM_THREADS)

@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    meta = attr.ib(default=DLENSALOT_Meta(), validator=model.meta)
    job = attr.ib(default=DLENSALOT_Job(), validator=model.job)
    analysis = attr.ib(default=DLENSALOT_Analysis(), validator=model.analysis)
    data  = attr.ib(default=DLENSALOT_Data(), validator=model.data)
    noisemodel = attr.ib(default=DLENSALOT_Noisemodel(), validator=model.noisemodel)
    qerec = attr.ib(default=DLENSALOT_Qerec(), validator=model.qerec)
    itrec = attr.ib(default=DLENSALOT_Itrec(), validator=model.itrec)
    madel = attr.ib(default=DLENSALOT_Mapdelensing(), validator=model.madel)
    config = attr.ib(default=DLENSALOT_Config(), validator=model.config)
    computing = attr.ib(default=DLENSALOT_Config(), validator=model.computing)
    obd = attr.ib(default=DLENSALOT_OBD(), validator=model.obd)