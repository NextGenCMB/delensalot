#!/usr/bin/env python

"""dlensalot_mm.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr

import numpy as np
from lenscarf.lerepi.core.validator import analysis, chaindescriptor, computing, data, filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper


class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


    def __str__(self):
        _str = ''
        for k, v in self.__dict__.items():
            _str+="\t{}:\t{}\n".format(k,v)
        return _str

@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default=-1, validator=chaindescriptor.p0)
    p1 = attr.ib(default=-1, validator=chaindescriptor.p1)
    p2 = attr.ib(default=-1, validator=chaindescriptor.p2)
    p3 = attr.ib(default=-1, validator=chaindescriptor.p3)
    p4 = attr.ib(default=-1, validator=chaindescriptor.p4)
    p5 = attr.ib(default=-1, validator=chaindescriptor.p5)
    p6 = attr.ib(default=-1, validator=chaindescriptor.p6)
    p7 = attr.ib(default=-1, validator=chaindescriptor.p7)

@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=-1, validator=stepper.typ)
    lmax_qlm = attr.ib(default=-1, validator=stepper.lmax_qlm)
    mmax_qlm = attr.ib(default=-1, validator=stepper.mmax_qlm)
    xa = attr.ib(default=-1, validator=stepper.xa)
    xb = attr.ib(default=-1, validator=stepper.xb)

@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    QE_lensrec = attr.ib(default=-1, validator=job.QE_lensrec)
    MAP_lensrec = attr.ib(default=-1, validator=job.MAP_lensrec)
    inspect_result = attr.ib(default=-1, validator=job.inspect_result)
    map_delensing = attr.ib(default=-1, validator=job.map_delensing)
    build_OBD = attr.ib(default=-1, validator=job.build_OBD)

@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    key = attr.ib(default=np.nan, validator=analysis.key)
    version = attr.ib(default=np.nan, validator=analysis.version)
    simidxs_mf = attr.ib(default=[], validator=analysis.simidxs_mf)
    TEMP_suffix = attr.ib(default='', validator=analysis.TEMP_suffix)
    lens_res = attr.ib(default=1.7, validator=analysis.lens_res)
    Lmin = attr.ib(default=1, validator=analysis.Lmin)
    zbounds = attr.ib(default=(-1,1), validator=analysis.zbounds)
    zbounds_len = attr.ib(default=(-1,1), validator=analysis.zbounds_len)
    pbounds = attr.ib(default=(-1,1), validator=analysis.pbounds)

    lmax_filt = attr.ib(default=np.nan, validator=filter.lmax_filt)
    lm_max_len = attr.ib(default=np.nan, validator=filter.lm_max_len)
    lm_max_unl = attr.ib(default=np.nan, validator=filter.lm_max_unl)
    lm_ivf = attr.ib(default=np.nan, validator=filter.lm_ivf)

    STANDARD_TRANSFERFUNCTION = attr.ib(default=True)

@attr.s
class DLENSALOT_Meta(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        version:
    """
    version = attr.ib(default=-1, validator=attr.validators.instance_of(str))

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
    meta = attr.ib(default=-1, validator=model.meta)
    job = attr.ib(default=-1, validator=model.job)
    analysis = attr.ib(default=-1, validator=model.analysis)
    data  = attr.ib(default=[], validator=model.data)
    noisemodel = attr.ib(default=[], validator=model.noisemodel)
    qerec = attr.ib(default=[], validator=model.qerec)
    itrec = attr.ib(default=-1, validator=model.itrec)
    madel = attr.ib(default=-1, validator=model.madel)

@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    simidxs = attr.ib(default=[], validator=data.simidxs)
    class_parameters = attr.ib(default=None, validator=data.class_parameters)
    package_ = attr.ib(default=None, validator=data.package_)
    module_ = attr.ib(default=None, validator=data.module_)
    class_ = attr.ib(default=None, validator=data.class_)
    data_type = attr.ib(default=None, validator=data.data_type)
    data_field = attr.ib(default=None, validator=data.data_field)
    beam = attr.ib(default=None, validator=data.beam)
    nside = attr.ib(default=np.nan, validator=data.nside)
    transferfunction = attr.ib(default=True, validator=data.transferfunction)
    lmax = attr.ib(default=True, validator=data.transferfunction)

@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    lowell_treat = attr.ib(default=None, validator=noisemodel.lowell_treat)
    OBD = attr.ib(default=None, validator=noisemodel.OBD)
    lmin_tlm = attr.ib(default=np.nan, validator=noisemodel.lmin_tlm)
    lmin_elm = attr.ib(default=np.nan, validator=noisemodel.lmin_elm)
    lmin_blm = attr.ib(default=np.nan, validator=noisemodel.lmin_blm)
    nlev_t = attr.ib(default=[], validator=noisemodel.nlev_t)
    nlev_p = attr.ib(default=[], validator=noisemodel.nlev_p)
    rhits_normalised = attr.ib(default=None, validator=noisemodel.rhits_normalised)
    mask = attr.ib(default=None, validator=noisemodel.mask)
    ninvjob_geometry = attr.ib(default=None, validator=noisemodel.ninvjob_geometry)

@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    simidxs = attr.ib(default=[], validator=qerec.simidxs)
    simidxs_mf = attr.ib(default=[], validator=qerec.simidxs_mf)
    ivfs = attr.ib(default=None, validator=qerec.ivfs)
    qlms = attr.ib(default=None, validator=qerec.qlms)
    cg_tol = attr.ib(default=np.nan, validator=qerec.cg_tol)
    ninvjob_qe_geometry = attr.ib(default=None, validator=qerec.ninvjob_qe_geometry)
    lmax_qlm = attr.ib(default=np.nan, validator=qerec.lmax_qlm)
    mmax_qlm = attr.ib(default=np.nan, validator=qerec.mmax_qlm)
    chain = attr.ib(default=None, validator=qerec.chain)
    cl_analysis = attr.ib(default=False, validator=qerec.cl_analysis)

@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    tasks = attr.ib(default=None, validator=itrec.tasks)
    simidxs = attr.ib(default=[], validator=itrec.simidxs)
    itmax = attr.ib(default=np.nan, validator=itrec.itmax)
    filter = attr.ib(default=None, validator=itrec.filter)
    cg_tol = attr.ib(default=np.nan, validator=itrec.cg_tol)
    lenjob_geometry = attr.ib(default=None, validator=itrec.lenjob_geometry)
    lenjob_pbgeometry = attr.ib(default=None, validator=itrec.lenjob_pbgeometry)
    iterator_typ = attr.ib(default='constmf', validator=itrec.iterator_typ)
    mfvar = attr.ib(default='', validator=itrec.mfvar)
    soltn_cond = attr.ib(default=None, validator=itrec.soltn_cond)
    stepper = attr.ib(default=None, validator=itrec.stepper)  

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

class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        BMARG_LIBDIR:
    """
    libdir = attr.ib(default=None, validator=obd.libdir)
    rescale = attr.ib(default=None, validator=obd.rescale)
    tpl = attr.ib(default=None, validator=obd.tpl)
    nlev_dep = attr.ib(default=np.nan, validator=obd.nlev_dep)