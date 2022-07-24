#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr

import numpy as np
from lenscarf.lerepi.core.metamodel.validator import analysis, chaindescriptor, computing, data, filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper


class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


    def __str__(self):
        _str = ''
        for k, v in self.__dict__.items():
            _str+="\t{}:\t{}\n".format(k,v)
        return _str


@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        BMARG_LIBDIR:
    """
    libdir = attr.ib(default=None, validator=obd.libdir)
    rescale = attr.ib(default=None, validator=obd.rescale)
    tpl = attr.ib(default=None, validator=obd.tpl)
    nlev_dep = attr.ib(default=np.nan, validator=obd.nlev_dep)


<<<<<<< HEAD
=======

>>>>>>> 5c89a4d (refactor validator)
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
    TEMP_suffix = attr.ib(default=None, validator=analysis.TEMP_suffix)
    lens_res = attr.ib(default=np.nan, validator=analysis.lens_res)
    zbounds =  attr.ib(default=(-1,1), validator=analysis.zbounds)
    zbounds_len = attr.ib(default=(-1,1), validator=analysis.zbounds_len)
    pbounds = attr.ib(default=(-1,1), validator=analysis.pbounds)


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 76828a1 (refactor)
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
=======
    TEMP_suffix = attr.ib(default=-1)
    data_type = attr.ib(default=None)
    data_field = attr.ib(default=None)
    fg = attr.ib(default=-1)
    sims = attr.ib(default=-1)
    nside = attr.ib(default=-1)
    beam = attr.ib(default=-1)
    lmax_transf = attr.ib(default=-1)
    transf = attr.ib(default=-1)
    tpl = attr.ib(default=-1)


@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default=-1)
    p1 = attr.ib(default=-1)
    p2 = attr.ib(default=-1)
    p3 = attr.ib(default=-1)
    p4 = attr.ib(default=-1)
    p5 = attr.ib(default=-1)
    p6 = attr.ib(default=-1)
    p7 = attr.ib(default=-1)


@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=-1)
    lmax_qlm = attr.ib(default=-1)
    mmax_qlm = attr.ib(default=-1)
    xa = attr.ib(default=-1)
    xb = attr.ib(default=-1)


@attr.s
class DLENSALOT_Iteration(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    K = attr.ib(default=-1)
    V = attr.ib(default=-1)
    QE_subtract_meanfield = attr.ib(default=True)
    tasks = attr.ib(default=-1)
    ITMAX = attr.ib(default=-1)
    IMIN = attr.ib(default=-1)
    IMAX = attr.ib(default=-1)
    mfvar = attr.ib(default=-1)
    ivfs = attr.ib(default=None)
    qlms = attr.ib(default=None)
    QE_LENSING_CL_ANALYSIS = attr.ib(default=-1)
    STANDARD_TRANSFERFUNCTION = attr.ib(default=-1)
    filter = attr.ib(default=-1)
    CHAIN_DESCRIPTOR = attr.ib(default=-1)
    FILTER_QE = attr.ib(default=-1)
    iterator_typ = attr.ib(default=-1)
    lmax_filt = attr.ib(default=-1)
    lmax_qlm = attr.ib(default=-1)
    mmax_qlm = attr.ib(default=-1)
    lmax_unl = attr.ib(default=-1)
    mmax_unl = attr.ib(default=-1)
    lmax_ivf = attr.ib(default=-1)
    mmax_ivf = attr.ib(default=-1)
    lmin_ivf = attr.ib(default=-1)
    mmin_ivf = attr.ib(default=-1)
    LENSRES = attr.ib(default=-1) 
    Lmin = attr.ib(default=-1)
    cg_tol = attr.ib(default=-1)
    TOL = attr.ib(default=-1)
    soltn_cond = attr.ib(default=-1)
    nsims_mf = attr.ib(default=-1)
    OMP_NUM_THREADS = attr.ib(default=-1)


@attr.s
class DLENSALOT_Geometry(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    lmax_unl = attr.ib(default=-1)
    zbounds = attr.ib(default=-1)
    zbounds_len = attr.ib(default=-1)
    pbounds = attr.ib(default=-1)
    nside = attr.ib(default=-1)
    lenjob_geometry = attr.ib(default=-1)
    lenjob_pbgeometry = attr.ib(default=-1)
    ninvjob_geometry = attr.ib(default=-1)
    ninvjob_qe_geometry = attr.ib(default=-1)


@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges = attr.ib(default=-1)
    dlm_mod = attr.ib(default=False)
    iterations = attr.ib(default=-1)
    masks = attr.ib(default=None)
    lmax = attr.ib(default=-1)
    Cl_fid = attr.ib(default=-1)
    libdir_it = attr.ib(default=None)
    binning = attr.ib(default=-1)
    spectrum_calculator = attr.ib(default=None)
    data_from_CFS = attr.ib(default=True)
>>>>>>> 3b8b5ed (minor update)


@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    lowell_treat = attr.ib(default=None, validator=noisemodel.lowell_treat)
    OBD = attr.ib(default=None, validator=noisemodel.OBD, type=[DLENSALOT_OBD, type(None)])
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
    Lmin = attr.ib(default=np.nan, validator=qerec.Lmin)
    filter = attr.ib(default=None, validator=qerec.filter)
    qest = attr.ib(default=None, validator=qerec.qest)
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
    simidxs_mf = attr.ib(default=[], validator=itrec.simidxs_mf)
    itmax = attr.ib(default=np.nan, validator=itrec.itmax)
<<<<<<< HEAD
<<<<<<< HEAD
=======
    lmax_filter = attr.ib(default=np.nan, validator=itrec.lmax_filter)
    mmax_filter = attr.ib(default=np.nan, validator=itrec.mmax_filter)
>>>>>>> 5c89a4d (refactor validator)
=======
>>>>>>> 76828a1 (refactor)
    lmax_plm = attr.ib(default=np.nan, validator=itrec.lmax_plm)
    mmax_plm = attr.ib(default=np.nan, validator=itrec.mmax_plm)
    filter = attr.ib(default=None, validator=itrec.filter)
    cg_tol = attr.ib(default=np.nan, validator=itrec.cg_tol)
    lenjob_geometry = attr.ib(default=None, validator=itrec.lenjob_geometry)
    lenjob_pbgeometry = attr.ib(default=None, validator=itrec.lenjob_pbgeometry)
    iterator_typ = attr.ib(default=None, validator=itrec.iterator_typ)
    mfvar = attr.ib(default=None, validator=itrec.mfvar)
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
    binning = attr.ib(default=-1, validator=mapdelensing.binning)
    dir_btempl = attr.ib(default='intern', validator=mapdelensing.dir_btempl)
    spectrum_calculator = attr.ib(default=None, validator=mapdelensing.spectrum_calculator)


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    meta = attr.ib(default=-1, validator=model.meta, type=DLENSALOT_Meta)
    computing = attr.ib(default=-1, validator=model.computing, type=DLENSALOT_Computing)
    job = attr.ib(default=-1, validator=model.job, type=DLENSALOT_Job)
    analysis = attr.ib(default=-1, validator=model.analysis, type=DLENSALOT_Analysis)
    data  = attr.ib(default=[], validator=model.data, type=DLENSALOT_Data)
    noisemodel = attr.ib(default=[], validator=model.noisemodel, type=DLENSALOT_Noisemodel)
    qerec = attr.ib(default=[], validator=model.qerec, type=DLENSALOT_Qerec)
    itrec = attr.ib(default=-1, validator=model.itrec, type=DLENSALOT_Itrec)
    madel = attr.ib(default=-1, validator=model.madel, type=DLENSALOT_Mapdelensing)


@attr.s
class DLENSALOT_Filter(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    directional = attr.ib(default=np.nan, validator=filter.directional)
    data_type = attr.ib(default=np.nan, validator=filter.data_type)
<<<<<<< HEAD
<<<<<<< HEAD
    lmax = attr.ib(default=np.nan, validator=filter.lmax)
    mmax = attr.ib(default=np.nan, validator=filter.mmax)
=======
>>>>>>> 5c89a4d (refactor validator)
=======
    lmax = attr.ib(default=np.nan, validator=filter.lmax)
    mmax = attr.ib(default=np.nan, validator=filter.mmax)
>>>>>>> 76828a1 (refactor)
    lmax_len = attr.ib(default=np.nan, validator=filter.lmax_len)
    mmax_len = attr.ib(default=np.nan, validator=filter.mmax_len)
    lmax_unl = attr.ib(default=np.nan, validator=filter.lmax_unl)
    mmax_unl = attr.ib(default=np.nan, validator=filter.mmax_unl)


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
