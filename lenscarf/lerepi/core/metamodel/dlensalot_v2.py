#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr

import numpy as np


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
    p0 = attr.ib(default=0)
    p1 = attr.ib(default=["diag_cl"])
    p2 = attr.ib(default=None)
    p3 = attr.ib(default=2048)
    p4 = attr.ib(default=np.inf)
    p5 = attr.ib(default=None)
    p6 = attr.ib(default='tr_cg')
    p7 = attr.ib(default='cache_mem')

@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default='harmonicbump')
    lmax_qlm = attr.ib(default=4000)
    mmax_qlm = attr.ib(default=4000)
    xa = attr.ib(default=400)
    xb = attr.ib(default=1500)

    
@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_lensrec:
    """
    QE_lensrec = attr.ib(default=False)
    MAP_lensrec = attr.ib(default=False)
    inspect_result = attr.ib(default=False)
    map_delensing = attr.ib(default=False)
    build_OBD = attr.ib(default=False)
    OMP_NUM_THREADS = attr.ib(default=False)


@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    TEMP_suffix = attr.ib(default=None)

    K = attr.ib(default=np.nan)
    V = attr.ib(default='')
    ITMAX = attr.ib(default=np.nan)
    simidxs_mf = attr.ib(default=[])
    LENSRES = attr.ib(default=np.nan)
    Lmin = attr.ib(default=np.nan)
    lmax_filt = attr.ib(default=np.nan)
    lmax_unl = attr.ib(default=np.nan)
    mmax_unl = attr.ib(default=np.nan)
    lmax_ivf = attr.ib(default=np.nan)
    mmax_ivf = attr.ib(default=np.nan)
    lmin_ivf = attr.ib(default=np.nan)
    mmin_ivf = attr.ib(default=np.nan)
    lmax_unl = attr.ib(default=np.nan)
    zbounds =  attr.ib(default=(-1,1))
    zbounds_len = attr.ib(default=(-1,1))
    pbounds = attr.ib(default=(-1,1))
    STANDARD_TRANSFERFUNCTION = attr.ib(default=True)


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    IMIN = attr.ib(default=np.nan)
    IMAX = attr.ib(default=np.nan)
    
    simidxs = attr.ib(default=[])
    class_parameters = attr.ib(default=None)
    package_ = attr.ib(default=None)
    module_ = attr.ib(default=None)
    class_ = attr.ib(default=None)
    root_path_ = attr.ib(default=None)
    data_type = attr.ib(default=None)
    data_field = attr.ib(default=None)
    beam = attr.ib(default=None)
    lmax_transf = attr.ib(default=np.nan)
    nside = attr.ib(default=np.nan)


@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=None)
    BMARG_LIBDIR = attr.ib(default=None)
    BMARG_LCUT = attr.ib(default=None)
    BMARG_RESCALE = attr.ib(default=None)
    ninvjob_geometry = attr.ib(default='healpix_geometry')
    lmin_tlm = attr.ib(default=np.nan)
    lmin_elm = attr.ib(default=np.nan)
    lmin_blm = attr.ib(default=np.nan)
    CENTRALNLEV_UKAMIN = attr.ib(default=np.nan)
    nlev_t = attr.ib(default=[])
    nlev_p = attr.ib(default=[])
    nlev_dep = attr.ib(default=np.nan)
    inf = attr.ib(default=np.nan)
    mask = attr.ib(default=None)
    rhits_normalised = attr.ib(default=None)
    tpl = attr.ib(default=None)


@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    ivfs = attr.ib(default=None)
    qlms = attr.ib(default=None)
    cg_tol = attr.ib(default=np.nan)
    ninvjob_qe_geometry = attr.ib(default='healpix_geometry_qe')
    lmax_qlm = attr.ib(default=np.nan)
    mmax_qlm = attr.ib(default=np.nan)
    chain = attr.ib(default=DLENSALOT_Chaindescriptor())
    tasks = attr.ib(default=None)
    QE_LENSING_CL_ANALYSIS = attr.ib(default=False)
    overwrite_libdir = attr.ib(default=None)


@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    filter = attr.ib(default=None)
    cg_tol = attr.ib(default=np.nan)
    lenjob_geometry = attr.ib(default='thin_gauss')
    lenjob_pbgeometry = attr.ib(default='pbdGeometry')
    iterator_typ = attr.ib(default=None)
    mfvar = attr.ib(default=None)
    soltn_cond = attr.ib(default=lambda it: True)
    stepper = attr.ib(default=DLENSALOT_Stepper())
    overwrite_itdir = attr.ib(default=None)
    tasks = attr.ib(default=None)
    btemplate_perturbative_lensremap = attr.ib(default=False)


@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges = attr.ib(default=-1)
    dlm_mod = attr.ib(default=[False,''])
    simidxs_mblt = attr.ib(default=[])
    subtract_mblt = attr.ib(default=[None, None])
    iterations = attr.ib(default=-1)
    masks = attr.ib(default=None)
    ringmask = attr.ib(default=False)
    lmax = attr.ib(default=-1)
    Cl_fid = attr.ib(default=-1)
    libdir_it = attr.ib(default=None)
    binning = attr.ib(default=-1)
    spectrum_calculator = attr.ib(default=None)
    data_from_CFS = attr.ib(default=False)
    btemplate_perturbative_lensremap = attr.ib(default=False)


@attr.s
class DLENSALOT_Config(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    outdir_plot_root = attr.ib(default=None)
    outdir_plot_rel = attr.ib(default=None)



@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    job = attr.ib(default=DLENSALOT_Job())
    analysis = attr.ib(default=DLENSALOT_Analysis())
    data  = attr.ib(default=DLENSALOT_Data())
    noisemodel = attr.ib(default=DLENSALOT_Noisemodel())
    qerec = attr.ib(default=DLENSALOT_Qerec())
    itrec = attr.ib(default=DLENSALOT_Itrec())
    madel = attr.ib(default=DLENSALOT_Mapdelensing())
    config = attr.ib(default=DLENSALOT_Config())
