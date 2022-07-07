#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr
from warnings import warn
warn('dlensalot is deprecated and will soon be replaced by dlensalot_v2. Please use dlensalot_v2, so the transition will be easier in the future', DeprecationWarning, stacklevel=2)

class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    job = attr.ib(default=-1)
    data = attr.ib(default=-1)
    iteration  = attr.ib(default=[])
    geometry = attr.ib(default=[])
    chain_descriptor = attr.ib(default=[])
    stepper = attr.ib(default=-1)
    map_delensing = attr.ib(default=-1)
    noisemodel = attr.ib(default=-1)


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    QE_lensrec = attr.ib(default=-1)
    MAP_lensrec = attr.ib(default=-1)
    inspect_result = attr.ib(default=-1)
    map_delensing = attr.ib(default=-1)
    build_OBD = attr.ib(default=-1)


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
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


@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=-1)
    BMARG_LIBDIR = attr.ib(default=-1)
    BMARG_LCUT = attr.ib(default=-1)
    BMARG_RESCALE = attr.ib(default=-1)
    lmin_tlm = attr.ib(default=-1)
    lmin_elm = attr.ib(default=-1)
    lmin_blm = attr.ib(default=-1)
    CENTRALNLEV_UKAMIN = attr.ib(default=-1)
    nlev_dep = attr.ib(default=-1)
    nlev_t = attr.ib(default=-1)
    nlev_p = attr.ib(default=-1)
    inf = attr.ib(default=-1)
    ratio = attr.ib(default=-1)
    mask = attr.ib(default=-1)
    rhits_normalised = attr.ib(default=-1)
