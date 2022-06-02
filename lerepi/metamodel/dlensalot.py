#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import abc
from enum import Enum
import attr


class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    data = attr.ib(default='')
    iteration  = attr.ib(default=[])
    geometry = attr.ib(default=[])
    chain_descriptor = attr.ib(default=[])
    stepper = attr.ib(default='')


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    DATA_LIBDIR = attr.ib(default='')
    rhits = attr.ib(default='')
    fg = attr.ib(default='')
    mask_suffix = attr.ib(default='')
    sims = attr.ib(default='')
    mask = attr.ib(default='')
    masks = attr.ib(default=[])
    nside = attr.ib(default='')
    BEAM = attr.ib(default='')
    lmax_transf = attr.ib(default='')
    transf = attr.ib(default='')
    zbounds = attr.ib(default='')
    zbounds_len = attr.ib(default='')
    pbounds = attr.ib(default='')
    isOBD = attr.ib(default='')
    BMARG_LIBDIR = attr.ib(default='')
    BMARG_LCUT = attr.ib(default='')
    tpl = attr.ib(default='')
    CENTRALNLEV_UKAMIN = attr.ib(default='')
    nlev_t = attr.ib(default='')
    nlev_p = attr.ib(default='')


@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default='')
    p1 = attr.ib(default='')
    p2 = attr.ib(default='')
    p3 = attr.ib(default='')
    p4 = attr.ib(default='')
    p5 = attr.ib(default='')
    p6 = attr.ib(default='')
    p7 = attr.ib(default='')


@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default='')
    lmax_qlm = attr.ib(default='')
    mmax_qlm = attr.ib(default='')
    xa = attr.ib(default='')
    xb = attr.ib(default='')


@attr.s
class DLENSALOT_Iteration(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    K = attr.ib(default='')
    # version, can be 'noMF
    V = attr.ib(default='')
    ITMAX = attr.ib(default='')
    IMIN = attr.ib(default='')
    IMAX = attr.ib(default='')
    get_btemplate_per_iteration = attr.ib(default='')
    # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
    # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
    # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
    QE_LENSING_CL_ANALYSIS = attr.ib(default='')
    # Change the following block only if exotic transferfunctions are desired
    STANDARD_TRANSFERFUNCTION = attr.ib(default='')
    # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
    FILTER = attr.ib(default='')
    # Change the following block only if exotic chain descriptor are desired
    CHAIN_DESCRIPTOR = attr.ib(default='')
    # Change the following block only if other than sepTP for QE is desired
    FILTER_QE = attr.ib(default='')
    # Choose your iterator. Either pertmf or const_mf
    ITERATOR = attr.ib(default='')
    # The following block defines various multipole limits. Change as desired
    lmax_filt = attr.ib(default='') # unlensed CMB iteration lmax
    lmin_tlm = attr.ib(default='')
    lmin_elm = attr.ib(default='')
    lmin_blm = attr.ib(default='')
    lmax_qlm = attr.ib(default='')
    mmax_qlm = attr.ib(default='')
    lmax_unl = attr.ib(default='')
    mmax_unl = attr.ib(default='')
    lmax_ivf = attr.ib(default='')
    mmax_ivf = attr.ib(default='')
    lmin_ivf = attr.ib(default='')
    mmin_ivf = attr.ib(default='')
    LENSRES = attr.ib(default='') # Deflection operations will be performed at this resolution
    Lmin = attr.ib(default='') # The reconstruction of all lensing multipoles below that will not be attempted
    # Meanfield, OBD, and tol settings
    CG_TOL = attr.ib(default='')
    TOL = attr.ib(default='')
    soltn_cond = attr.ib(default='')
    nsims_mf = attr.ib(default='')
    OMP_NUM_THREADS = attr.ib(default='')


@attr.s
class DLENSALOT_Geometry(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    lmax_unl = attr.ib(default='')
    zbounds = attr.ib(default='')
    zbounds_len = attr.ib(default='')
    pbounds = attr.ib(default='')
    nside = attr.ib(default='')
    lenjob_geometry = attr.ib(default='')
    lenjob_pbgeometry = attr.ib(default='')
    ninvjob_geometry = attr.ib(default='')