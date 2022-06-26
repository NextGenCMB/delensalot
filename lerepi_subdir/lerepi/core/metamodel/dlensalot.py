#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"
# TODO I would like to come up with a better structure for this whole 'DLENSALOT_Model'

import abc
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
    # version, can be 'noMF
    V = attr.ib(default=-1)
    tasks = attr.ib(default=-1)
    ITMAX = attr.ib(default=-1)
    IMIN = attr.ib(default=-1)
    IMAX = attr.ib(default=-1)
    dlm_mod = attr.ib(default=-1)
    mfvar = attr.ib(default=-1)
    # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
    # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
    # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
    QE_LENSING_CL_ANALYSIS = attr.ib(default=-1)
    # Change the following block only if exotic transferfunctions are desired
    STANDARD_TRANSFERFUNCTION = attr.ib(default=-1)
    # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
    FILTER = attr.ib(default=-1)
    # Change the following block only if exotic chain descriptor are desired
    CHAIN_DESCRIPTOR = attr.ib(default=-1)
    # Change the following block only if other than sepTP for QE is desired
    FILTER_QE = attr.ib(default=-1)
    # Choose your iterator. Either pertmf or const_mf
    iterator_typ = attr.ib(default=-1)
    # The following block defines various multipole limits. Change as desired
    lmax_filt = attr.ib(default=-1) # unlensed CMB iteration lmax
    lmax_qlm = attr.ib(default=-1)
    mmax_qlm = attr.ib(default=-1)
    lmax_unl = attr.ib(default=-1)
    mmax_unl = attr.ib(default=-1)
    lmax_ivf = attr.ib(default=-1)
    mmax_ivf = attr.ib(default=-1)
    lmin_ivf = attr.ib(default=-1)
    mmin_ivf = attr.ib(default=-1)
    LENSRES = attr.ib(default=-1) # Deflection operations will be performed at this resolution
    Lmin = attr.ib(default=-1) # The reconstruction of all lensing multipoles below that will not be attempted
    # Meanfield, OBD, and tol settings
    CG_TOL = attr.ib(default=-1)
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
    IMIN = attr.ib(default=-1)
    IMAX = attr.ib(default=-1)
    droplist = attr.ib(default=-1)
    ITMAX = attr.ib(default=-1)
    fg = attr.ib(default=-1)
    base_mask = attr.ib(default=-1)
    nlevels = attr.ib(default=-1)
    nside = attr.ib(default=-1)
    lmax_cl = attr.ib(default=-1)
    beam = attr.ib(default=-1)
    lmax_transf = attr.ib(default=-1)
    transf = attr.ib(default=-1)
    Cl_fid = attr.ib(default=-1)


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