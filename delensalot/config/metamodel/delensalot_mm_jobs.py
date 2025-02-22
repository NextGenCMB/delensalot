
import abc, attr, os, sys
from os.path import join as opj
from pathlib import Path
import numpy as np
import attr
import traceback

import logging
log = logging.getLogger(__name__)

from delensalot.config.metamodel import DEFAULT_NotAValue, DEFAULT_NotASTR
import importlib.util

class DELENSALOT_Concept_v3:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta

    def __str__(self):
        """ overwrites __str__ to summarize dlensalot model in a prettier way

        Returns:
            str: A table with all attributes of the model
        """        
        ##
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
class DELENSALOT_Job(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here.

    Attributes:
        jobs (list[str]): Job identifier(s)
    """
    jobs =                  attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_DataContainer(DELENSALOT_Concept_v3):

    flavour =       attr.field(default=DEFAULT_NotAValue)
    libdir_suffix = attr.field(default='generic')
    maps =          attr.field(default=DEFAULT_NotAValue)
    geominfo =      attr.field(default=DEFAULT_NotAValue)
    fid_info =      attr.field(default=DEFAULT_NotAValue)
    CMB_info =      attr.field(default=DEFAULT_NotAValue)
    sec_info =      attr.field(default=DEFAULT_NotAValue)
    obs_info =      attr.field(default=DEFAULT_NotAValue)
    operator_info = attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_QE_scheduler(DELENSALOT_Concept_v3):

    template_operator  =  attr.field(default=DEFAULT_NotAValue)
    idxs =                attr.field(default=DEFAULT_NotAValue)
    idxs_mf =             attr.field(default=DEFAULT_NotAValue)
    tasks =               attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_MAP_scheduler(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the iterative reconstruction job.

    Attributes:
        tasks (list[str]):          tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        itmax (int):                maximum number of iterations
        cg_tol (float):             tolerance of the conjugate gradient method
        iterator_typ (str):         mean-field handling identifier. Can be either 'const_mf' or 'pert_mf'
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lenjob_geominfo (str):      can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lenjob_pbgeominfo (str):    can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lm_max_unl (tuple[int]):    maximum multipoles `\ell` and m for reconstruction the unlensed CMB
        lm_max_qlm (tuple[int]):    maximum multipoles L and m for reconstruction the lensing potential
        mfvar (str):                path to precalculated mean-field, to be used instead
        soltn_cond (type):          TBD
        stepper (DELENSALOT_STEPPER):configuration for updating the current likelihood iteration point with the likelihood gradient
              
    """
    idxs =                  attr.field(default=DEFAULT_NotAValue)
    idxs_mf =               attr.field(default=DEFAULT_NotAValue)
    data_container =        attr.field(default=DEFAULT_NotAValue)
    QE_searchs =            attr.field(default=DEFAULT_NotAValue)
    tasks =                 attr.field(default=DEFAULT_NotAValue)
    MAP_minimizers =        attr.field(default=DEFAULT_NotAValue)
    filter_desc =           attr.field(default=DEFAULT_NotAValue)
    curvature_desc =        attr.field(default=DEFAULT_NotAValue)
    desc =                  attr.field(default=DEFAULT_NotAValue)