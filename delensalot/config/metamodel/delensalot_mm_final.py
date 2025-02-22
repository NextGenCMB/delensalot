#!/usr/bin/env python

"""dlensalot_mm.py: Contains classes defining the metamodel of the Dlensalot formalism.
    The metamodel is a structured representation, with the `DELENSALOT_Model` as the main building block.
    We use the attr package. It provides handy ways of validation and defaulting.
"""

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
class DELENSALOT_Analysis(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the specific analysis performed on the data.

    Attributes:
        key (str):                          reconstruction estimator key
        version (str):                      specific configuration for the esimator (e.g. `noMF`, which turns off mean-field subtraction)
        simidxs (np.array[int]):            simulation indices to use for the delensalot job
        simidxs_mf (np.array[int]):         simulation indices to use for the calculation of the mean-field
        TEMP_suffix (str):                  identifier to customize TEMP directory of the analysis
        Lmin (int):                         minimum L for reconstructing the lensing potential
        zbounds (tuple[int or str,float]):  latitudinal boundary (-1 to 1), or identifier together with noise level ratio treshold at which lensing reconstruction is perfromed.
        zbounds_len (tuple[int]):           latitudinal extended boundary at which lensing reconstruction is performed, and used for iterative lensing reconstruction
        lm_max_ivf (tuple[int]):            maximum `\ell` and m for which inverse variance filtering is done
        lm_max_blt (tuple[int]):            maximum `\ell` and m for which B-lensing template is calculated
        mask (list[str]):                   TBD
        lmin_teb (int):                     minimum `\ell` and m of the data which the reconstruction uses, and is set to zero below via the transfer function
        cls_len (str):                      path to the fiducial lensed CAMB-like CMB data
        cpp (str):                          path to the power spectrum of the prior for the iterative reconstruction
        beam (float):                       The beam used in the filters
    """
    LM_max =                attr.field(default=DEFAULT_NotAValue)
    lm_max_pri =            attr.field(default=DEFAULT_NotAValue)
    lm_max_sky =            attr.field(default=DEFAULT_NotAValue)
    key =                   attr.field(default=DEFAULT_NotAValue)
    simidxs =               attr.field(default=DEFAULT_NotAValue)
    simidxs_mf =            attr.field(default=DEFAULT_NotAValue)
    TEMP_suffix =           attr.field(default=DEFAULT_NotAValue)
    Lmin =                  attr.field(default=DEFAULT_NotAValue)
    zbounds =               attr.field(default=DEFAULT_NotAValue)
    zbounds_len =           attr.field(default=DEFAULT_NotAValue)
    mask =                  attr.field(default=DEFAULT_NotAValue)
    lmin_teb =              attr.field(default=DEFAULT_NotAValue)
    cls_len =               attr.field(default=DEFAULT_NotAValue)
    beam =                  attr.field(default=DEFAULT_NotAValue)
    transfunction_desc =    attr.field(default=DEFAULT_NotAValue)
    CLfids =                attr.field(default=DEFAULT_NotAValue)
    secondary =             attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_DataSource(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the input maps, and values can differ from the noise model and analysis.

    Attributes:
                                     
    """
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
class DELENSALOT_Noisemodel(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the noise model used for Wiener-filtering the data.

    Attributes:
        sky_coverage (str):     Can be either 'masked' or 'unmasked'
        spectrum_type (str):    TBD
        OBD (str):              OBD identifier. Can be 'OBD', 'trunc'. Defines how lowest B-modes will be handled.
        nlev_t (float):         (central) noise level of temperature data in muK arcmin.
        nlev_p (float):         (central) noise level of polarization data in muK arcmin.
        rhits_normalised (str): path to the hits-count map, used to calculate the noise levels, and the mask tracing the noise level. Second entry in tuple is the <inverse hits-count multiplier>.
        geominfo (tuple): geominfo of the noise map
    """
    sky_coverage =          attr.field(default=DEFAULT_NotAValue)
    spatial_type =          attr.field(default=DEFAULT_NotAValue)
    spectrum_type =         attr.field(default=DEFAULT_NotAValue)
    OBD =                   attr.field(default=DEFAULT_NotAValue)
    nlev =                  attr.field(default=DEFAULT_NotAValue)
    geominfo =              attr.field(default=DEFAULT_NotAValue)
    rhits_normalised =      attr.field(default=DEFAULT_NotAValue)
    nivt_map =              attr.field(default=DEFAULT_NotAValue)
    nivp_map =              attr.field(default=DEFAULT_NotAValue)


@ attr.s
class DELENSALOT_Operator(DELENSALOT_Concept_v3):

    beam =                 attr.field(default=DEFAULT_NotAValue)
    ivf =                  attr.field(default=DEFAULT_NotAValue)
    wf =                   attr.field(default=DEFAULT_NotAValue)
    inoise =               attr.field(default=DEFAULT_NotAValue)
    secondary =            attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_QE_search(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the quadratic estimator reconstruction job.

    Attributes:
        tasks (list[tuple]):        tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        qlm_type (str):             lensing potential estimator identifier. Can be 'sepTP' or 'jTP'
        cg_tol (float):             tolerance of the conjugate gradient method
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lm_max_qlm (type):          maximum multipole `\ell` and m to reconstruct the lensing potential
        cl_analysis (bool):         If tru, performs lensing power spectrum analysis
        blt_pert (bool):            If True, delensing is performed perurbitivly (recommended)
    
    """
    estimator_key =         attr.field(default=DEFAULT_NotAValue)
    CLfids =                attr.field(default=DEFAULT_NotAValue)
    subtract_meanfield =    attr.field(default=DEFAULT_NotAValue)
    QE_filterqest_desc =    attr.field(default=DEFAULT_NotAValue)
    ID =                    attr.field(default=DEFAULT_NotAValue)
    libdir =                attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_Computing(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the usage of computing resources.

    Attributes:
        OMP_NUM_THREADS (int):  number of threads used per Job
    """
    OMP_NUM_THREADS =       attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_Model(DELENSALOT_Concept_v3):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        defaults_to (str):              Identifier for default-dictionary if user hasn't specified value in configuration file
        job (DELENSALOT_Job):            delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, analyse_phi) which is controlled here
        analysis (DELENSALOT_Analysis):  configurations related to the specific analysis performed on the data
        data (DELENSALOT_Data):          configurations related to the input CMB maps
        noisemodel (DELENSALOT_Noisemodel):  configurations related to the noise model used for Wiener-filtering the data
        qerec (DELENSALOT_Qerec):        configurations related to the quadratic estimator reconstruction job
        itrec (DELENSALOT_Itrec):        configurations related to the iterative reconstruction job
        madel (DELENSALOT_Mapdelensing): configurations related to the internal map delensing job
        config (DELENSALOT_Config):      configurations related to general behaviour to the operating system
        computing (DELENSALOT_Computing):    configurations related to the usage of computing resources
        obd (DELENSALOT_OBD):            configurations related to the overlapping B-mode deprojection
        phana (DELENSALOT_Phyanalysis):  configurations related to the simple power spectrum analaysis of phi

    """
    
    defaults_to =           attr.field(default='default_jointrec_v3')
    validate_model =        attr.field(default=True)
    analysis =              attr.field(default=DELENSALOT_Analysis())
    data_source =           attr.field(default=DELENSALOT_DataSource())
    operators =             attr.field(default=DELENSALOT_Noisemodel())
    computing =             attr.field(default=DELENSALOT_Computing())

    QE_filterqest =         attr.field(default=DEFAULT_NotAValue)
    QE_search =             attr.field(default=DEFAULT_NotAValue)

    wf_filter =             attr.field(default=DEFAULT_NotAValue)
    ivf_filter =            attr.field(default=DEFAULT_NotAValue)
    gradient =              attr.field(default=DEFAULT_NotAValue)
    likelihood =            attr.field(default=DEFAULT_NotAValue)
    minimizer =             attr.field(default=DEFAULT_NotAValue)


    def __attrs_post_init__(self):
        """Ensure missing attributes are set to their class-level default values."""
        for field in attr.fields(self.__class__):
            if not hasattr(self, field.name) or getattr(self, field.name) is None:
                setattr(self, field.name, field.default)

        print('setting defaults')
        default_path = Path(__file__).parent.parent / f"default/{self.defaults_to.replace('.py', '')}.py"
        spec = importlib.util.spec_from_file_location("default", default_path)
        default_module = importlib.util.module_from_spec(spec)
        sys.modules["default"] = default_module
        spec.loader.exec_module(default_module)
        default_dict = default_module.DL_DEFAULT

        def update_defaults(target, defaults):
            """Recursively update target with default values, ensuring all keys from defaults exist."""
            for key, default_value in defaults.items():
                if isinstance(target, dict):
                    if key not in target or np.all(target[key] == DEFAULT_NotAValue):
                        print("target key::::", key, default_value)
                        target[key] = default_value
                    elif isinstance(default_value, dict) and isinstance(target[key], dict):
                        update_defaults(target[key], default_value)
                else:  # Handle object attributes
                    if not hasattr(target, key) or np.all(getattr(target, key) == DEFAULT_NotAValue):
                        setattr(target, key, default_value)
                    elif isinstance(default_value, dict) and isinstance(getattr(target, key), dict):
                        update_defaults(getattr(target, key), default_value)

        # apply updates to all top-level attributes
        for default_key, default_value in default_dict.items():
            if default_key in ['defaults_to', 'validate_model']:
                continue  # Skip special attributes
            if default_key in ['data_source']:#, 'analysis']:
                # NOTE this only updates secondary keys if any secondary is actually listed in the analysis of the config file. 
                # By this I make sure that the library only receives the secondaries that the user wants,
                # while at the same time setting the defaults for that secondary if the user did not specify
                for value in default_dict[default_key]:
                    target_attr = getattr(self, default_key)
                    if value in ['sec_info']:#, 'secondary']:
                        attr_value = getattr(target_attr, value)
                        if attr_value == DEFAULT_NotAValue:
                            # NOTE if no sec_info is given, we need to set the default sec_info
                            setattr(target_attr, value, default_value[value])
                        else:
                            for sub_key in default_value[value]:
                                if sub_key in getattr(target_attr, value, {}):
                                    update_defaults(getattr(target_attr, value)[sub_key], default_value[value][sub_key])
                    else:
                        attr_value = getattr(target_attr, value)
                        if isinstance(attr_value, dict):
                            update_defaults(attr_value, default_value[value])
                        else:
                            if attr_value == DEFAULT_NotAValue:
                                setattr(target_attr, value, default_value[value])
                            else:
                                setattr(target_attr, value, attr_value)
            else:
                if not hasattr(self, default_key) or getattr(self, default_key) == DEFAULT_NotAValue:
                    print('setting default for class key:', default_key)
                    setattr(self, default_key, default_value)
                update_defaults(getattr(self, default_key), default_value)


    def fill_with_defaults(self):
        self.__attrs_post_init__()
