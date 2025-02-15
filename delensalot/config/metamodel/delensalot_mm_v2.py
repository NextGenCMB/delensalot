#!/usr/bin/env python

"""dlensalot_mm.py: Contains classes defining the metamodel of the Dlensalot formalism.
    The metamodel is a structured representation, with the `DELENSALOT_Model` as the main building block.
    We use the attr package. It provides handy ways of validation and defaulting.
"""

import abc, attr, os, sys
from os.path import join as opj
from pathlib import Path
from attrs import validators
import numpy as np

import logging
log = logging.getLogger(__name__)

from delensalot.config.metamodel import DEFAULT_NotAValue, DEFAULT_NotASTR
from delensalot.config.validator import analysis, chaindescriptor, computing, data, filter as v_filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper

import importlib.util


class DELENSALOT_Concept_v2:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


    # def __str__(self):
    #     """ overwrites __str__ to summarize dlensalot model in a prettier way

    #     Returns:
    #         str: A table with all attributes of the model
    #     """        
    #     ##
    #     _str = ''
    #     for key, val in self.__dict__.items():
    #         keylen = len(str(key))
    #         if type(val) in [list, np.ndarray, np.array, dict]:
    #             _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(type(val))
    #         else:
    #             _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(val)
    #         _str += '\n'
    #     return _str


@attr.s
class DELENSALOT_Chaindescriptor(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to conjugate gradient solver. There are currently not many options for this. Better don't touch it.
    
    Attributes:
        p0: 0
        p1: type of the conditioner. Can be in ["diag_cl"]
        p2: value of lm_max_ivf[0]
        p3: value of nside of the data
        p4: np.inf
        p5: value of cg_tol
        p6: `tr_cg`: value of cd_solve.tr_cg
        p7: cacher setting
    """
    # TODO change names after testing various chains - can we find better heuristics?
    p0 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p0)
    p1 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p1)
    p2 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p2)
    p3 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p3)
    p4 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p4)
    p5 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p5)
    p6 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p6)
    p7 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p7)


@attr.s
class DELENSALOT_Job(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here.

    Attributes:
        jobs (list[str]): Job identifier(s)
    """
    jobs =                  attr.field(default=DEFAULT_NotAValue, validator=job.jobs)

@attr.s
class DELENSALOT_Analysis(DELENSALOT_Concept_v2):
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
    simidxs =               attr.field(default=DEFAULT_NotAValue, validator=analysis.simidxs)
    simidxs_mf =            attr.field(default=DEFAULT_NotAValue, validator=analysis.simidxs_mf)
    TEMP_suffix =           attr.field(default=DEFAULT_NotAValue, validator=analysis.TEMP_suffix)
    Lmin =                  attr.field(default=DEFAULT_NotAValue)
    zbounds =               attr.field(default=DEFAULT_NotAValue, validator=analysis.zbounds)
    zbounds_len =           attr.field(default=DEFAULT_NotAValue, validator=analysis.zbounds_len) # TODO rename
    mask =                  attr.field(default=DEFAULT_NotAValue, validator=analysis.mask) # TODO is this used? 
    lmin_teb =              attr.field(default=DEFAULT_NotAValue, validator=analysis.lmin_teb)
    cls_len =               attr.field(default=DEFAULT_NotAValue, validator=analysis.cls_len)
    beam =                  attr.field(default=DEFAULT_NotAValue, validator=analysis.beam)
    transfunction_desc =    attr.field(default=DEFAULT_NotAValue, validator=analysis.transfunction)
    CLfids =                attr.field(default=DEFAULT_NotAValue)
    secondary =             attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_Simulation(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the input maps, and values can differ from the noise model and analysis.

    Attributes:
                                     
    """
    flavour =       attr.field(default=DEFAULT_NotAValue, validator=data.flavour)
    libdir_suffix = attr.field(default='generic', validator=data.libdir_suffix)
    maps =          attr.field(default=DEFAULT_NotAValue, validator=data.maps)
    geominfo =      attr.field(default=DEFAULT_NotAValue, validator=data.geominfo)
    fid_info =      attr.field(default=DEFAULT_NotAValue)
    CMB_info =      attr.field(default=DEFAULT_NotAValue)
    sec_info =      attr.field(default=DEFAULT_NotAValue)
    obs_info =      attr.field(default=DEFAULT_NotAValue)
    operator_info = attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_Noisemodel(DELENSALOT_Concept_v2):
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
    sky_coverage =          attr.field(default=DEFAULT_NotAValue, validator=noisemodel.sky_coverage)
    spatial_type =          attr.field(default=DEFAULT_NotAValue)
    spectrum_type =         attr.field(default=DEFAULT_NotAValue, validator=noisemodel.spectrum_type)
    OBD =                   attr.field(default=DEFAULT_NotAValue, validator=noisemodel.OBD)
    nlev =                  attr.field(default=DEFAULT_NotAValue, validator=noisemodel.nlev_t)
    geominfo =              attr.field(default=DEFAULT_NotAValue, validator=noisemodel.ninvjob_geominfo) # FIXME this must match the data geominfo.. validate accordingly
    rhits_normalised =      attr.field(default=DEFAULT_NotAValue, validator=noisemodel.rhits_normalised)
    nivt_map =              attr.field(default=DEFAULT_NotAValue, validator=noisemodel.ninvjob_geominfo) # TODO test if it works
    nivp_map =              attr.field(default=DEFAULT_NotAValue, validator=noisemodel.ninvjob_geominfo) # TODO test if it works

@attr.s
class DELENSALOT_Qerec(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the quadratic estimator reconstruction job.

    Attributes:
        tasks (list[tuple]):        tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        qlm_type (str):             lensing potential estimator identifier. Can be 'sepTP' or 'jTP'
        cg_tol (float):             tolerance of the conjugate gradient method
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lm_max_qlm (type):          maximum multipole `\ell` and m to reconstruct the lensing potential
        chain (DELENSALOT_Chaindescriptor): configuration of the conjugate gradient method. Configures the chain and preconditioner
        cl_analysis (bool):         If tru, performs lensing power spectrum analysis
        blt_pert (bool):            If True, delensing is performed perurbitivly (recommended)
    
    """
    tasks =                 attr.field(default=DEFAULT_NotAValue, validator=qerec.tasks)
    estimator_type =        attr.field(default=DEFAULT_NotAValue, validator=qerec.qlms)
    qlm_type =              attr.field(default=DEFAULT_NotAValue, validator=qerec.qlms)
    cg_tol =                attr.field(default=DEFAULT_NotAValue, validator=qerec.cg_tol)
    chain =                 attr.field(default=DELENSALOT_Chaindescriptor(), validator=qerec.chain)
    subtract_QE_meanfield = attr.field(default=DEFAULT_NotAValue)

@attr.s
class DELENSALOT_Itrec(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the iterative reconstruction job.

    Attributes:
        tasks (list[str]):          tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        itmax (int):                maximum number of iterations
        cg_tol (float):             tolerance of the conjugate gradient method
        iterator_typ (str):         mean-field handling identifier. Can be either 'const_mf' or 'pert_mf'
        chain (DELENSALOT_Chaindescriptor): configuration for the conjugate gradient solver
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lenjob_geominfo (str):      can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lenjob_pbgeominfo (str):    can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lm_max_unl (tuple[int]):    maximum multipoles `\ell` and m for reconstruction the unlensed CMB
        lm_max_qlm (tuple[int]):    maximum multipoles L and m for reconstruction the lensing potential
        mfvar (str):                path to precalculated mean-field, to be used instead
        soltn_cond (type):          TBD
        stepper (DELENSALOT_STEPPER):configuration for updating the current likelihood iteration point with the likelihood gradient
              
    """
    tasks =                 attr.field(default=DEFAULT_NotAValue, validator=itrec.tasks)
    itmax =                 attr.field(default=DEFAULT_NotAValue, validator=itrec.itmax)
    cg_tol =                attr.field(default=DEFAULT_NotAValue, validator=itrec.cg_tol)
    chain =                 attr.field(default=DELENSALOT_Chaindescriptor(), validator=itrec.chain)
    mfvar =                 attr.field(default=DEFAULT_NotAValue, validator=itrec.mfvar) # TODO rename and check if it still works 
    soltn_cond =            attr.field(default=DEFAULT_NotAValue, validator=itrec.soltn_cond)
    gradient_descs =        attr.field(default=DEFAULT_NotAValue)
    filter_desc =           attr.field(default=DEFAULT_NotAValue)
    curvature_desc =        attr.field(default=DEFAULT_NotAValue)
    desc =                  attr.field(default=DEFAULT_NotAValue)

    
@attr.s
class DELENSALOT_Mapdelensing(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the internal map delensing job.

    Attributes:
        data_from_CFS (bool):   if set, use B-lensing templates located at the $CFS directory instead of the $TEMP directory\n
        edges (np.array):       binning to calculate the (delensed) power spectrum on\n
        iterations (list[int]): which iterations to calculate delensed power spectrum for\n
        nlevels (list[float]):  noiselevel ratio treshold up to which the maps are delensed, uses the rhits_normalized map to generate masks.
        lmax (int):             maximum multipole to calculate the (delensed) power spectrum\n
        Cl_fid (type):          fiducial power spectrum, and needed for template calculation of the binned power spectrum package\n
        libdir_it (type):       TBD\n
        binning (type):         can be either 'binned' or 'unbinned'. If 'unbinned', overwrites :code:`edges` and calculates power spectrum for each multipole\n
        spectrum_calculator (package): name of the package of the power spectrum calculator. Can be 'healpy' if :code:`binning=unbinned`\n
        masks_fn (list[str]):   the sky patches to calculate the power spectra on. Note that this is different to using `nlevels`. Here, no tresholds are calculated, but masks are used 'as is' for delensing.\n
        basemap (str):          the delensed map Bdel is calculated as Bdel = basemap - blt. Basemap can be two things: 'obs' or 'lens', where 'obs' will use the observed sky map, and lens will use the pure B-lensing map.
    """

    data_from_CFS =         attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.data_from_CFS)
    edges =                 attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.edges)
    iterations =            attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.iterations)
    nlevels =               attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.nlevels)
    lmax =                  attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.lmax)
    Cl_fid =                attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.Cl_fid)
    libdir_it =             attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.libdir_it)
    binning =               attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.binning)
    spectrum_calculator =   attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.spectrum_calculator)
    masks_fn =              attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.masks)
    basemap =               attr.field(default=DEFAULT_NotAValue, validator=mapdelensing.basemap)

@attr.s
class DELENSALOT_Phianalysis(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the internal map delensing job.

    Attributes:
        custom_WF_TEMP (str):   Path to the dir of an exisiting WF. fn must be 'WFemp_%s_simall%s_itall%s_avg.npy'\n
    """
    custom_WF_TEMP =        attr.field(default=DEFAULT_NotAValue)


@attr.s
class DELENSALOT_OBD(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the overlapping B-mode deprojection.

    Attributes:
        libdir (str):       path to the OBD matrix
        rescale (float):    rescaling of OBD matrix amplitude. Useful if matrix already calculated, but noiselevel changed
        tpl (type):         function name for calculating OBD matrix
        nlev_dep (float):   deprojection factor, or, strength of B-mode deprojection                   
    """
    libdir =                attr.field(default=DEFAULT_NotAValue, validator=obd.libdir)
    rescale =               attr.field(default=DEFAULT_NotAValue, validator=obd.rescale) # TODO this is a very specific parameter.. keep?
    tpl =                   attr.field(default=DEFAULT_NotAValue, validator=obd.tpl)
    nlev_dep =              attr.field(default=DEFAULT_NotAValue, validator=obd.nlev_dep)


@attr.s
class DELENSALOT_Meta(DELENSALOT_Concept_v2): # TODO do we really need a Meta?
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to internal behaviour of delensalot.

    Attributes:
        version (str):  version control of the delensalot model
    """
    version =               attr.field(default=DEFAULT_NotAValue, validator=attr.validators.instance_of(int))


@attr.s
class DELENSALOT_Computing(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the usage of computing resources.

    Attributes:
        OMP_NUM_THREADS (int):  number of threads used per Job
    """
    OMP_NUM_THREADS =       attr.field(default=DEFAULT_NotAValue, validator=computing.OMP_NUM_THREADS)


@attr.s
class DELENSALOT_Model(DELENSALOT_Concept_v2):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        defaults_to (str):              Identifier for default-dictionary if user hasn't specified value in configuration file
        meta (DELENSALOT_Meta):          configurations related to internal behaviour of delensalot
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
    
    defaults_to =           attr.field(default='default_jointrec')
    validate_model =        attr.field(default=True)
    meta =                  attr.field(default=DELENSALOT_Meta(), validator=model.meta)
    job =                   attr.field(default=DELENSALOT_Job(), validator=model.job)
    analysis =              attr.field(default=DELENSALOT_Analysis(), validator=model.analysis)
    simulationdata =        attr.field(default=DELENSALOT_Simulation(), validator=model.data)
    noisemodel =            attr.field(default=DELENSALOT_Noisemodel(), validator=model.noisemodel)
    qerec =                 attr.field(default=DELENSALOT_Qerec(), validator=model.qerec)
    itrec =                 attr.field(default=DELENSALOT_Itrec(), validator=model.itrec)
    madel =                 attr.field(default=DELENSALOT_Mapdelensing(), validator=model.madel)
    computing =             attr.field(default=DELENSALOT_Computing(), validator=model.computing)
    obd =                   attr.field(default=DELENSALOT_OBD(), validator=model.obd)
    phana =                 attr.field(default=DELENSALOT_Phianalysis())
    

    def __attrs_post_init__(self):
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
                        target[key] = default_value
                    elif isinstance(default_value, dict) and isinstance(target[key], dict):
                        update_defaults(target[key], default_value)
                else:  # Handle object attributes
                    if not hasattr(target, key) or np.all(getattr(target, key) == DEFAULT_NotAValue):
                        setattr(target, key, default_value)
                    elif isinstance(default_value, dict) and isinstance(getattr(target, key), dict):
                        update_defaults(getattr(target, key), default_value)

        # apply updates to all top-level attributes
        for key, default_value in default_dict.items():
            if key in ['defaults_to', 'validate_model']:
                continue  # Skip special attributes
            if key in ['simulationdata', 'analysis']:
                # NOTE this only updates secondary keys if any secondary is actually listed in the analysis of the config file. 
                # By this I make sure that the library only receives the secondaries that the user wants,
                # while at the same time setting the defaults for that secondary if the user did not specify
                for value in default_dict[key]:
                    target_attr = getattr(self, key)
                    if value in ['sec_info', 'secondary']:
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
                if not hasattr(self, key) or getattr(self, key) == DEFAULT_NotAValue:
                    setattr(self, key, default_value)
                update_defaults(getattr(self, key), default_value)
