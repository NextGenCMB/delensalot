#!/usr/bin/env python

"""dlensalot_mm.py: Contains classes defining the metamodel of the Dlensalot formalism.
    The metamodel is a structured representation, with the `DLENSALOT_Model` as the main building block.
    We use the attr package. It provides handy ways of validation and defaulting.
"""

import abc, attr, os, sys
from os.path import join as opj
from pathlib import Path
import numpy as np

import logging
log = logging.getLogger(__name__)

from delensalot.config.metamodel import DEFAULT_NotAValue, DEFAULT_NotASTR
import importlib.util


class DLENSALOT_Concept:
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
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
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
    p0 =                    attr.field(default=DEFAULT_NotAValue)
    p1 =                    attr.field(default=DEFAULT_NotAValue)
    p2 =                    attr.field(default=DEFAULT_NotAValue)
    p3 =                    attr.field(default=DEFAULT_NotAValue)
    p4 =                    attr.field(default=DEFAULT_NotAValue)
    p5 =                    attr.field(default=DEFAULT_NotAValue)
    p6 =                    attr.field(default=DEFAULT_NotAValue)
    p7 =                    attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    Defines the stepper function. The stepper function controls how the increment in the likelihood search is added to the current solution.
    Currently, this is pretty much just the harmonicbump class.
    
    Attributes:
        typ (str): The name of the stepper function
        lmax_qlm (int): maximum `\ell` of the lensing potential reconstruction
        mmax_qlm (int): maximum `m` of the lensing potential reconstruction
        a: TBD
        b: TBD
        xa: TBD
        xb: TBD
    """
    # FIXME this is very 'harmonicbump'-specific.
    typ =                   attr.field(default=DEFAULT_NotAValue)
    lmax_qlm =              attr.field(default=DEFAULT_NotAValue)
    mmax_qlm =              attr.field(default=DEFAULT_NotAValue)
    a =                     attr.field(default=DEFAULT_NotAValue)
    b =                     attr.field(default=DEFAULT_NotAValue)
    xa =                    attr.field(default=DEFAULT_NotAValue)
    xb =                    attr.field(default=DEFAULT_NotAValue)


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here.

    Attributes:
        jobs (list[str]): Job identifier(s)
    """
    jobs =                  attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
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
        cls_unl (str):                      path to the fiducial unlensed CAMB-like CMB data
        cls_len (str):                      path to the fiducial lensed CAMB-like CMB data
        cpp (str):                          path to the power spectrum of the prior for the iterative reconstruction
        beam (float):                       The beam used in the filters
    """
    key =                   attr.field(default=DEFAULT_NotAValue)
    version =               attr.field(default=DEFAULT_NotAValue)
    simidxs =               attr.field(default=DEFAULT_NotAValue)
    simidxs_mf =            attr.field(default=DEFAULT_NotAValue)
    TEMP_suffix =           attr.field(default=DEFAULT_NotAValue)
    Lmin =                  attr.field(default=DEFAULT_NotAValue)
    zbounds =               attr.field(default=DEFAULT_NotAValue)
    zbounds_len =           attr.field(default=DEFAULT_NotAValue)
    lm_max_ivf =            attr.field(default=DEFAULT_NotAValue)
    lm_max_blt =            attr.field(default=DEFAULT_NotAValue)
    mask =                  attr.field(default=DEFAULT_NotAValue)
    lmin_teb =              attr.field(default=DEFAULT_NotAValue)
    cls_unl =               attr.field(default=DEFAULT_NotAValue)
    cls_len =               attr.field(default=DEFAULT_NotAValue)
    cpp =                   attr.field(default=DEFAULT_NotAValue)
    beam =                  attr.field(default=DEFAULT_NotAValue)
    transfunction_desc =    attr.field(default=DEFAULT_NotAValue)


@attr.s
class DLENSALOT_Simulation(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the input maps, and values can differ from the noise model and analysis.

    Attributes:
        flavour      (str): Can be in ['obs', 'sky', 'unl'] and defines the type of data provided.
        space        (str): Can be in ['map', 'alm', 'cl'] and defines the space of the data provided.
        maps         (np.array, optional): These maps will be put into the cacher directly. They are used for settings in which no data is generated or accesed on disk, but directly provided (like in `delensalot.anafast()`) Defaults to DNaV.
        geominfo     (tuple, optional): Lenspyx geominfo descriptor, describes the geominfo of the data provided (e.g. `('healpix', 'nside': 2048)). Defaults to DNaV.
        field        (str, optional): the type of data provided, can be in ['temperature', 'polarization']. Defaults to DNaV.
        libdir       (str, optional): directory of the data provided. Defaults to DNaV.
        libdir_noise (str, optional): directory of the noise provided. Defaults to DNaV.
        libdir_phi   (str, optional): directory of the lensing potential provided. Defaults to DNaV.
        fns          (dict with str with formatter, optional): file names of the data provided. It expects `{'T': <filename{simidx}.something>, 'Q': <filename{simidx}.something>, 'U': <filename{simidx}.something>}`, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
        fnsnoise     (dict with str with formatter, optional): file names of the noise provided. It expects `{'T': <filename{simidx}.something>, 'Q': <filename{simidx}.something>, 'U': <filename{simidx}.something>}`, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
        fnsP         (str with formatter, optional): file names of the lensing potential provided. It expects `<filename{simidx}.something>, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
        lmax         (int, optional): Maximum l of the data provided. Defaults to DNaV.
        transfunction(np.array, optional): transfer function. Defaults to DNaV.
        nlev         (dict, optional): noise level of the individual fields. It expects `{'T': <value>, 'P': <value>}. Defaults to DNaV.
        spin         (int, optional): the spin of the data provided. Defaults to 0. Always defaults to 0 for temperature.
        CMB_fn       (str, optional): path+name of the file of the power spectra of the CMB. Defaults to DNaV.
        phi_fn       (str, optional): path+name of the file of the power spectrum of the lensing potential. Defaults to DNaV.
        phi_field    (str, optional): the type of potential provided, can be in ['potential', 'deflection', 'convergence']. This simulation library will automatically rescale the field, if needded. Defaults to DNaV.
        phi_space    (str, optional): can be in ['map', 'alm', 'cl'] and defines the space of the lensing potential provided.. Defaults to DNaV.
        phi_lmax     (_type_, optional): the maximum multipole of the lensing potential. if simulation library perfroms lensing, it is advisable that `phi_lmax` is somewhat larger than `lmax` (+ ~512-1024). Defaults to DNaV.
        epsilon      (float, optional): Lenspyx lensing accuracy. Defaults to 1e-7.
        libdir_suffix(str, optional): defines the directory the simulation data will be stored to, defaults to 'generic'. Helpful if one wants to keep track of different projects.
        CMB_modifier (callable, optional): operation defined in the callable will be applied to each of the input maps/alms/cls
        phi_modifier (callable, optional): operation defined in the callable will be applied to the input phi lms
                                               
    """

    flavour =       attr.field(default=DEFAULT_NotAValue)
    space =         attr.field(default=DEFAULT_NotAValue)
    maps =          attr.field(default=DEFAULT_NotAValue)
    geominfo =      attr.field(default=DEFAULT_NotAValue)
    lenjob_geominfo=attr.field(default=DEFAULT_NotAValue)
    field =         attr.field(default=DEFAULT_NotAValue)
    libdir =        attr.field(default=DEFAULT_NotAValue)
    libdir_noise =  attr.field(default=DEFAULT_NotAValue)
    libdir_phi =    attr.field(default=DEFAULT_NotAValue)
    fns =           attr.field(default=DEFAULT_NotAValue)
    fnsnoise =      attr.field(default=DEFAULT_NotAValue)
    fnsP =          attr.field(default=DEFAULT_NotAValue)
    lmax =          attr.field(default=DEFAULT_NotAValue)
    transfunction = attr.field(default=DEFAULT_NotAValue)
    nlev =          attr.field(default=DEFAULT_NotAValue)
    spin =          attr.field(default=DEFAULT_NotAValue)
    CMB_fn =        attr.field(default=DEFAULT_NotAValue)
    phi_fn =        attr.field(default=DEFAULT_NotAValue)
    phi_field =     attr.field(default=DEFAULT_NotAValue)
    phi_space =     attr.field(default=DEFAULT_NotAValue)
    phi_lmax =      attr.field(default=DEFAULT_NotAValue)
    epsilon =       attr.field(default=DEFAULT_NotAValue)
    libdir_suffix = attr.field(default='generic')
    CMB_modifier =  attr.field(default=DEFAULT_NotAValue)
    phi_modifier =  attr.field(default=lambda x: x)
    
    
@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
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
    spectrum_type =         attr.field(default=DEFAULT_NotAValue)
    OBD =                   attr.field(default=DEFAULT_NotAValue)
    nlev =                  attr.field(default=DEFAULT_NotAValue)
    geominfo =              attr.field(default=DEFAULT_NotAValue)
    rhits_normalised =      attr.field(default=DEFAULT_NotAValue)
    nivt_map =              attr.field(default=DEFAULT_NotAValue)
    nivp_map =              attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the quadratic estimator reconstruction job.

    Attributes:
        tasks (list[tuple]):        tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        qlm_type (str):             lensing potential estimator identifier. Can be 'sepTP' or 'jTP'
        cg_tol (float):             tolerance of the conjugate gradient method
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lm_max_qlm (type):          maximum multipole `\ell` and m to reconstruct the lensing potential
        chain (DLENSALOT_Chaindescriptor): configuration of the conjugate gradient method. Configures the chain and preconditioner
        cl_analysis (bool):         If tru, performs lensing power spectrum analysis
        blt_pert (bool):            If True, delensing is performed perurbitivly (recommended)
    
    """

    tasks =                 attr.field(default=DEFAULT_NotAValue)
    qlm_type =              attr.field(default=DEFAULT_NotAValue)
    cg_tol =                attr.field(default=DEFAULT_NotAValue)
    filter_directional =    attr.field(default=DEFAULT_NotAValue)
    lm_max_qlm =            attr.field(default=DEFAULT_NotAValue)
    chain =                 attr.field(default=DLENSALOT_Chaindescriptor())
    cl_analysis =           attr.field(default=DEFAULT_NotAValue)
    blt_pert =              attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the iterative reconstruction job.

    Attributes:
        tasks (list[str]):          tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        itmax (int):                maximum number of iterations
        cg_tol (float):             tolerance of the conjugate gradient method
        iterator_typ (str):         mean-field handling identifier. Can be either 'const_mf' or 'pert_mf'
        chain (DLENSALOT_Chaindescriptor): configuration for the conjugate gradient solver
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        lenjob_geominfo (str):      can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lenjob_pbgeominfo (str):    can be 'healpix_geominfo', 'thin_gauss' or 'pbdGeometry'
        lm_max_unl (tuple[int]):    maximum multipoles `\ell` and m for reconstruction the unlensed CMB
        lm_max_qlm (tuple[int]):    maximum multipoles L and m for reconstruction the lensing potential
        mfvar (str):                path to precalculated mean-field, to be used instead
        soltn_cond (type):          TBD
        stepper (DLENSALOT_STEPPER):configuration for updating the current likelihood iteration point with the likelihood gradient
              
    """
    tasks =                 attr.field(default=DEFAULT_NotAValue)
    itmax =                 attr.field(default=DEFAULT_NotAValue)
    cg_tol =                attr.field(default=DEFAULT_NotAValue)
    iterator_typ =          attr.field(default=DEFAULT_NotAValue)
    chain =                 attr.field(default=DLENSALOT_Chaindescriptor())
    filter_directional =    attr.field(default=DEFAULT_NotAValue)
    lenjob_geominfo =       attr.field(default=DEFAULT_NotAValue)
    lenjob_pbdgeominfo =    attr.field(default=DEFAULT_NotAValue)
    lm_max_unl =            attr.field(default=DEFAULT_NotAValue)
    lm_max_qlm =            attr.field(default=DEFAULT_NotAValue)
    mfvar =                 attr.field(default=DEFAULT_NotAValue)
    soltn_cond =            attr.field(default=DEFAULT_NotAValue)
    stepper =               attr.field(default=DLENSALOT_Stepper())
    epsilon =               attr.field(default=DEFAULT_NotAValue)
    
@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the internal map delensing job.

    Attributes:
        data_from_CFS (bool):   if set, use B-lensing templates located at the $CFS directory instead of the $TEMP directory\n
        edges (np.array):       binning to calculate the (delensed) power spectrum on\n
        dlm_mod (bool):         if set, modfies the lensing potential before calculating the B-lensing template\n
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

    data_from_CFS =         attr.field(default=DEFAULT_NotAValue)
    edges =                 attr.field(default=DEFAULT_NotAValue)
    dlm_mod =               attr.field(default=DEFAULT_NotAValue)
    iterations =            attr.field(default=DEFAULT_NotAValue)
    nlevels =               attr.field(default=DEFAULT_NotAValue)
    lmax =                  attr.field(default=DEFAULT_NotAValue)
    Cl_fid =                attr.field(default=DEFAULT_NotAValue)
    libdir_it =             attr.field(default=DEFAULT_NotAValue)
    binning =               attr.field(default=DEFAULT_NotAValue)
    spectrum_calculator =   attr.field(default=DEFAULT_NotAValue)
    masks_fn =              attr.field(default=DEFAULT_NotAValue)
    basemap =               attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Phianalysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the internal map delensing job.

    Attributes:
        custom_WF_TEMP (str):   Path to the dir of an exisiting WF. fn must be 'WFemp_%s_simall%s_itall%s_avg.npy'\n
    """

    custom_WF_TEMP =        attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the overlapping B-mode deprojection.

    Attributes:
        libdir (str):       path to the OBD matrix
        rescale (float):    rescaling of OBD matrix amplitude. Useful if matrix already calculated, but noiselevel changed
        tpl (type):         function name for calculating OBD matrix
        nlev_dep (float):   deprojection factor, or, strength of B-mode deprojection                   
    """
    libdir =                attr.field(default=DEFAULT_NotAValue)
    rescale =               attr.field(default=DEFAULT_NotAValue)
    tpl =                   attr.field(default=DEFAULT_NotAValue)
    nlev_dep =              attr.field(default=DEFAULT_NotAValue)

@attr.s
class DLENSALOT_Config(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to general behaviour to the operating system. 

    Attributes:
        outdir_plot_root (str): root path for the plots to be stored at
        outdir_plot_rel (str):  relative path folder for the plots to be stored at
    """
    outdir_plot_root =      attr.field(default=opj(os.environ['HOME'], 'plots'))
    outdir_plot_rel =       attr.field(default='')

@attr.s
class DLENSALOT_Meta(DLENSALOT_Concept): # TODO do we really need a Meta?
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to internal behaviour of delensalot.

    Attributes:
        version (str):  version control of the delensalot model
    """
    version =               attr.field(default=DEFAULT_NotAValue)


@attr.s
class DLENSALOT_Computing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the usage of computing resources.

    Attributes:
        OMP_NUM_THREADS (int):  number of threads used per Job
    """
    OMP_NUM_THREADS =       attr.field(default=DEFAULT_NotAValue)


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        defaults_to (str):              Identifier for default-dictionary if user hasn't specified value in configuration file
        meta (DLENSALOT_Meta):          configurations related to internal behaviour of delensalot
        job (DLENSALOT_Job):            delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, analyse_phi) which is controlled here
        analysis (DLENSALOT_Analysis):  configurations related to the specific analysis performed on the data
        data (DLENSALOT_Data):          configurations related to the input CMB maps
        noisemodel (DLENSALOT_Noisemodel):  configurations related to the noise model used for Wiener-filtering the data
        qerec (DLENSALOT_Qerec):        configurations related to the quadratic estimator reconstruction job
        itrec (DLENSALOT_Itrec):        configurations related to the iterative reconstruction job
        madel (DLENSALOT_Mapdelensing): configurations related to the internal map delensing job
        config (DLENSALOT_Config):      configurations related to general behaviour to the operating system
        computing (DLENSALOT_Computing):    configurations related to the usage of computing resources
        obd (DLENSALOT_OBD):            configurations related to the overlapping B-mode deprojection
        phana (DLENSALOT_Phyanalysis):  configurations related to the simple power spectrum analaysis of phi

    """
    
    defaults_to =           attr.field(default='default_CMBS4_fullsky_polarization')
    validate_model =        attr.field(default=True)
    meta =                  attr.field(default=DLENSALOT_Meta())
    job =                   attr.field(default=DLENSALOT_Job())
    analysis =              attr.field(default=DLENSALOT_Analysis())
    simulationdata =        attr.field(default=DLENSALOT_Simulation())
    noisemodel =            attr.field(default=DLENSALOT_Noisemodel())
    qerec =                 attr.field(default=DLENSALOT_Qerec())
    itrec =                 attr.field(default=DLENSALOT_Itrec())
    madel =                 attr.field(default=DLENSALOT_Mapdelensing())
    config =                attr.field(default=DLENSALOT_Config())
    computing =             attr.field(default=DLENSALOT_Computing())
    obd =                   attr.field(default=DLENSALOT_OBD())
    phana =                 attr.field(default=DLENSALOT_Phianalysis())
    

    def __attrs_post_init__(self):
        """
        The logic is as follow:
         * All variables default to 'DEFAULT_NotAValue' upon start - validator checks and passes due to 'DEFAULT_NotAValue' being allowed
         * Upon loading config file:
            * 1st init: all user-variables are set)
            * 2nd init (this function here): remaining variables with value 'DEFAULT_NotAValue' are set to user-specified 'default_to'-dictionary
         * 'validator' takes care of validating post-init, thus all default-dict keys are validated
         comment: __attrs_post_init must be in DLENSALOT_Model, as this is the only one who knows of the default dictionary (defaults_to), and cannot simply be passed along to sub-classes.

        """

        spec = importlib.util.spec_from_file_location("default", opj(Path(__file__).parent.parent, "default/{}.py".format(self.defaults_to.replace('.py', ''))))
        default_module = importlib.util.module_from_spec(spec)
        sys.modules["default"] = default_module
        spec.loader.exec_module(default_module)
        default_dict = default_module.DL_DEFAULT
        for key, val in list(filter(lambda x: '__' not in x[0] and x[0] not in ['defaults_to', 'validate_model'], self.__dict__.items())):
            for k, v in val.__dict__.items():
                if k in ['chain', 'stepper']:
                    for ke, va in v.__dict__.items():
                        if np.all(va == DEFAULT_NotAValue):
                            if key in default_dict:
                                if k in default_dict[key]:
                                    if ke in default_dict[key][k]:
                                        self.__dict__[key].__dict__[k].__dict__.update({ke: default_dict[key][k][ke]})
                elif np.all(v == DEFAULT_NotAValue):
                    if key in default_dict:
                        if k in default_dict[key]:
                            # log.info('\t\t{}: Found default for k {}: {}'. format(key, k, default_dict[key][k]))
                            self.__dict__[key].__dict__.update({k: default_dict[key][k]})
                        else:
                            if key not in ['simulationdata']:
                                # It is ok to not have defaults for simulationdata, as the simlib will handle it
                                log.debug('{}: couldnt find matching default value for {}'.format(key, k))
                    else:
                        log.debug('couldnt find matching default value for key {}'.format(key))
                elif callable(v):
                    # Cannot evaluate functions, so hopefully they didn't change..
                    pass