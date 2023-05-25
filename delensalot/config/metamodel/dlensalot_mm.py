#!/usr/bin/env python

"""dlensalot_mm.py: Contains classes defining the metamodel of the Dlensalot formalism.
    The metamodel is a structured representation, with the `DLENSALOT_Model` as the main building block.
    We use the attr package. It provides handy ways of validation and defaulting.
"""

import abc, attr, os
from os.path import join as opj
from attrs import validators
import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from delensalot.config.metamodel import DEFAULT_NotAValue, DEFAULT_NotASTR, DL_DEFAULT
from delensalot.config.validator import analysis, chaindescriptor, computing, data, filter as v_filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper


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
    This class collects all configurations related to conjugate gradient solver.

    Attributes:
        p0: TBD
        p1: TBD
        p2: TBD
        p3: TBD
        p4: TBD
        p5: TBD
        p6: TBD
        p7: TBD
    """
    p0 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p0)
    p1 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p1)
    p2 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p2)
    p3 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p3)
    p4 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p4)
    p5 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p5)
    p6 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p6)
    p7 =                    attr.field(default=DEFAULT_NotAValue, validator=chaindescriptor.p7)

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
   
    typ =                   attr.field(default=DEFAULT_NotAValue, validator=stepper.typ)
    lmax_qlm =              attr.field(default=DEFAULT_NotAValue, validator=stepper.lmax_qlm) # must match lm_max_qlm -> validator
    mmax_qlm =              attr.field(default=DEFAULT_NotAValue, validator=stepper.mmax_qlm) # must match lm_max_qlm -> validator
    a =                     attr.field(default=DEFAULT_NotAValue, validator=stepper.a)
    b =                     attr.field(default=DEFAULT_NotAValue, validator=stepper.b)
    xa =                    attr.field(default=DEFAULT_NotAValue, validator=stepper.xa)
    xb =                    attr.field(default=DEFAULT_NotAValue, validator=stepper.xb)


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here.

    Attributes:
        jobs (list[str]): Job identifier(s)
    """
    jobs =                  attr.field(default=DEFAULT_NotAValue, validator=job.jobs)

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
        pbounds (tuple[int]):               longitudinal boundary at which lensing reconstruction is perfromed
        lm_max_len (tuple[int]):            TODO: TBD (deprecated?)
        lm_max_ivf (tuple[int]):            maximum `\ell` and m for which inverse variance filtering is done
        lm_max_blt (tuple[int]):            maximum `\ell` and m for which B-lensing template is calculated
        mask (list[str]):                   TBD
        lmin_teb (int):                     minimum `\ell` and m of the data which the reconstruction uses, and is set to zero below via the transfer function
        cls_unl (str):                      path to the fiducial unlensed CAMB-like CMB data
        cls_len (str):                      path to the fiducial lensed CAMB-like CMB data
        cpp (str):                          path to the power spectrum of the prior for the iterative reconstruction
        beam (float):                       The beam used in the filters
    """
    key =                   attr.field(default=DEFAULT_NotAValue, on_setattr=[validators.instance_of(str), analysis.key], type=str)
    version =               attr.field(default=DEFAULT_NotAValue, on_setattr=[validators.instance_of(str), analysis.version], type=str)
    reconstruction_method = attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.reconstruction_method)
    simidxs =               attr.field(default=DEFAULT_NotAValue, on_setattr=data.simidxs)
    simidxs_mf =            attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.simidxs_mf)
    TEMP_suffix =           attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.TEMP_suffix)
    Lmin =                  attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.Lmin)
    zbounds =               attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.zbounds)
    zbounds_len =           attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.zbounds_len)
    pbounds =               attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.pbounds)
    lm_max_len =            attr.field(default=DEFAULT_NotAValue, on_setattr=v_filter.lm_max_len)
    lm_max_ivf =            attr.field(default=DEFAULT_NotAValue, on_setattr=v_filter.lm_max_ivf)
    lm_max_blt =            attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.lm_max_blt)
    mask =                  attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.mask)
    lmin_teb =              attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.lmin_teb)
    cls_unl =               attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.cls_unl)
    cls_len =               attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.cls_len)
    cpp =                   attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.cpp)
    beam =                  attr.field(default=DEFAULT_NotAValue, on_setattr=analysis.beam)

@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the input CMB maps.

    Attributes:
        class_parameters (dic): parameters of the class of the data
        package_ (str):         package name of the data (can be, e.g. 'delensalot')
        module_ (str):      	module name of the data (can be, e.g. sims.generic) (?)
        class_ (str):           class name of the data (can be, e.g. sims_cmb_len) 
        transferfunction (str): predefined isotropic transfer function. Can bei either with or without pixelwindow function applied
        beam (float):           assuming a Gaussian beam, this defines the FWHM in arcmin
        nside (int):            resolution of the data
        nlev_t (type):          TBD
        nlev_p (type):          TBD
        lmax_transf (int):      maxmimum multipole to apply transfer function to data
        epsilon (float):        lenspyx precision    

    comment:
        data_type (str)         data may come on spherical harmonics or real space. Can be either 'map' or 'alm'
        data_field (str)        data may be spin-2 or spin-0. Can either be 'qu' or 'eb'                                                  
    """

    class_parameters =      attr.field(default=DEFAULT_NotAValue, on_setattr=data.class_parameters)
    package_ =              attr.field(default=DEFAULT_NotAValue, on_setattr=data.package_)
    module_ =               attr.field(default=DEFAULT_NotAValue, on_setattr=data.module_)
    class_ =                attr.field(default=DEFAULT_NotAValue, on_setattr=data.class_)
    transferfunction =      attr.field(default=DEFAULT_NotAValue, on_setattr=data.transferfunction)
    beam =                  attr.field(default=DEFAULT_NotAValue, on_setattr=data.beam)
    nside =                 attr.field(default=DEFAULT_NotAValue, on_setattr=data.nside)
    nlev_t =                attr.field(default=DEFAULT_NotAValue, on_setattr=data.nlev_t)
    nlev_p =                attr.field(default=DEFAULT_NotAValue, on_setattr=data.nlev_p)
    lmax_transf =           attr.field(default=DEFAULT_NotAValue, on_setattr=data.lmax_transf)
    epsilon =               attr.field(default=DEFAULT_NotAValue, on_setattr=data.epsilon)
    maps =                  attr.field(default=DEFAULT_NotAValue, on_setattr=data.maps)
    phi =                   attr.field(default=DEFAULT_NotAValue, on_setattr=data.phi)

    
@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the noise model used for Wiener-filtering the data.

    Attributes:
        sky_coverage (str):     Can be either 'masked' or 'unmasked'
        spectrum_type (str):    TBD
        OBD (str):              OBD identifier. Can be 'OBD', 'trunc', or None. Defines how lowest B-modes will be handled.
        nlev_t (float):         (central) noise level of temperature data in muK arcmin.
        nlev_p (float):         (central) noise level of polarization data in muK arcmin.
        rhits_normalised (str): path to the hits-count map, used to calculate the noise levels, and the mask tracing the noise level. Second entry in tuple is the <inverse hits-count multiplier>.
        ninvjob_geometry (str): geometry of the noise map
    """
    sky_coverage =          attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.sky_coverage)
    spectrum_type =         attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.spectrum_type)
    OBD =                   attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.OBD)
    nlev_t =                attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.nlev_t)
    nlev_p =                attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.nlev_p)
    rhits_normalised =      attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.rhits_normalised)
    ninvjob_geometry =      attr.field(default=DEFAULT_NotAValue, on_setattr=noisemodel.ninvjob_geometry)

@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the quadratic estimator reconstruction job.

    Attributes:
        tasks (list[tuple]):        tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        qlm_type (str):             lensing potential estimator identifier. Can be 'sepTP' or 'jTP'
        cg_tol (float):             tolerance of the conjugate gradient method
        filter_directional (str):   can be either 'isotropic' (unmasked sky) or 'isotropic' (masked sky)
        ninvjob_qe_geometry (str):  noise model spherical harmonic geometry. Can be, e.g. 'healpix_geometry_qe' (?)
        lm_max_qlm (type):          maximum multipole `\ell` and m to reconstruct the lensing potential
        chain (DLENSALOT_Chaindescriptor): configuration of the conjugate gradient method. Configures the chain and preconditioner
        cl_analysis (bool):         If tru, performs lensing power spectrum analysis
        blt_pert (bool):            If True, delensing is performed perurbitivly (recommended)
    
    """

    tasks =                 attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.tasks)
    qlm_type =              attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.qlms)
    cg_tol =                attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.cg_tol)
    filter_directional =    attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.filter_directional)
    ninvjob_qe_geometry =   attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.ninvjob_qe_geometry)
    lm_max_qlm =            attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.lm_max_qlm) # TODO qe.lm_max_qlm and it.lm_max_qlm must be same. Test at validator?
    chain =                 attr.field(default=DLENSALOT_Chaindescriptor(), on_setattr=qerec.chain)
    cl_analysis =           attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.cl_analysis)
    blt_pert =              attr.field(default=DEFAULT_NotAValue, on_setattr=qerec.btemplate_perturbative_lensremap)

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
        lenjob_geometry (str):      can be 'healpix_geometry', 'thin_gauss' or 'pbdGeometry'
        lenjob_pbgeometry (str):    can be 'healpix_geometry', 'thin_gauss' or 'pbdGeometry'
        lm_max_unl (tuple[int]):    maximum multipoles `\ell` and m for reconstruction the unlensed CMB
        lm_max_qlm (tuple[int]):    maximum multipoles L and m for reconstruction the lensing potential
        mfvar (str):                path to precalculated mean-field, to be used instead
        soltn_cond (type):          TBD
        stepper (DLENSALOT_STEPPER):configuration for updating the current likelihood iteration point with the likelihood gradient
              
    """
    tasks =                 attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.tasks)
    itmax =                 attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.itmax)
    cg_tol =                attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.cg_tol)
    iterator_typ =          attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.iterator_type)
    chain =                 attr.field(default=DLENSALOT_Chaindescriptor(), on_setattr=itrec.chain)
    filter_directional =    attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.filter_directional)
    lenjob_geometry =       attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.lenjob_geometry)
    lenjob_pbgeometry =     attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.lenjob_pbgeometry)
    lm_max_unl =            attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.lm_max_unl)
    lm_max_qlm =            attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.lm_max_qlm)
    mfvar =                 attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.mfvar)
    soltn_cond =            attr.field(default=DEFAULT_NotAValue, on_setattr=itrec.soltn_cond)
    stepper =               attr.field(default=DLENSALOT_Stepper(), on_setattr=itrec.stepper)
    
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

    data_from_CFS =         attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.data_from_CFS)
    edges =                 attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.edges)
    dlm_mod =               attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.dlm_mod)
    iterations =            attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.iterations)
    nlevels =               attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.nlevels)
    lmax =                  attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.lmax)
    Cl_fid =                attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.Cl_fid)
    libdir_it =             attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.libdir_it)
    binning =               attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.binning)
    spectrum_calculator =   attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.spectrum_calculator)
    masks_fn =              attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.masks)
    basemap =               attr.field(default=DEFAULT_NotAValue, on_setattr=mapdelensing.basemap)

@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the overlapping B-mode deprojection.

    Attributes:
        libdir (str):       path to the OBD matrix
        rescale (float):    rescaling of OBD matrix amplitude. Useful if matrix already calculated, but noiselevel changed
        tpl (type):         function name for calculating OBD matrix
        nlev_dep (float):   deprojection factor, or, strength of B-mode deprojection
        nside (type):       TBD
        lmax (int):         maximum multipole to deproject B-modes
        beam (type):        TBD                         
    """
    libdir =                attr.field(default=DEFAULT_NotAValue, on_setattr=obd.libdir)
    rescale =               attr.field(default=DEFAULT_NotAValue, on_setattr=obd.rescale)
    tpl =                   attr.field(default=DEFAULT_NotAValue, on_setattr=obd.tpl)
    nlev_dep =              attr.field(default=DEFAULT_NotAValue, on_setattr=obd.nlev_dep)
    nside =                 attr.field(default=DEFAULT_NotAValue, on_setattr=obd.nside)
    lmax =                  attr.field(default=DEFAULT_NotAValue, on_setattr=obd.lmax)
    beam =                  attr.field(default=DEFAULT_NotAValue, on_setattr=obd.beam)

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
# @add_defaults
class DLENSALOT_Meta(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to internal behaviour of delensalot.

    Attributes:
        version (str):  version control of the delensalot model
    """
    version =               attr.field(default=DEFAULT_NotAValue, on_setattr=attr.validators.instance_of(int))


@attr.s
class DLENSALOT_Computing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    This class collects all configurations related to the usage of computing resources.

    Attributes:
        OMP_NUM_THREADS (int):  number of threads used per Job
    """
    OMP_NUM_THREADS =       attr.field(default=DEFAULT_NotAValue, on_setattr=computing.OMP_NUM_THREADS)


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        defaults_to (str):              Identifier for default-dictionary if user hasn't specified value in configuration file
        meta (DLENSALOT_Meta):          configurations related to internal behaviour of delensalot
        job (DLENSALOT_Job):            delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here
        analysis (DLENSALOT_Analysis):  configurations related to the specific analysis performed on the data
        data (DLENSALOT_Data):          configurations related to the input CMB maps
        noisemodel (DLENSALOT_Noisemodel):  configurations related to the noise model used for Wiener-filtering the data
        qerec (DLENSALOT_Qerec):        configurations related to the quadratic estimator reconstruction job
        itrec (DLENSALOT_Itrec):        configurations related to the iterative reconstruction job
        madel (DLENSALOT_Mapdelensing): configurations related to the internal map delensing job
        config (DLENSALOT_Config):      configurations related to general behaviour to the operating system
        computing (DLENSALOT_Computing):    configurations related to the usage of computing resources
        obd (DLENSALOT_OBD):            configurations related to the overlapping B-mode deprojection

    """
    
    defaults_to =           attr.field(default='P_FS_CMBS4')
    meta =                  attr.field(default=DLENSALOT_Meta(), on_setattr=model.meta)
    job =                   attr.field(default=DLENSALOT_Job(), on_setattr=model.job)
    analysis =              attr.field(default=DLENSALOT_Analysis(), on_setattr=model.analysis)
    data  =                 attr.field(default=DLENSALOT_Data(), on_setattr=model.data)
    noisemodel =            attr.field(default=DLENSALOT_Noisemodel(), on_setattr=model.noisemodel)
    qerec =                 attr.field(default=DLENSALOT_Qerec(), on_setattr=model.qerec)
    itrec =                 attr.field(default=DLENSALOT_Itrec(), on_setattr=model.itrec)
    madel =                 attr.field(default=DLENSALOT_Mapdelensing(), on_setattr=model.madel)
    config =                attr.field(default=DLENSALOT_Config(), on_setattr=model.config)
    computing =             attr.field(default=DLENSALOT_Computing(), on_setattr=model.computing)
    obd =                   attr.field(default=DLENSALOT_OBD(), on_setattr=model.obd)
    

    def __attrs_post_init__(self):
        """
        The logic is as follow:
         * All variables default to 'DEFAULT_NotAValue' upon start - validator checks and passes due to 'DEFAULT_NotAValue' being allowed
         * Upon loading config file:
            * 1st init: all user-variables are set, validator checks
            * 2nd init (this function here): remaining variables with value 'DEFAULT_NotAValue' are set to user-specified 'default_to'-dictionary
         * 'on_setattr' takes care of validating post-init, thus all default-dict keys are validated
         comment: __attrs_post_init must be in DLENSALOT_Model, as this is the only one who knows of the default dictionary (defaults_to), and cannot simply be passed along to sub-classes.

        """
        log.info("Setting default, using {}:\n\t{}".format(self.defaults_to, DL_DEFAULT[self.defaults_to]))
        for key, val in list(filter(lambda x: '__' not in x[0] and x[0] != 'defaults_to', self.__dict__.items())):
            for k, v in val.__dict__.items():
                if k in ['chain', 'stepper']:
                    for ke, va in v.__dict__.items():
                        if type(va) == type(DEFAULT_NotAValue):
                            if key in DL_DEFAULT[self.defaults_to]:
                                if k in DL_DEFAULT[self.defaults_to][key]:
                                    if ke in DL_DEFAULT[self.defaults_to][key][k]:
                                        self.__dict__[key].__dict__[k].__dict__.update({ke: DL_DEFAULT[self.defaults_to][key][k][ke]})
                                        # print('\t{}={}'.format(ke, DL_DEFAULT[self.defaults_to][key][k][ke]))
                elif type(v) == type(DEFAULT_NotAValue):
                    if v == DEFAULT_NotAValue:
                        # print('found item which needs replacing: {} = {}'.format(k, v))
                        if key in DL_DEFAULT[self.defaults_to]:
                            if k in DL_DEFAULT[self.defaults_to][key]:
                                self.__dict__[key].__dict__.update({k: DL_DEFAULT[self.defaults_to][key][k]})
                                # print('\t{}={}'.format(k, DL_DEFAULT[self.defaults_to][key][k]))
                            else:
                                log.info('couldnt find matching default value for k {}'.format(key))
                        else:
                            log.info('couldnt find matching default value for key {}'.format(key))



        