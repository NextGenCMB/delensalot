#!/usr/bin/env python

"""dlensalot_mm.py: Contains classes defining the metamodel of the Dlensalot formalism.
    The metamodel is a structured representation, with the `DLENSALOT_Model` as the main building block.
    We use the attr package. It provides handy ways of validation and defaulting.
"""

import abc, attr, psutil, os
from os.path import join as opj
from attrs import validators
import numpy as np
if "SCRATCH" not in os.environ:
    os.environ["SCRATCH"] = os.path.expanduser("~")+'/SCRATCH/'

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import delensalot
from delensalot.config.metamodel import DEFAULT_NotAValue, DL_DEFAULT
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
    p1 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p1)
    p0 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p0)
    p2 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p2)
    p3 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p3)
    p4 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p4)
    p5 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p5)
    p6 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p6)
    p7 =                    attr.ib(default=DEFAULT_NotAValue, validator=chaindescriptor.p7)

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
   
    typ =                   attr.ib(default=DEFAULT_NotAValue, validator=stepper.typ)
    lmax_qlm =              attr.ib(default=DEFAULT_NotAValue, validator=stepper.lmax_qlm) # must match lm_max_qlm -> validator
    mmax_qlm =              attr.ib(default=DEFAULT_NotAValue, validator=stepper.mmax_qlm) # must match lm_max_qlm -> validator
    a =                     attr.ib(default=DEFAULT_NotAValue, validator=stepper.a)
    b =                     attr.ib(default=DEFAULT_NotAValue, validator=stepper.b)
    xa =                    attr.ib(default=DEFAULT_NotAValue, validator=stepper.xa)
    xb =                    attr.ib(default=DEFAULT_NotAValue, validator=stepper.xb)


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.
    delensalot can executte different jobs (QE reconstruction, simulation generation, MAP reconstruction, delensing, ..) which is controlled here.

    Attributes:
        jobs (list[str]): Job identifier(s)
    """
    jobs =                  attr.ib(default=DEFAULT_NotAValue, validator=job.jobs)

@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        key (str): reconstruction estimator key
        version (str): specific configuration for the esimator (e.g. `noMF`, which turns off mean-field subtraction)
        simidxs (np.array[int]): simulation indices to use for the delensalot job
        simidxs_mf (np.array[int]): simulation indices to use for the calculation of the mean-field
        TEMP_suffix (str): identifier to customize TEMP directory of the analysis
        Lmin (int): minimum L for reconstructing the lensing potential
        zbounds (tuple[int] or tuple[str,float]): latitudinal boundary (-1 to 1), or identifier together with noise level ratio treshold at which lensing reconstruction is perfromed.
        zbounds_len (tuple[int]): latitudinal extended boundary at which lensing reconstruction is performed, and used for iterative lensing reconstruction
        pbounds (tuple[int]): longitudinal boundary at which lensing reconstruction is perfromed
        lm_max_len (tuple[int]): 
        lm_max_ivf (tuple[int]): maximum `\ell` and m for which inverse variance filtering is done
        lm_max_blt (tuple[int]): maximum `\ell` and m for which B-lensing template is calculated
        mask (list[str]): TBD
        lmin_teb (int): minimum `\ell` and m of the data which the reconstruction uses, and is set to zero below via the transfer function
        cls_unl (str): path to the fiducial unlensed CAMB-like CMB data
        cls_len (str): path to the fiducial lensed CAMB-like CMB data
        cpp (str): path to the power spectrum of the prior for the iterative reconstruction
        beam (float): 
    """
    key =                   attr.ib(default=DEFAULT_NotAValue, on_setattr=[validators.instance_of(str), analysis.key], type=str)
    version =               attr.ib(default=DEFAULT_NotAValue, on_setattr=[validators.instance_of(str), analysis.version], type=str)
    simidxs =               attr.ib(default=DEFAULT_NotAValue, on_setattr=data.simidxs)
    simidxs_mf =            attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.simidxs_mf)
    TEMP_suffix =           attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.TEMP_suffix)
    Lmin =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.Lmin)
    zbounds =               attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.zbounds)
    zbounds_len =           attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.zbounds_len)
    pbounds =               attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.pbounds)
    lm_max_len =            attr.ib(default=DEFAULT_NotAValue, on_setattr=v_filter.lm_max_len)
    lm_max_ivf =            attr.ib(default=DEFAULT_NotAValue, on_setattr=v_filter.lm_max_ivf)
    lm_max_blt =            attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.lm_max_blt)
    mask =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.mask)
    lmin_teb =              attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.lmin_teb)
    cls_unl =               attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.cls_unl)
    cls_len =               attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.cls_len)
    cpp =                   attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.cpp)
    beam =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=analysis.beam)

@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        class_parameters (type): TBD
        package_ (type): TBD
        module_ (type): TBD
        class_ (type): TBD
        transferfunction (type): TBD
        beam (type): TBD
        nside (type): TBD
        nlev_t (type): TBD
        nlev_p (type): TBD
        lmax_transf (type): TBD
        epsilon (type): TBD
    """

    class_parameters =      attr.ib(default=DEFAULT_NotAValue, on_setattr=data.class_parameters)
    package_ =              attr.ib(default=DEFAULT_NotAValue, on_setattr=data.package_)
    module_ =               attr.ib(default=DEFAULT_NotAValue, on_setattr=data.module_)
    class_ =                attr.ib(default=DEFAULT_NotAValue, on_setattr=data.class_)
    transferfunction =      attr.ib(default=DEFAULT_NotAValue, on_setattr=data.transferfunction)
    beam =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=data.beam)
    nside =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=data.nside)
    nlev_t =                attr.ib(default=DEFAULT_NotAValue, on_setattr=data.nlev_t)
    nlev_p =                attr.ib(default=DEFAULT_NotAValue, on_setattr=data.nlev_p)
    lmax_transf =           attr.ib(default=DEFAULT_NotAValue, on_setattr=data.lmax_transf)
    epsilon =               attr.ib(default=DEFAULT_NotAValue, on_setattr=data.epsilon)

    
@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        sky_coverage (type): TBD
        spectrum_type (type): TBD
        OBD (type): TBD
        nlev_t (type): TBD
        nlev_p (type): TBD
        rhits_normalised (type): TBD
        ninvjob_geometry (type): TBD
    """
    sky_coverage =          attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.sky_coverage)
    spectrum_type =         attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.spectrum_type)
    OBD =                   attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.OBD)
    nlev_t =                attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.nlev_t)
    nlev_p =                attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.nlev_p)
    rhits_normalised =      attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.rhits_normalised)
    ninvjob_geometry =      attr.ib(default=DEFAULT_NotAValue, on_setattr=noisemodel.ninvjob_geometry)

@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        tasks (type): TBD
        qlm_type (type): TBD
        cg_tol (type): TBD
        filter_directional (type): TBD
        ninvjob_qe_geometry (type): TBD
        lm_max_qlm (type): TBD
        chain (type): TBD
        cl_analysis (type): TBD
        blt_pert (type): TBD
    """
    tasks =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.tasks)
    qlm_type =              attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.qlms)
    cg_tol =                attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.cg_tol)
    filter_directional =    attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.filter_directional)
    ninvjob_qe_geometry =   attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.ninvjob_qe_geometry)
    lm_max_qlm =            attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.lm_max_qlm) # TODO qe.lm_max_qlm and it.lm_max_qlm must be same. Test at validator?
    chain =                 attr.ib(default=DLENSALOT_Chaindescriptor(), on_setattr=qerec.chain)
    cl_analysis =           attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.cl_analysis)
    blt_pert =              attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.btemplate_perturbative_lensremap)

@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        tasks (type): TBD
        itmax (type): TBD
        cg_tol (type): TBD
        iterator_typ (type): TBD
        chain (type): TBD
        filter_directional (type): TBD
        lenjob_geometry (type): TBD
        lenjob_pbgeometry (type): TBD
        lm_max_unl (type): TBD
        lm_max_qlm (type): TBD
        mfvar (type): TBD
        soltn_cond (type): TBD
        stepper (type): TBD
    """
    tasks =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.tasks)
    itmax =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.itmax)
    cg_tol =                attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.cg_tol)
    iterator_typ =          attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.iterator_type)
    chain =                 attr.ib(default=DLENSALOT_Chaindescriptor(), on_setattr=itrec.chain)
    filter_directional =    attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.filter_directional)
    lenjob_geometry =       attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.lenjob_geometry)
    lenjob_pbgeometry =     attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.lenjob_pbgeometry)
    lm_max_unl =            attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.lm_max_unl)
    lm_max_qlm =            attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.lm_max_qlm)
    mfvar =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.mfvar)
    soltn_cond =            attr.ib(default=DEFAULT_NotAValue, on_setattr=itrec.soltn_cond)
    stepper =               attr.ib(default=DLENSALOT_Stepper(), on_setattr=itrec.stepper)
    
@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Attributes:
        data_from_CFS (type): TBD
        edges (type): TBD
        dlm_mod (type): TBD
        iterations (type): TBD
        nlevels (type): TBD
        lmax (type): TBD
        Cl_fid (type): TBD
        libdir_it (type): TBD
        binning (type): TBD
        spectrum_calculator (type): TBD
        masks_fn (type): TBD
    """
    data_from_CFS =         attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.data_from_CFS)
    edges =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.edges)
    dlm_mod =               attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.dlm_mod)
    iterations =            attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.iterations)
    nlevels =               attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.nlevels)
    lmax =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.lmax)
    Cl_fid =                attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.Cl_fid)
    libdir_it =             attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.libdir_it)
    binning =               attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.binning)
    spectrum_calculator =   attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.spectrum_calculator)
    masks_fn =              attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.masks)

@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        libdir (type): TBD
        rescale (type): TBD
        tpl (type): TBD
        nlev_dep (type): TBD
        nside (type): TBD
        lmax (type): TBD
        beam (type): TBD
    """
    libdir =                attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.libdir)
    rescale =               attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.rescale)
    tpl =                   attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.tpl)
    nlev_dep =              attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.nlev_dep)
    nside =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.nside)
    lmax =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.lmax)
    beam =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=obd.beam)

@attr.s
class DLENSALOT_Config(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        outdir_plot_root (type): TBD
        outdir_plot_rel (type): TBD
    """
    outdir_plot_root =      attr.ib(default=opj(os.environ['HOME'], 'plots'))
    outdir_plot_rel =       attr.ib(default='')

@attr.s
# @add_defaults
class DLENSALOT_Meta(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        version (str): version control of the delensalot model
    """
    version =               attr.ib(default=DEFAULT_NotAValue, on_setattr=attr.validators.instance_of(int))


@attr.s
class DLENSALOT_Computing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        OMP_NUM_THREADS (int): number of threads used per Job
    """
    OMP_NUM_THREADS =       attr.ib(default=DEFAULT_NotAValue, on_setattr=computing.OMP_NUM_THREADS)


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        defaults_to (type): TBD
        meta (type): TBD
        job (type): TBD
        analysis (type): TBD
        data (type): TBD
        noisemodel (type): TBD
        qerec (type): TBD
        itrec (type): TBD
        madel (type): TBD
        config (type): TBD
        computing (type): TBD
        obd (type): TBD
    """
    
    defaults_to =           attr.ib(default='P_FS_CMBS4')
    meta =                  attr.ib(default=DLENSALOT_Meta(), on_setattr=model.meta)
    job =                   attr.ib(default=DLENSALOT_Job(), on_setattr=model.job)
    analysis =              attr.ib(default=DLENSALOT_Analysis(), on_setattr=model.analysis)
    data  =                 attr.ib(default=DLENSALOT_Data(), on_setattr=model.data)
    noisemodel =            attr.ib(default=DLENSALOT_Noisemodel(), on_setattr=model.noisemodel)
    qerec =                 attr.ib(default=DLENSALOT_Qerec(), on_setattr=model.qerec)
    itrec =                 attr.ib(default=DLENSALOT_Itrec(), on_setattr=model.itrec)
    madel =                 attr.ib(default=DLENSALOT_Mapdelensing(), on_setattr=model.madel)
    config =                attr.ib(default=DLENSALOT_Config(), on_setattr=model.config)
    computing =             attr.ib(default=DLENSALOT_Computing(), on_setattr=model.computing)
    obd =                   attr.ib(default=DLENSALOT_OBD(), on_setattr=model.obd)
    

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



        