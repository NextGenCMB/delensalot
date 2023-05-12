#!/usr/bin/env python

"""dlensalot_mm.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc, attr, psutil, os
if "SCRATCH" not in os.environ:
    os.environ["SCRATCH"] = os.path.expanduser("~")+'/SCRATCH/'
    
from attrs import validators


from os.path import join as opj
import numpy as np

from plancklens import utils

import delensalot
from delensalot.lerepi.core.metamodel import DEFAULT_NotAValue, DL_DEFAULT
from delensalot.lerepi.core.validator import analysis, chaindescriptor, computing, data, filter as v_filter, itrec, job, mapdelensing, meta, model, noisemodel, obd, qerec, stepper



class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


    def __str__(self):
        ## overwrite print to summarize dlensalot model
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
        p0: 
    """
    p1 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p1)
    p0 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p0)
    p2 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p2)
    p3 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p3)
    p4 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p4)
    p5 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p5)
    p6 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p6)
    p7 =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=chaindescriptor.p7)

@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ =                   attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.typ)
    lmax_qlm =              attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.lmax_qlm)
    mmax_qlm =              attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.mmax_qlm)
    a =                     attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.a)
    b =                     attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.b)
    xa =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.xa)
    xb =                    attr.ib(default=DEFAULT_NotAValue, on_setattr=stepper.xb)


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    jobs =                  attr.ib(default=DEFAULT_NotAValue, validator=job.jobs)

@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
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
        DATA_LIBDIR: path to the data
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
        typ:
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
        typ:
    """
    tasks =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.tasks)
    qlm_type =              attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.qlms)
    cg_tol =                attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.cg_tol)
    filter_directional =    attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.filter_directional)
    ninvjob_qe_geometry =   attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.ninvjob_qe_geometry)
    lm_max_qlm =            attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.lm_max_qlm)
    chain =                 attr.ib(default=DLENSALOT_Chaindescriptor(), on_setattr=qerec.chain)
    cl_analysis =           attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.cl_analysis)
    blt_pert =              attr.ib(default=DEFAULT_NotAValue, on_setattr=qerec.btemplate_perturbative_lensremap)

@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
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

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.edges)
    dlm_mod =               attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.dlm_mod)
    iterations =            attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.iterations)
    masks =                 attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.masks)
    lmax =                  attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.lmax)
    Cl_fid =                attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.Cl_fid)
    libdir_it =             attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.libdir_it)
    binning =               attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.binning)
    spectrum_calculator =   attr.ib(default=DEFAULT_NotAValue, on_setattr=mapdelensing.spectrum_calculator)

@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        BMARG_LIBDIR:
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
        typ:
    """
    outdir_plot_root =      attr.ib(default=opj(os.environ['HOME'], 'plots'))
    outdir_plot_rel =       attr.ib(default='')

@attr.s
# @add_defaults
class DLENSALOT_Meta(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        version:
    """
    version =               attr.ib(default=DEFAULT_NotAValue, on_setattr=attr.validators.instance_of(int))


@attr.s
class DLENSALOT_Computing(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    OMP_NUM_THREADS =       attr.ib(default=DEFAULT_NotAValue, on_setattr=computing.OMP_NUM_THREADS)


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    
    defaults_to =           attr.ib(default='default')
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

        """
        print("Setting default, using {}:\n\t{}".format(self.defaults_to, DL_DEFAULT[self.defaults_to]))
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
                                print('couldnt find matching default value for k {}'.format(key))
                        else:
                            print('couldnt find matching default value for key {}'.format(key))



        