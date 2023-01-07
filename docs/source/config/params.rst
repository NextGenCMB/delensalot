.. _Configuration Files:

====================
configuration files
====================

D.lensalot is best used via configuration files, defining all analysis details in a structured, human-readable metamodel.
This section describes the structure of configuration files, used for running D.lensalot jobs.

At the end of this section, you will understand `example`_ configuration files, and know how to modify parameters,
and to write a configuration file to perfrom lensing reconstruction on your very own data.


Structure:
-------------

A complete D.lensalot model includes the following objects,

==================== ===========
        Type         Description
-------------------- -----------
    Job              jobs to run for this analysis
    Analysis         Analysis settings
    Data             Data/simulation settings
    Noisemodel       A noisemodel to define the Wiener-filter
    Qerec            Quadratic estimator lensing reconstruction settings
    Itrec            Iterative lensing reconstruction settings
    Mapdelensing     Delensing settings
    Config           General configuration
==================== ===========

A minimal working configuration file includes a :code:`Job`, :code:`Analysis`, :code:`Data`, :code:`Noisemodel`, and :code:`Config` object.


The following tables describe the individual parameters.
Default and valid values for each are defined via validators and may be found in the API.

==================== =============================================================================
    :code:`Job`          Description
-------------------- -----------------------------------------------------------------------------
    build_OBD        Calculates the Overlapping B-Mode Deprojection matrix from the noise model
    QE_lensrec       Calculates the Quadratic estimator lensing potential
    MAP_lensrec      Calculates the iterative estimator lensing potential
    map_delensing    Calculates the internally delensed power spectra using B-lensing templates
    inspect_result   Provides convenience functions for accessing (intermediate) results
==================== =============================================================================


=============================================== =============================================================================
:code:`Analysis`                                Description
----------------------------------------------- -----------------------------------------------------------------------------
    K (str)                                      source-identifier
                                                  'p_p' for polarization-only, 'ptt' for Temperatur-only, 'p_eb' for EB-only, etc
    V (str)                                      iterator-identifier
                                                  'noMF' will not use any mean-field at the very first step
    ITMAX (int)                                  maximum number of iterations 
    simidxs_mf (array-like int)                      simulation indices used for the calculation of the mean-field(s)
    LENSRES (float)                              Resolution of the deflection operation in arcmin
    Lmin (int)                                   Minimum multipole for which lensing potential is reconstructed
    lmax_filt (int)                              max multipole for the filtering step 
    lmrange_unl (array-like int, shape=(4,))     l,m-range for reconstruction of unlensed CMB (applies to MAP only)
    lmrange_ivf (array-like int, shape=(4,))     l,m-range for applying inverse variance filtering
    zbounds (array-like float, shape=(2,))       z-axis bounds at which lensing reconstruction is performed, zbounds :math:` [-1,1]`
    zbounds_len (array-like float, shape=(2,))   z-axis bounds at which lensing reconstruction is performed, zbounds :math:` [-1,1]`
    pbounds (array-like float, shape=(2,))       y-axis bounds at which lensing reconstruction is performed, pbounds :math:` [-1,1]`
    STANDARD_TRANSFERFUNCTION (boolean)          data transferfunction
                                                  if 'True', uses Gaussian beam and pixel window function. 'False' to be implemented  
=============================================== =============================================================================

=============================================== =============================================================================
:code:`Data`                                    Description
----------------------------------------------- -----------------------------------------------------------------------------
    IMIN (int)                                  
    IMAX (int)                                  
    simidxs (array-like int)                    
    package\_ (str)                              
    module\_ (str)                               
    class\_ (str)                                
    class_parameters (dict-like)                                    
    data_type (str)
    data_field (str)
    beam (float)
    lmax_transf (int)
    nside (int)
=============================================== =============================================================================


=============================================== =============================================================================
:code:`Noisemodel`                              Description
----------------------------------------------- -----------------------------------------------------------------------------
    typ
    BMARG_LIBDIR
    BMARG_LCUT
    BMARG_RESCALE
    ninvjob_geometry
    lmin_tlm
    lmin_elm
    lmin_blm
    CENTRALNLEV_UKAMIN
    nlev_t
    nlev_p
    nlev_dep
    inf
    mask
    rhits_normalised
    tpl
=============================================== =============================================================================


=============================================== =============================================================================
:code:`qerec`                                   Description
----------------------------------------------- -----------------------------------------------------------------------------
        ivfs
        qlms
        cg_tol
        tasks                                   ["calc_phi", "calc_meanfield", "calc_blt"],
        ninvjob_qe_geometry
        lmax_qlm
        mmax_qlm
        QE_LENSING_CL_ANALYSIS
        chain = DLENSALOT_Chaindescriptor
=============================================== =============================================================================


=============================================== =============================================================================
:code:`itrec`                                   Description
----------------------------------------------- -----------------------------------------------------------------------------
    filter
    cg_tol
    tasks                                        ["calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
    lenjob_geometry
    lenjob_pbgeometry
    iterator_typ
    mfvar
    soltn_cond
    stepper
=============================================== =============================================================================

The following is an example Dlensalot model for CMBS-4 configurations, for which iterative lensing reconstruction and map delensing is chosen as Job.


..  literalinclude:: /_static/c08d_v2.py
    :language: python
    :emphasize-lines: 6-14,
    :linenos:


More `example`_ models are provided for different use cases, among them,

* cmbs4-like setting with no foregrounds and no masking
* cmbs4-like setting with no foreground and masking
* cmbs4-like setting with foregrounds and no masking



.. _example: https://github.com/NextGenCMB/D.lensalot/tree/main/lenscarf/lerepi/config
