.. _Configuration Files:

====================
configuration files
====================

delensalot is best used via configuration files, defining all analysis details in a structured, human-readable metamodel.
This section describes the structure of configuration files, used for running delensalot jobs.

At the end of this section, you will understand `example`_ configuration files, and know how to modify parameters,
and to write a configuration file to perfrom lensing reconstruction on your very own data.


Structure:
-------------

A complete delensalot model includes the following objects,

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
    LENSRES (float)                              resolution of the deflection operation in arcmin
    Lmin (int)                                   minimum multipole for which lensing potential is reconstructed
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
    IMIN (int)                                  minimum simulation index
    IMAX (int)                                  maximum simulation index, will autogenerate simindices between IMAX and IMIN if both set
    simidxs (array-like int)                    similar to above. Simulation indices to run delensalot jobs on
    package\_ (str)                             package name of the data (can be, e.g. 'dlensalot')
    module\_ (str)                              module name of the data (can be, e.g. dlensalot.config.example.data.ffp10) (?)
    class\_ (str)                               class name of the data (can be, e.g. cmbs4_no_foreground) 
    class_parameters (dict-like)                parameters of the class of the data                 
    data_type (str)                             data may come on spherical harmonics or real space. Can be either 'map' or 'alm'
    data_field (str)                            data may be spin-2 or spin-0. Can either be 'qu' or 'eb'
    beam (float)                                assuming a Gaussian beam, this defines the FWHM in arcmin
    lmax_transf (int)                           maxmimum multipole to apply transfer function to data
    nside (int)                                 resolution of the data
=============================================== =============================================================================


=============================================== =============================================================================
:code:`Noisemodel`                              Description
----------------------------------------------- -----------------------------------------------------------------------------
    typ (str)                                   OBD identifier. Can be 'OBD', 'trunc', or None
    BMARG_LIBDIR (str)                          path to the OBD matrix
    BMARG_LCUT (int)                            maximum multipole to deproject B-modes
    BMARG_RESCALE (float)                       rescaling of OBD matrix amplitude. Useful if matrix calculated, but noiselevel changed
    ninvjob_geometry (str)                      geometry of the noise map (?)
    lmin_tlm (int)                              minimum multipole to deproject B-modes. Modes below will be discarded completely
    lmin_elm (int)                              minimum multipole to deproject B-modes. Modes below will be discarded completely
    lmin_blm (int)                              minimum multipole to deproject B-modes. Modes below will be discarded completely
    CENTRALNLEV_UKAMIN (float)                  central noise level in muK arcmin, for both temperature and polarization. If set, temperature central noise level will be scaled by 1/sqrt(2) 
    nlev_t (float)                              central noise level of temperature data in muK arcmin. If set, overrides :code:`CENTRALNLEV_UKAMIN`
    nlev_p (float)                              central noise level of polarization data in muK arcmin. If set, overrides :code:`CENTRALNLEV_UKAMIN`
    nlev_dep (float)                            deprojection factor, or, strength of B-mode deprojection (?)
    inf (float)                                 deprojection factor, or, strength of B-mode deprojection (?)
    mask (tuple)                                OBD matrix, and noise model, will be calculated for this mask. To use an existing mask, set it to ('mask', <path1>). To create mask upon runtime tracing the noise level, set it to ('nlev', <inverse hits-count multiplier>)
    rhits_normalised (tuple)                    path to the hits-count map, used to calculate the noise levels, and the mask tracing the noise level. Second entry in tuple is the <inverse hits-count multiplier>.
    tpl (str)                                   function name for calculating OBD matrix
=============================================== =============================================================================


=============================================== =============================================================================
:code:`qerec`                                   Description
----------------------------------------------- -----------------------------------------------------------------------------
        ivfs (str)                              Inverse variance filter identifier. Can be 'sepTP' or 'jTP'
        qlms (str)                              lensing potential estimator identifier. Can be 'sepTP' or 'jTP'
        cg_tol (float)                          tolerance of the conjugate gradient method
        tasks (array-like of str)               tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
        ninvjob_qe_geometry (str)               noise model spherical harmonic geometry. Can be, e.g. 'healpix_geometry_qe' (?)
        lmmax_qlm (tuple of int)                maximum multipole coefficients to reconstruct the lensing potential
        QE_LENSING_CL_ANALYSIS (bool)           calculation of the lensing potential power spectra (?)
        chain (DLENSALOT_Chaindescriptor)       configuration of the conjugate gradient method. Configures the chain and preconditioner
=============================================== =============================================================================


=============================================== =============================================================================
:code:`itrec`                                   Description
----------------------------------------------- -----------------------------------------------------------------------------
    filter (str)                                filter identifier. Can be any class inside the :code:`dlensalot.opfilt` module
    cg_tol (float)                              tolerance of the conjugate gradient method
    tasks (array-like of str)                   tasks to perfrom. Can be any combination of :code:`calc_phi`, :code:`calc_meanfield`, :code:`calc_blt`
    lenjob_geometry (str)                       can be 'healpix_geometry', 'thin_gauss' or 'pbdGeometry'
    lenjob_pbgeometry (str)                     can be 'healpix_geometry', 'thin_gauss' or 'pbdGeometry'
    iterator_typ (str)                          mean-field handling identifier. Can be either 'const_mf' or 'pert_mf'
    mfvar (str)                                 path to precalculated mean-field, to be used instead
    soltn_cond (func)                           - (?)
    stepper (DLENSALOT_STEPPER)                 configuration about updating the current likelihood iteration point with the likelihood gradient
=============================================== =============================================================================


=============================================== =============================================================================
:code:`madel`                                   Description
----------------------------------------------- -----------------------------------------------------------------------------
    edges (array-like of str)                   binning to calculate the (delensed) power spectrum on. Can be any combination of 'ioreco', 'cmbs4'
    iterations (array-like of int)              which iterations to calculate delensed power spectrum for
    dlm_mod (tuple)                             if set, modfies the lensing potential before calculating the B-lensing template
    ringmask (bool)                             if set, use ring mask instead of full mask
    data_from_CFS (bool)                        if set, use B-lensing templates located at the $CFS directory instead of the $temp directory
    subtract_mblt (tuple)                       if set, subtract the mean-B-lensing template from the B-lensing template before delensing
    masks (tuple)                               the sky patches to calculate the power spectra on. Can either be 'masks' and a list of paths to masks, or 'nlevels' and a list of <inverse hits-count multiplier>'s
    lmax (int)                                  maximum multipole to calculate the (delensed) power spectrum
    Cl_fid (str)                                fiducial power spectrum
    binning (str)                               can be either 'binned' or 'unbinned'. If 'unbinned', overwrites :code:`edges` and calculates power spectrum for each multipole
    spectrum_calculator (package)               name of the package of the power spectrum calculator. Can be 'healpy' if :code:`binning=unbinned`
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



.. _example: https://github.com/NextGenCMB/delensalot/tree/main/delensalot/lerepi/config
