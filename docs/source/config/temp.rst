.. _analysis results:

===================
Analysis results
===================

:code:`delensalot` uses an analysis result directory :code:`TEMP` to store the analysis setting, intermediate and final results.
Among those are files related to the noise model, QE and MAP filtering, QE and MAP lensing potentials, and mean-fields.

At the root of :code:`TEMP` is the the configuration file used in this analysis.

For the noise model :code:`delensalot` stores, 

 * Noise inverted power spectra for both T and P in :code:`cinv_t` and :code:`cinv_p` 

The QE TEMP dir is :code:`TEMP/QE/`. For QE lensing reconstruction,

 * lensing potentials in :code:`qlms_dd`
 * mean-field in :code:`qlms_dd`
 * inverse variance filtered maps in :code:`ivfs`

The MAP TEMP dir is :code:`TEMP/MAP/`. 
For MAP lensing reconstruction, a realization dependent subdirectory :code:`<key>_sim<simidx><V>` is made and therefore depends on the users choice of :code:`<key>` and :code:`<V>` in the :ref:`Configuration Files`,
 
 * a subdirectory :code:`hessian` with the hessian for each iteration.
 * a subdirectory :code:`wflms` with the Wiener-filtered lensing potential increments for each iteration and B-lensing templates
 * the QE mean-field :code:`mf.npy`
 * the realization dependend QE lensing potential :code:`phi_plm_it000.npy` (with the normalized QE meanfield-subtracted from it)