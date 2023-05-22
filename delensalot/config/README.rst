lerepi
===========

CMB lensing reconstruction pipelines for various experiments (CMB-S4, PICO)
Introduces config files for user-friendliy delensalot handling


dlensalot model:
--------------------

Description of available parameters.


A complete Dlensalot config file includes,

==================== ===========
        Type         Description
-------------------- -----------
    Job              Which jobs to run
    Analysis         Analysis specific settings
    Data             Which data to use
    Noisemodel       A noisemodel to define the Wiener-filter
    Qerec            Quadratic estimator lensing reconstruction specific settings
    Itrec            Iterative delensing lensing reconstruction specific settings
    Mapdelensing     Delensing specific settings
==================== ===========