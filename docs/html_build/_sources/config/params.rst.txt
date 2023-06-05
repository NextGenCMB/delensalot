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
    Stepper          iterative reconstruction likelihood search settings
    Chaindescriptor  Conjugate gradient solver settings
    Config           General configuration
==================== ===========


The following shows the delensalot metamodel and its attributes with the docstring describing the purpose of it.
`Defaults`_ and valid values for each are defined via `validators`_ and may be found in the API.

.. _`Defaults`: https://github.com/NextGenCMB/delensalot/blob/main/delensalot/config/metamodel/__init__.py
.. _`validators`: https://github.com/NextGenCMB/delensalot/blob/main/delensalot/config/validator/

..  literalinclude:: ../../../delensalot/config/metamodel/dlensalot_mm.py
    :language: python
    :linenos:



`example`_ models are provided for different use cases, among them,

* cmbs4-like setting with no foregrounds and no masking
* cmbs4-like setting with no foreground and masking
* cmbs4-like setting with foregrounds and no masking



.. _example: https://github.com/NextGenCMB/delensalot/tree/main/first_steps/notebooks/
