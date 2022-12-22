.. _Configuration Files:

========================
configuration files
========================

This section describes how configuration files are structured and used. At the end of this section, you will understand the settings of the example configuration files, know how to modify them for your needs,
and write a configuration file which uses your very own data.
D.lensalot is best used via configuration files, defining all analysis details in a structured, human-readable metamodel.

Some `example`_ models are provided for different use cases, among them,

* cmbs4-like setting with no foregrounds and no masking
* cmbs4-like setting with no foreground and masking
* cmbs4-like setting with foregrounds and no masking



.. _example: https://github.com/NextGenCMB/D.lensalot/tree/main/lenscarf/lerepi/config



Structure:
--------------------

A complete Dlensalot model includes the following objects,

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
    Config           General configuration
==================== ===========

A minimal working configuration file needs a :code:`Job`, :code:`Analysis`, :code:`Data`, :code:`Noisemodel`, and :code:`Config` object.

:code:`Job`
++++++++++++

* There is a predefined list of Jobs from which the user can choose. Use ``<boolean choice>`` = ``True`` or ``False``,
    * build_OBD = <boolean choice>,
    * QE_lensrec = <boolean choice>,
    * MAP_lensrec = <boolean choice>,
    * map_delensing = <boolean choice>,
    * inspect_result = <boolean choice>,


:code:`Analysis`
++++++++++++++++

:code:`Data`
++++++++++++++++

:code:`Noisemodel`
++++++++++++++++++

:code:`Config`
++++++++++++++++++


The following is an example Dlensalot model for CMBS-4 configurations, for which iterative lensing reconstruction and map delensing is chosen as Job.


..  literalinclude:: /_static/c08d_v2.py
    :language: python
    :emphasize-lines: 6-14,
    :linenos:
