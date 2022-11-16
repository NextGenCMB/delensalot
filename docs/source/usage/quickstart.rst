============
Quickstart
============


Type :code:`python3 run.py [-h]` for quickhelp,

.. code-block:: bash
    
    usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

    D.lensalot entry point.

    optional arguments:
    -h, --help            show this help message and exit
    -p NEW                Relative path to config file to run analysis.
    -r RESUME             Absolute path to config file to resume.
    -s STATUS             Absolute path for the analysis to write a report.


Configuration File
--------------------

To run a configutation file :code:`<path-to-config>`, type in your favourite :code:`bash`,

.. code-block:: bash

    python3 run.py -p <path-to-config>

:code:`<path-to-config>` is a relative path, pointing to a config file in :code:`lenscarf/lerepi/config/`.

For example,

.. code-block:: bash

    python3 run.py -p examples/example_c08b.py

runs the example configuration for :code:`c08b`. See [lenscarf/lerepi/README](https://github.com/NextGenCMB/lenscarf/blob/f/mergelerepi/lenscarf/lerepi/README.rst) for a description of the configuation parameters

If you already have an analysis, located at `$path`, with config file `conf.py`, you may resume this analysis with,

.. code-block:: bash

    python3 run.py -r $path/conf.py


If you'd like to know the status of the analysis done with :code:`$path/conf.py`, run,

.. code-block:: bash

    python3 run.py -s $path/conf.py


Interactive Mode
--------------------

D.lensalot supports interactive mode. See :code:`lenscarf/notebooks/interactive.ipynb` for guidance.