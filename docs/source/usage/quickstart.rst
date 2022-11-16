============
Quickstart
============


Type :code:`python3 lenscarf/run.py [-h]` for quickhelp,

.. code-block:: text
    
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

runs the example configuration for :code:`c08b`.

If you already have an analysis, located at `$path`, with config file `conf.py`, you may resume this analysis with,

.. code-block:: bash

    python3 run.py -r $path/conf.py


If you'd like to know the status of the analysis done with :code:`$path/conf.py`, run,

.. code-block:: bash

    python3 run.py -s $path/conf.py


Interactive Mode
--------------------

D.lensalot supports interactive mode, providing direct access to all objects and parameters and step by step execution.
Check out this `interactive`_ notebook for guidance.

.. _interactive: https://github.com/NextGenCMB/D.lensalot/blob/main/notebooks/interactive.ipynb

As a minimal working example, run a new analysis with the :code:`map_delensing` job set to :code:`True` as follows,

.. code-block:: python

    from lenscarf.run import run
    ana_delensing = run(config=<path-to-your-config-file>, job_id='map_delensing')


:code:`interactive` now has the D.lensalot model and the analysis, and all functionalities of the :code:`map_delensing` Job,

.. code-block:: python

    ana_delensing.__dict__.keys()
    >> dict_keys(['data_from_CFS', 'k', 'version', 'imin', 'imax', 'simidxs', 'its', 'Nmf', 'fg', '_package', '_module', '_class', 'class_parameters', 'sims', 'ec', 'nside', 'data_type', 'data_field', 'TEMP', 'libdir_iterators', 'analysis_path', 'base_mask', 'masks', 'binmasks', 'mask_ids', 'beam', 'lmax_transf', 'transf', 'cls_path', 'cls_len', 'clg_templ', 'clc_templ', 'binning', 'lmax', 'lmax_mask', 'edges', 'edges_id', 'sha_edges', 'dirid', 'edges_center', 'ct', 'vers_str', 'TEMP_DELENSED_SPECTRUM', 'dlm_mod_bool', 'file_op', 'cl_calc', 'outdir_plot_rel', 'outdir_plot_root', 'outdir_plot_abs', 'lib', 'jobs'])


And we have access to the simulation data used for this job (here shown an example simulation data)

.. code-block:: python

    ana_delensing.sims.__dict__
    >> {'facunits': 1000000.0,
    >> 'fg': '00',
    >> 'path_set1': '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.00_umilta_210511//cmbs4_08b00_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits',
    >> 'path_noise_set1': '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.00_umilta_210511//cmbs4_08b00_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits',
    >> 'rhitsi': True,
    >> 'p2mask': '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.00_umilta_210511//ILC_mask_08b_smooth_30arcmin.fits',
    >> 'path_set2': '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.00_umilta_210921//cmbs4_08b00_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits',
    >> 'path_noise_set2': '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.00_umilta_210921//cmbs4_08b00_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'}

    

