============
Quickstart
============

This section discusses how to successfully run your first delensing job. Our final result will be a delensed power spectrum. 
We provide a minimal working example by delensing simulation data generated at runtime, generate B-lensing templates and find a lensing potential estimate and Wiener-filtered maps.

Type :code:`python3 lenscarf/run.py [-h]` for quickhelp,

.. code-block:: text
    
    usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

    D.lensalot entry point.

    optional arguments:
    -h, --help            show this help message and exit
    -p NEW                Relative path to config file to run analysis.
    -r RESUME             Absolute path to config file to resume.
    -s STATUS             Absolute path for the analysis to write a report.


Run D.lensalot
--------------------
You can run D.lensalot in two different ways, via a terminal, or 'interactively', e.g. in a notebook.


Terminal Mode
++++++++++


To run a configutation file :code:`<path-to-config>`, type in your favourite :code:`bash`,

.. code-block:: bash

    python3 run.py -p <path-to-config>

:code:`<path-to-config>` is a relative path, pointing to a config file in :code:`lenscarf/lerepi/config/`.

For example,

.. code-block:: bash

    python3 run.py -p examples/example_c08b.py

runs the example configuration for :code:`c08b`.
This example file runs QE and MAP lensing reconstruction on a CMB-S4 like configuration on the full sky, and generates map delensed power spectra.
It also calculates the QE and MAP mean-fields along the way.
Temporary and final results are stored in the :code:`$temp` directory.

If you already have an analysis located at `$path`, with config file `conf.py`, you may resume this analysis with,

.. code-block:: bash

    python3 run.py -r $path/conf.py

This is in particular handy if your run didn't finish, or you would like an additional job to be executed for this analysis.

If you'd like to know the status of the analysis done with :code:`$path/conf.py`, run,

.. code-block:: bash

    python3 run.py -s $path/conf.py


Interactive Mode
+++++++++++++++++

D.lensalot supports interactive mode, providing direct access to all objects and parameters and step by step execution.
Check out this `interactive`_ notebook for guidance.

.. _interactive: https://github.com/NextGenCMB/D.lensalot/blob/main/notebooks/interactive.ipynb

As a minimal working example, run a new analysis with the :code:`map_delensing` job set to :code:`True` as follows,

.. code-block:: python

    from lenscarf.run import run
    my_mapdelensing_job = run(config=<path-to-your-config-file>, job_id='map_delensing').job


:code:`my_mapdelensing_job` contains the D.lensalot model, and all functionalities of the :code:`map_delensing` Job,

.. code-block:: python

    my_mapdelensing_job.__dict__.keys()
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

    
We provide an exhaustive list of available jobs in the :ref:`Configuration Files` section, such as QE lensing reconstruction, MAP lensing reconstruction, and an interactive job helper.

Assess D.lensalot output
---------------------------

Depending on your job, you may be interested in the

 * QE or MAP lensing potential,
 * QE or MAP mean-field,
 * QE or MAP B-lensing template,
 * inverse variance, or QE or MAP Wiener-filtered maps,
 * QE or MAP delensed power spectrum.

Which D.lensalot has stored for you at :code:`$temp`.
We recommend using a dedicated interactive job for this, which can simply be run by running the following line inside your favourite interactive python interface,

.. code-block:: python

    from lenscarf.run import run
    my_dlensalot_results = run(config=<path-to-your-config-file>).job


This implicitly runs a :code:`notebook_interactor` job, and provides convenience functions to access the output.
Most functions rely on two parameters; :code:`simidx` is the index of the simulation, put :code:`simdix=-1` if you'd like to access your real data.
:code:`it` is the index of the iteration. Use :code:`it=0` for QE, and :code:`it=-1` for the last iteration, i.e. the MAP result.
All convencience functions return the data in spherical harmonic coefficients and Healpy-format.


.. code-block:: python

    QE_lensing_potential = my_dlensalot_results.load_plm(simidx=0, it=0)
    MAP_lensing_potential = my_dlensalot_results.load_plm(simidx=0, it=-1)

    QE_mean_field = my_dlensalot_results.load_mf(simidx=0)
    MAP_mean_field = my_dlensalot_results.load_mf(simidx=0)

    QE_Blensing_template = my_dlensalot_results.get_blt(simidx=0, it=0)
    MAP_Blensing_template = my_dlensalot_results.get_blt(simidx=0, it=-1)

    QE_Eivf = my_dlensalot_results.get_ivf('E', simidx=0, it=0)
    MAP_Eivf = my_dlensalot_results.get_ivf('E', simidx=0, it=-1)

    QE_EWF = my_dlensalot_results.get_wf('E', simidx=0, it=0)
    MAP_EWF = my_dlensalot_results.get_wf('E', simidx=0, it=-1)

    MAP_Blensing_template = my_dlensalot_results.get_blt(simidx=0, it=-1)
    MAP_Blensing_template = my_dlensalot_results.get_blt(simidx=0, it=-1)


To view the results,
you could use healpy and either calculate the power spectrum with its :code:`alm2cl` function, or calculate the map with its :code:`alm2map()` functions.


.. code-block:: python

    import healpy as hp 
    plt.plot(hp.alm2cl(MAP_lensing_potential))
    plt.show()

    hp.mollview(hp.alm2map(MAP_lensing_potential))
    plt.show()



You may want to compare your result to either the fiducial input, or the simulation data. We have got you covered.
You can load the fiducial, and simulation data as follows.


.. code-block:: python

    fiducial_spectra = my_dlensalot_results.get_fiducial_spectrum()
    fiducial_map = my_dlensalot_results.get_fiducial_map()
    simulation_data = my_dlensalot_results.get_simulation_data()


To calculate delensed maps, simply subtract one from the other.


.. code-block:: python

    fiducial_map = my_dlensalot_results.get_fiducial_map()
    MAP_Blensing_template = my_dlensalot_results.get_blt(simidx=0, it=-1)
    MAP_delensed_map = fiducial_map - MAP_Blensing_template


If you are working on a masked sky, calculating the power spectrum of this would involve using algoirhtms which handle the mode-coupling. D.lensalot comes with its own implementation for it.
Simply run the :code:`map_delensing`-job. Then the delensed power spectra with the mask and binning defined inside the configuration file are available via,

.. code-block:: python

    bcl = my_dlensalot_results.get_binned_cl()


:code:`bcl` has shape :code:`[nit,nmasks,nbins,nsims]`.




Assess D.lensalot analysis
---------------------------

To access all variables and functions of a D.lensalot job, simply start an Interactive Mode with the :code:`<job-of-my-interest>`.


.. code-block:: python

    from lenscarf.run import run
    my_dlensalot_job = run(config=<path-to-your-config-file>, job_id=<job-of-my-interest>).job


If e.g. :code:`<job-of-my-interest>='MAP_lensrec`, then :code:`my_dlensalot_job` will give you access to the Wiener-filters, response functions, noise models, simulation data, the remapping, etc.


.. code-block:: python

    my_dlensalot_job.__dict__.keys()
    >> dict_keys([])
