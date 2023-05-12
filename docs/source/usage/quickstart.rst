.. _get started:

============
Quickstart
============

This section discusses how to successfully run an iterative lensing reconstruction job and how to calculate a delensed power spectrum from that.
We provide a minimal working example by delensing simulation data generated at runtime, generate B-lensing templates, calculate a lensing potential estimate and Wiener-filtered maps.

Type :code:`python3 dlensalot/run.py [-h]` for help,

.. code-block:: text
    
    usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

    delensalot entry point.

    optional arguments:
    -h, --help            show this help message and exit
    -p NEW                Relative path to config file to run analysis.
    -r RESUME             Absolute path to config file to resume.
    -s STATUS             Absolute path for the analysis to write a report.


Run delensalot
--------------------

You can run :code:`delensalot` in two different ways, via a terminal (e.g. to run an analysis), or 'interactively', e.g. in a notebook (to view results or the analysis parameters).


Terminal Mode
++++++++++++++++


To run a configutation file `<path/to/config.py>`, type in your favourite :code:`bash`,

.. code-block:: bash

    python3 run.py -p <path/to/config.py>

:code:`<path/to/config.py>` is a relative path, pointing to a config file in :code:`dlensalot/lerepi/config/`.

For example, when inside :code:`delensalot`'s root folder, 

.. code-block:: bash

    python3 run.py -p examples/cmbs4_fullsky.py

runs the example configuration file :code:`cmbs4_fullsky.py`.
You may want to open the configuration file to look at its settings, as these are the ones you want to make yourself comfortable with if running :code:`delensalot`.

Executing above command copies the configuration file into the :code:`$temp` folder.
In case the analysis has stopped and you'd like to resume where you left off, simply run your analysis from inside the :code:`$temp` folder with the resume (:code:`-r`) parameter,

.. code-block:: bash

    python3 run.py -r $temp/cmbs4_fullsky.py


If you'd like to know the status of the analysis done with :code:`$path/conf.py`, run,

.. code-block:: bash

    python3 run.py -s $temp/cmbs4_fullsky.py

    
This prints the number of calculated files (Wiener-filtered maps, lensing potentials, ..), per iteration, and for all simulation indices.


:code:`cmbs4_fullsky.py` runs QE and MAP lensing reconstruction on a CMB-S4 like configuration on the full sky, i.e. no masking, and generates map delensed power spectra.
The simulation data is generated upon runtime, via :code:`data/sims.py`,
and calculates the QE and MAP mean-fields along the way.
Temporary and final results are stored in the :code:`$temp` directory,
and we recommend using the 'interactive mode' for accessing them.



Interactive Mode
++++++++++++++++++++

:code:`delensalot` supports interactive mode, providing direct access to all objects and parameters and step by step execution.
Check out this `interactive`_ notebook for guidance.

.. _interactive: https://github.com/NextGenCMB/delensalot/blob/main/notebooks/interactive.ipynb

As a minimal working example, start a new analysis for :code:`examples/cmbs4_fullsky.py` with the parameter :code:`job_id=map_delensing`,


.. code-block:: python

    from dlensalot.run import run
    my_mapdelensing_job = run(config='examples/cmbs4_fullsky.py', job_id='map_delensing').job


:code:`my_mapdelensing_job` contains the delensalot model, and all functionalities of the :code:`map_delensing` Job,

.. code-block:: python

    my_mapdelensing_job.__dict__.keys()
    >> dict_keys(['data_from_CFS', 'k', 'version', 'imin', 'imax', 'simidxs', 'its', 'Nmf', 'fg', '_package', '_module', '_class', 'class_parameters', 'sims', 'ec', 'nside', 'data_type', 'data_field', 'TEMP', 'libdir_iterators', 'analysis_path', 'base_mask', 'masks', 'binmasks', 'mask_ids', 'beam', 'lmax_transf', 'transf', 'cls_path', 'cls_len', 'clg_templ', 'clc_templ', 'binning', 'lmax', 'lmax_mask', 'edges', 'edges_id', 'sha_edges', 'dirid', 'edges_center', 'ct', 'vers_str', 'TEMP_DELENSED_SPECTRUM', 'dlm_mod_bool', 'file_op', 'cl_calc', 'outdir_plot_rel', 'outdir_plot_root', 'outdir_plot_abs', 'lib', 'jobs'])

We provide an exhaustive list of available jobs and the structure of the delensalot model in the :ref:`Configuration Files` section.

To run the analysis, simply execute,

.. code-block:: python

    my_mapdelensing_job.run()


This may take a while.
Good time to grab a coffee, tea, or drink.


View delensalot results
------------------------


Depending on your job, you may be interested in the

 * QE or MAP lensing potential,
 * QE or MAP mean-field,
 * QE or MAP B-lensing template,
 * inverse variance, or QE or MAP Wiener-filtered maps,
 * QE or MAP delensed power spectrum.

Which delensalot has stored for you at :code:`$temp`.
We recommend using a dedicated interactive job for this, and we built a conventient interface to the frequently used outputs.
If you followed previous section, simply remove the :code:`job_id` parameter,

.. code-block:: python

    from dlensalot.run import run
    my_dlensalot_results = run(config=<path-to-your-config-file>).job


This provides convenience functions to access the output.


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


If you are working on a masked sky, calculating the power spectrum of this would involve using algoirhtms which handle the mode-coupling. delensalot comes with its own implementation for it.
Simply run the :code:`map_delensing`-job. Then the delensed power spectra with the mask and binning defined inside the configuration file are available via,

.. code-block:: python

    bcl = my_dlensalot_results.get_binned_cl()


:code:`bcl` has shape :code:`[nit,nmasks,nbins,nsims]`.


Assess delensalot analysis
---------------------------

To access all variables and functions of a delensalot job, simply start an Interactive Mode with the :code:`<job-of-my-interest>`.


.. code-block:: python

    from dlensalot.run import run
    my_dlensalot_job = run(config=<path-to-your-config-file>, job_id=<job-of-my-interest>).job


If e.g. :code:`<job-of-my-interest>='MAP_lensrec`, then :code:`my_dlensalot_job` will give you access to the Wiener-filters, response functions, noise models, simulation data, the remapping, etc.


.. code-block:: python

    my_dlensalot_job.__dict__.keys()
    >> dict_keys([])
