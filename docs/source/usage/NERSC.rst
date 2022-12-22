============================
High Performance Computing
============================


D.lensalot is computationally demanding. Depending on your configuration, it may be advisable to use its built-in :code:`MPI`-support.
D.lensalot will return information upon runtime, if :code:`MPI` is working.

In general, D.lensalot distributes the simulations across :code:`MPI`-jobs.


NERSC
------

To use D.lensalot on NERSC, load some libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
Type these lines in the terminal or include them into your :code:`~/.bash_profile`:

.. code-block:: bash
    
    module load fftw
    module load gsl
    module load cfitsio
    module swap PrgEnv-intel PrgEnv-gnu
    module load python