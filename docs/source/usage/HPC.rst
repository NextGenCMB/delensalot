============================
High Performance Computing
============================


D.lensalot is computationally demanding.
Depending on your configuration, it may be advisable to use its built-in :code:`MPI`-support,

In general, D.lensalot distributes simulations across :code:`MPI`-jobs, and will return information about the :code:`MPI`-distribution upon runtime.

To use :code:`MPI`, simply run D.lensalot using slurms :code:`srun` command,

.. code-block:: bash

    srun -n N -c C python3 dlensalot/run.py -p <path/to/config.py>


NERSC
------


Please see the NERSC help for detailed support.

.. _help: https://docs.nersc.gov/development/programming-models/mpi/

.. To use D.lensalot on NERSC, load the following libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
.. Type these lines in the terminal or include them into your :code:`~/.bash_profile`:

.. .. code-block:: bash
    
..     module load fftw
..     module load gsl
..     module load cfitsio
..     module swap PrgEnv-intel PrgEnv-gnu
..     module load python