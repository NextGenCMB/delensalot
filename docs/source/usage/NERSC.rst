use on NERSC
=============


D.lensalot is computationally demanding and therefore needs NERSC.

To use D.lensalot on NERSC, you need to load some libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
Type these lines in the terminal or include them into your :code:`~/.bash_profile`:

.. code-block:: bash
    module load fftw
    module load gsl
    module load cfitsio
    module swap PrgEnv-intel PrgEnv-gnu
    module load python