===========================================
D.Lensalot 
==========================================

.. image:: res/dlensalot2.PNG
  :width: 400
  :alt: D.lensalot logo

(formerly known as Lenscarf)

Curved-sky iterative CMB lensing tools

Installation
----------------
Download the project to your computer, navigate to the root folder and execute the command,

.. code-block:: bash
 
 python setup.py install


Dependencies
---------------

 based on
  * [Scarf](https://github.com/samuelsimko/scarf)
  * [Plancklens](https://github.com/carronj/plancklens)

Doc
----------------

Documentation may be found [HERE]


Use on NERSC
----------------

To use D.lensalot on NERSC, you need to load some libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
Type these lines in the terminal or include them into your `~/.bash_profile`:

```
module load fftw
module load gsl
module load cfitsio
module swap PrgEnv-intel PrgEnv-gnu
module load python
```


