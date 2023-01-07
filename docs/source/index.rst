.. D.lensalot documentation master file, created by
   sphinx-quickstart on Wed Jun 29 13:23:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================
Welcome to D.lensalot
======================

Curved-sky quadratic and iterative CMB delensing tools.

Lensing reconstruction, B-lensing template generation and power spectrum calculation of (partially delensed) maps.

Includes quadratic estimated and iterative lensing reconstruction on map-, and spectrum level.

Quadratic estimated lensing reconstruction comes from the built-in interface to `Plancklens`_.
Iterative lensing reconstruction comes from the curved sky implementation of `LensIt`_.
D.lensalot is best used via :ref:`Configuration Files`.

If you are new to D.lensalot, go check out how to :ref:`get started`.


.. _Plancklens: https://github.com/carronj/plancklens
.. _LensIt: https://github.com/carronj/LensIt


.. toctree::
   :maxdepth: 2
   :caption: Usage:

   usage/installation
   usage/quickstart
   usage/HPC


Model Configuration
====================

.. toctree::
   :maxdepth: 2
   :caption: Analysis set-up:

   config/params
   config/data
   config/temp


Lensing Reconstruction
=======================

.. toctree::
   :maxdepth: 2
   :caption: Numerics:

   lensrec/overview
   lensrec/QE
   lensrec/MAP


Map-level Delensing
==========

.. toctree::
   :maxdepth: 2
   :caption: Internal:

   delensing/overview
   delensing/BLT


Analytical Prediction
=====================

.. toctree::
   :maxdepth: 2
   :caption: Internal:

   prediction/n0
   prediction/n1
   prediction/delensing



Data Product
=============

.. toctree::
   :maxdepth: 2
   :caption: Internal:

   product/filtered_maps
   product/phi
   product/mean-field
   product/BLT


Other
=====

.. toctree::
   :maxdepth: 2
   :caption: Removing nuisance:

   other/OBD
   other/input_data

About
=======

.. toctree::
   :maxdepth: 2
   :caption: Framework:

   about/development
   about/publications
   about/people


Modules
==========

.. toctree::
   :maxdepth: 2
   :caption: API:
   
   modules/modules



Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
