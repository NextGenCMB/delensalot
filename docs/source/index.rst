.. D.lensalot documentation master file, created by
   sphinx-quickstart on Wed Jun 29 13:23:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================
Welcome to D.lensalot
===============================

Curved-sky quadratic and iterative CMB lensing tools.

Lensing reconstruction, B-lensing template generation and power spectrum calculation of (partially) delensed maps.


Includes quadratic estimated and iterative lensing reconstruction on map-level.
All calculations are done on curved sky.

Quadratic estimated lensing reconstruction comes from the built-in interface to `Plancklens`_.
Iterative lensing reconstruction comes from the curved sky implementation of `LensIt`_.
D.lensalot is controlled via :ref:`Configuration Files`.


.. _Plancklens: https://github.com/carronj/plancklens
.. _LensIt: https://github.com/carronj/plancklens


.. toctree::
   :maxdepth: 2
   :caption: Usage:

   usage/installation
   usage/quickstart
   usage/NERSC


Model Configuration
====================

.. toctree::
   :maxdepth: 2
   :caption: Simple analysis set-up:

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


Delensing
==========

.. toctree::
   :maxdepth: 2
   :caption: Internal:

   delensing/overview
   delensing/B-lensing-template


Other
=============

.. toctree::
   :maxdepth: 2
   :caption: Removing nuisance:

   other/OBD
   other/other

About
=============

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
