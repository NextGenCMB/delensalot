lerepi
===========

CMB lensing reconstruction pipelines for various experiments (CMB-S4, PICO)

Installation
=================

git clone and then,

.. code-block:: console

    python3 -m pip install -e .

`-e` only if you are a developer, of course.

You may also use the setup.py if you prefer.

Lerepi in addition depends on Plancklens and D.lensalot. Ask for guidance if needed


Pipelines
=============

Individual pipelines are found as a distinct branch whith the following convention,
    **p/<experiment>**,
where **p** stands for pipeline, and experiment is the identifier of the **experiment**



p/pico branch
-----------------

Parameter file and scripts to run iterative lensing reconstruction on PICO simulation data
