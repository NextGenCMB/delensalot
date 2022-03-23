#!/usr/bin/env python
# coding: utf-8

# # Tutorial 0: create your own Parameterfile
# In this tutorial, you will learn how to create a parameterfile which contains all configurations to successfully delens your data.
# 
# To convert this tutorial into a parameterfile, execute `jupyter nbconvert --to script 'my-notebook.ipynb'` in your shell.

# ## Basic imports

# In[1]:


import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens
from plancklens import utils
from plancklens import qresp
from plancklens import qest, qecl
from plancklens.qcinv import cd_solve

from plancklens.sims import maps, phas, planck2018_sims
from plancklens.filt import filt_simple, filt_util

from lenscarf import remapping
from lenscarf import utils_scarf, utils_sims
from lenscarf.iterators import cs_iterator as scarf_iterator, steps

from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam, almxfl, alm_copy
from lenscarf.opfilt.opfilt_iso_ee_wl import alm_filter_nlev_wl


# ## Overview
# Delensing consists of the following steps which need to be configured,
#  * output path
#  * general
#  * geometry
#  * fiducial spectra
#  * transfer function
#  * quadratic estimator
#  * chain descriptor
#  * libdir iterator
#  * data / simulations

# ## output path
# Let us store the `Dlensalot` output @ `$SCRATCH/Dlensalot/tutorial0/cmbs4_idealized/`

# In[3]:


TEMP =  opj(os.environ['SCRATCH'], 'Dlensalot/tutorial0', 'cmbs4_idealized')


# ## Geometry
#  * When working with CMB data, most likely you will be working with healpix maps, which are defined by the healpix geometry.
#  * There could be use cases in which you could want the geometry to be different.
#  * D.lensalot uses Scarf, a python wrapper about DUCC to calculate the spherical harmonics.
#  * DUCC itself supports various predefined, and custom geometries.
#  * For more info, see https://scarfcmb.readthedocs.io/en/latest/geometry.html

# In[ ]:




