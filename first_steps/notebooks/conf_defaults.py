"""
Our shortest configuration file. All parameters default to an idealizaed CMB-S4-like setting for polarization data.
Simulates CMB polarization maps generated on the fly, inclusive of isotropic white noise. Performs QE and iterative reconstruction on the full-sky.
"""

from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model

dlensalot_model = DLENSALOT_Model(
    defaults_to = 'default_CMBS4_fullsky_polarization'
)