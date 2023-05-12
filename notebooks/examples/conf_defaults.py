"""
Simulates CMB polarization maps generated on the fly, inclusive of isotropic white noise.
"""

from delensalot.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model


dlensalot_model = DLENSALOT_Model(
    defaults_to = 'P_FS_CMBS4'
)