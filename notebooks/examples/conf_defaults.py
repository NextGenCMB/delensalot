"""
Our shortest Config yet, CMB-S4 polarization on full sky
"""

from delensalot.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Job

dlensalot_model = DLENSALOT_Model(
        defaults_to = 'P_FS_CMBS4'
    )