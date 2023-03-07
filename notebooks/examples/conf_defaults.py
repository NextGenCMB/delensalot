"""
Our shortest Config yet, CMB-S4 polarization on full sky
"""

from delensalot.lerepi.core.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Job

dlensalot_model = DLENSALOT_Model(
        defaults_to='FS_CMB-S4_Pol',
        job = DLENSALOT_Job(jobs = ["generate_sim", "MAP_lensrec"])
    )