"""
Extra sims utility.
"""

import os
import healpy as hp
import numpy as np


class Extra(object):
    """
    Example: extra_tlm = Extra('fgs', fgnames)
    """

    def __init__(self, baseWebsky, name, fgnames):
        self.name = name
        self.fgnames = fgnames
        self.directory = baseWebsky

    def __call__(self, idx):
        return np.sum([hp.read_map(opj(self.directory, f'{fgname}.fits')) for fgname in self.fgnames], axis = 0)
    
    def get_name(self):
        return self.name
                