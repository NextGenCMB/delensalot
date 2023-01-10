#!/usr/bin/env python

"""sims.py: Template module for accessing data, for the user to adapt to their needs.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj
import numpy as np
import lenscarf


class mwe:
    def __init__(self, nlev=1):
        self.data_path =  opj(os.path.abspath(lenscarf.__file__),  'dlensalot/lerepi/config/examples/data')
        self.tfn = opj(self.data_path, 'example_Tmap_nside512.npy') # temperature CMB + noise
        self.efn = opj(self.data_path, 'example_Emap_nside512.npy') # e-polarization CMB + noise
        self.bfn = opj(self.data_path, 'example_Bmap_nside512.npy') # b-polarization CMB + noise
        self.nlev = nlev


    def hashdict(self):
        ret = {'sims':'example', self.nside:512}
        
        return ret


    def get_sim_tlm(self, idx):
        ret = np.load(self.tfn%idx)

        return ret


    def get_sim_elm(self, idx):
        ret = np.load(self.efn%idx)

        return ret


    def get_sim_blm(self, idx):
        ret = np.load(self.bfn%idx)

        return ret


    def get_sim_eblm(self, idx):
        rete = np.load(self.efn%idx)
        retb = np.load(self.bfn%idx)

        return rete, retb