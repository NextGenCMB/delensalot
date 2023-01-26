#!/usr/bin/env python

"""sims.py: Template module for accessing data, for the user to adapt to their needs.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj
import numpy as np
import lenscarf
from lenscarf.sims import sims_ffp10

class mwe:
    def __init__(self, nlev=1):
        self.data_type = 'alm'
        self.data_field = "eb"
        self.beam = 1
        self.lmax_transf = 4000
        self.nside = 2048
        self.data_path =  opj(os.path.abspath(lenscarf.__file__),  'dlensalot/lerepi/config/examples/data')
        self.nlev = nlev
        self.efn='placeholder_%d'


        self.sims = sims_ffp10.cmb_len_ffp10()


    def hashdict(self):
        ret = {'sims':'example', self.nside:512}
        
        return ret


    def get_sim_tlm(self, idx):
        ret = np.load(self.tfn%idx)

        return ret


    def get_sim_elm(self, idx):
        ret = np.load(self.efn%idx)

        return ret
    
    
    def get_sim_elm_filename(self, idx):

        return self.efn%idx


    def get_sim_blm(self, idx):
        ret = np.load(self.bfn%idx)

        return ret


    def get_sim_eblm(self, idx):
        rete = np.load(self.efn%idx)
        retb = np.load(self.bfn%idx)

        return rete, retb