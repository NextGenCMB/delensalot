#!/usr/bin/env python

"""sims.py: Template module for accessing data, for the user to adapt to their needs.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj
import numpy as np
import healpy as hp

from plancklens.sims import maps, phas

import lenscarf
from lenscarf.utils_hp import gauss_beam
from lenscarf.sims import sims_ffp10


class mwe:
    def __init__(self, nlev_p=1):
        self.data_type = 'alm'
        self.data_field = "eb"
        self.beam = 1
        self.lmax_transf = 4000
        self.nside = 2048
        self.nlev_p = nlev_p
        self.nlev_t = nlev_p*np.sqrt(2)
        pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%self.nside), 3, (hp.nside2npix(self.nside),)) # T, Q, and U noise phases
        transf_dat = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lmax_transf) # (taking here full FFP10 cmb's which are given to 4096)
        self.sims = maps.cmb_maps_nlev(sims_ffp10.cmb_len_ffp10(), transf_dat, self.nlev_t, self.nlev_p, self.nside, pix_lib_phas=pix_phas)


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