#!/usr/bin/env python

"""sims.py: Template module for accessing data, for the user to adapt to their needs.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
from os.path import join as opj
import numpy as np
import healpy as hp

from plancklens.sims import maps, phas

from lenscarf.utils_hp import gauss_beam
from lenscarf.sims import sims_ffp10


class ffp10:
    def __init__(self, nlev_p=1):
        self.data_type = 'alm'
        self.data_field = "eb"
        self.beam = 1
        self.lmax_transf = 4096
        self.nside = 2048
        self.nlev_p = nlev_p
        self.nlev_t = nlev_p*np.sqrt(2)
        pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%self.nside), 3, (hp.nside2npix(self.nside),)) # T, Q, and U noise phases
        transf_dat = gauss_beam(self.beam / 180 / 60 * np.pi, lmax=self.lmax_transf) # (taking here full FFP10 cmb's which are given to 4096)
        self.sims = maps.cmb_maps_nlev(sims_ffp10.cmb_len_ffp10(), transf_dat, self.nlev_t, self.nlev_p, self.nside, pix_lib_phas=pix_phas)


    def hashdict(self):
        ret = {'sims':'fullsky', 'nside':self.nside}
        
        return ret


class userdata:
    def __init__(self, user_parameter='no value chosen'):
        print(user_parameter)
        self.data_type = 'map'
        self.data_field = "qu"
        self.beam = 2
        self.lmax_transf = 4096
        self.nside = 2048

        p_set1 =  os.environ['CFS']+'/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        self.path_set1 = p_set1 + 'filename.fits'
        self.path_noise_set1 =   p_set1 + 'filename.fits'
        self.p2mask = p_set1 + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b
        self.sims = userdata(user_parameter)

    def hashdict(self):
        ret = {'sims':'example', 'nside':self.nside}
        
        return ret


    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_set1%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_set1%idx, field=2)) * self.facunits
        
        return retq, retu


    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_noise_set1%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise_set1%idx, field=2)) * self.facunits
        
        return retq, retu


