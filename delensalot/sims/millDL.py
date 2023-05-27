"""Generic cmb-only sims module

"""

import os
import numpy as np, healpy as hp
from os.path import join as opj
import logging
log = logging.getLogger(__name__)
from plancklens.sims.maps import cmb_maps_nlev

from plancklens.sims import phas

from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import data_functions as df, LEREPI_Constants as lc
     
class millDL:
    def __init__(self, nlev_p, beam, lib_dir=None):
        self.path = '/global/cfs/cdirs/cmb/data/generic/mmDL/healpix/%05d'
        self.fnsQ = 'lensed_cmb_Q_%05d.fits'
        self.fnsU = 'lensed_cmb_Q_%05d.fits'
        self.pix_lib_phas = phas.pix_lib_phas(lib_dir, 3, (hp.nside2npix(2048),))
        self.nlev_p = nlev_p
        self.cl_transf_P = gauss_beam(df.a2r(beam), lmax=4096)


    def get_sim_qnoise(self, simidx):
        """Returns noise Q-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        noisescaling = 1 # hp.read_map('/mnt/c/Users/sebas/OneDrive/SCRATCH/delensalot/generic/sims_cmb_len_lminB200_mfda_rhitssky_center/noisescaling.fits')
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=1) * noisescaling

    def get_sim_unoise(self, simidx):
        """Returns noise U-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=2)

    def get_sim_pmap(self, simidx):
        Qmap = hp.read_map(opj(self.path.format(simidx), self.fnsQ.format(simidx)))
        Umap = hp.read_map(opj(self.path.format(simidx), self.fnsU.format(simidx)))
        elm, blm = hp.map2alm_spin([Qmap, Umap], lmax=4096)
        hp.almxfl(elm,self.cl_transf_P,inplace=True)
        hp.almxfl(blm, self.cl_transf_P, inplace=True)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2, hp.Alm.getlmax(elm.size))

        return Q + self.get_sim_qnoise(simidx), U + self.get_sim_unoise(simidx)

    def hashdict(self):
        return {}