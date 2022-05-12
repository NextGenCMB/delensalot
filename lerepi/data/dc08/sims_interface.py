"""08b sims

    Params in Table 2-2 PBDR

"""
import os
import numpy as np
import plancklens
from plancklens import utils
import healpy as hp
import cmbs4

class caterinaILC_May12:
    """ILC maps from C Umilta on s06b May 12 2021

        These maps are multiplied with the weights used for the ILC

    """
    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['00', '07', '09']
        self.facunits = facunits
        self.fg = fg
        p =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        self.path = p + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi=rhitsi
        self.p2mask = p + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b

    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'cmbs4_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path%0}
        return ret

    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)

    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_noise%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


class caterinaILC_Sep12:
    """ILC maps from C Umilta on s08b September 12 2021

        These maps are multiplied with the weights used for the ILC

    """
    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['00', '07', '09']
        self.facunits = facunits
        self.fg = fg
        p_set1 =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        p =  '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210921/'%fg
        self.path = p + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path0 = p_set1 + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi=rhitsi
        self.p2mask = p_set1 + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b

    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'cmbs4_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path0%0}
        return ret

    def get_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)

    def get_noise_sim_pmap(self, idx):
        retq = np.nan_to_num(hp.read_map(self.path_noise%idx, field=1)) * self.facunits
        retu = np.nan_to_num(hp.read_map(self.path_noise%idx, field=2)) * self.facunits
        fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)