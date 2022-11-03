"""08b sims
    Params in Table 2-2 PBDR
    Each class defines a set of data which has been provided for delensing.
    For each class, define the directory, configuration, mask, ..
    Configuration of sims can be found in /config_survey
"""
import os
import numpy as np
from plancklens import utils #TODO switch this to lenscarf
import healpy as hp
from os.path import join as opj

class experiment_config:
    def __init__(self):
        self.freqs = ['030', '040', '095', '145', '220', '270']
        self.beams = np.array([7.3, 5.5, 2.3, 1.5, 1.0, 0.8])
        self.freq2beam = {'030': 7.3, '040': 5.5, '095': 2.3, '145': 1.5, '220': 1.0, '270': 0.8}

        
class foreground:

    def __init__(self, fg):
        rd_skyyy = '/global/cfs/cdirs/cmbs4/awg/lowellbb/sky_yy/'
        rd_exptxx = '/global/cfs/cdirs/cmbs4/awg/lowellbb/expt_xx/'
        self.fg_beamstring = {'030':'07', '040':'06', '095':'02', '145':'02', '220':'01', '270':'01'}
        self.simidx = None
        if fg == '00':
            self.fns_sync = rd_exptxx+'gsync/map/gsync_f{freq}_b{fg_beamstring}_ellmin30_map_{nside}_mc_{simidx:04d}.fits'
            self.fns_dust = rd_exptxx+'gdust/map/gdust_f{freq}_b{fg_beamstring}_ellmin30_map_{nside}_mc_{simidx:04d}.fits'
            self.fns_syncdust = None
            self.flavour = 'QU'
            self.coord = 'celestial'
            self.simidx = 1
        elif fg == '07':
            self.fns_sync = rd_exptxx+'amsync/map/amsync_f{freq}_b{fg_beamstring}_ellmin30_map_{nside}_mc_{simidx:04d}.fits'
            self.fns_dust = rd_exptxx+'amdust/map/amdust_f{freq}_b{fg_beamstring}_ellmin30_map_{nside}_mc_{simidx:04d}.fits'
            self.fns_syncdust = None
            self.flavour = 'QU'
            self.coord = 'celestial'
            self.simidx = 1
        elif fg == '09':
            self.fns_syncdust = rd_skyyy+'09/vans_d1s1_SOS4_{freq}_tophat_map_{nside}.fits'
            self.flavour = 'QU'
            self.coord = 'galactic'


class ILC_config:
    def __init__(self):
        self.weights_fns = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/06b.{}_umilta_210214/w{}_analytical.txt')

        
class caterinaILC_May12:
    """ILC maps from C Umilta on s06b May 12 2021

        These maps are multiplied with the weights used for the ILC

    """
    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['00', '07', '09']
        self.facunits = facunits
        self.fg = fg
        p_set1 =  os.environ['CFS']+'/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        self.path_set1 = p_set1 + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise_set1 =   p_set1 + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi = rhitsi
        self.p2mask = p_set1 + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b
        
        p_set2 =  os.environ['CFS']+'/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210921/'%fg
        self.path_set2 = p_set2 + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise_set2 =   p_set2 + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'


    def hashdict(self):
        ret = {'rhits':self.rhitsi, 'sim_lib':'cmbs4_08b_ILC_%s'%self.fg, 'units':self.facunits, 'path2sim0':self.path_set1%0}
        
        return ret


    def get_sim_pmap(self, idx):
        if idx<200:
            retq = np.nan_to_num(hp.read_map(self.path_set1%idx, field=1)) * self.facunits
            retu = np.nan_to_num(hp.read_map(self.path_set1%idx, field=2)) * self.facunits
            fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        else:
            retq = np.nan_to_num(hp.read_map(self.path_set2%idx, field=1)) * self.facunits
            retu = np.nan_to_num(hp.read_map(self.path_set2%idx, field=2)) * self.facunits
            fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        
        return retq * utils.cli(fac), retu * utils.cli(fac)


    def get_noise_sim_pmap(self, idx):
        if idx<200:
            retq = np.nan_to_num(hp.read_map(self.path_noise_set1%idx, field=1)) * self.facunits
            retu = np.nan_to_num(hp.read_map(self.path_noise_set1%idx, field=2)) * self.facunits
            fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        else:
            retq = np.nan_to_num(hp.read_map(self.path_noise_set2%idx, field=1)) * self.facunits
            retu = np.nan_to_num(hp.read_map(self.path_noise_set2%idx, field=2)) * self.facunits
            fac = 1. if not self.rhitsi else np.nan_to_num(hp.read_map(self.p2mask))
        return retq * utils.cli(fac), retu * utils.cli(fac)


# TODO depracated.. soon 
class caterinaILC_Sep12:
    """ILC maps from C Umilta on s08b September 12 2021

        These maps are multiplied with the weights used for the ILC

    """
    def __init__(self, fg, facunits=1e6, rhitsi=True):
        assert fg in ['00', '07', '09']
        self.facunits = facunits
        self.fg = fg
        p_set1 =  os.environ['CFS']+'/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg
        p =  os.environ['CFS']+'/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210921/'%fg
        self.path = p + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path0 = p_set1 + '/cmbs4_08b' + fg + '_cmb_b02_ellmin30_ellmax4050_map_2048_%04d.fits' # CMB + noise
        self.path_noise =   p + '/cmbs4_08b' + fg + '_noise_b02_ellmin30_ellmax4050_map_2048_%04d.fits'
        self.rhitsi=rhitsi
        self.p2mask = p_set1 + '/ILC_mask_08b_smooth_30arcmin.fits' # Same mask as 06b
        self.noisemodel_mask = '/global/cscratch1/sd/sebibel/cmbs4/08b_rhits_positive_nonan.fits'


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