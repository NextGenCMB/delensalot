"""
Generates simple point sources, can be correlated with an input lensing convergence map.
"""

import numpy as np
import healpy as hp


class Foreground(object):
    def __init__(self):
        pass

    @staticmethod
    def randomizing_fg(mappa: np.ndarray):
        """
        Randomizes the phase of the input map, preserving the amplitude.
        """
        f = lambda z: np.abs(z) * np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
        return f(mappa)
    
    def randomized_map(self, mappa: np.ndarray, nside: int):
        """
        Randomizes the phase of the input map, preserving the amplitude.
        """
        alm = hp.map2alm(mappa)
        alm = self.randomizing_fg(alm)
        return hp.alm2map(alm, nside)
    

    @staticmethod
    def matched_filter(input_map_alm: np.ndarray, total_cl: np.ndarray, signal_cl: np.ndarray, nside: int):
        """
        Returns the matched filter map.
        """
        alm = hp.almxfl(input_map_alm, np.nan_to_num(1/total_cl))
        alm = hp.almxfl(alm, signal_cl)
        return hp.alm2map(alm, nside)
    
    def mask_from_matched_filter(self, input_map_alm: np.ndarray, total_cl: np.ndarray, signal_cl: np.ndarray, nside: int, threshold: float = 0.5):
        """
        Returns a mask from the matched filter map.
        """
        mappa = self.matched_filter(input_map_alm, total_cl, signal_cl, nside)
        SN_map = abs(mappa) / np.std(mappa)
        mask =  np.where(mappa > threshold, 1, 0)
        return mask


class PointSourcesSimple(Foreground):
     
    def __init__(self, nside: int = 2048) -> None:
        self.nside = nside
    
    @staticmethod
    def phi_lm_to_kappa_lm(plm: np.ndarray) -> np.ndarray:
        """
        Converts the input phi_lm to kappa_lm.
        """
        lmax = hp.Alm.getlmax(plm.size)
        ls = np.arange(0, lmax)
        factor = (ls * (ls + 1.)) / 2.
        return hp.almxfl(plm, factor)
    

    def _get_position_from_kappa_default(self, rng, kappa: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """
        Returns the positions of the point sources from the input kappa map.
        """
        positions = np.where(rng.poisson(1+kappa*factor) > 0)[0]
        return positions
    
    
    def _get_position_from_kappa_alternative(self, rng, kappa: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """
        Returns the positions of the point sources from the input kappa map.
        """
        positions = np.where(rng.poisson(abs(kappa)*factor) > 0)[0]
        return positions
     
    
    def generate_ps(self, nsrc: int, amp: float = 100, seed: int = 0, plm: np.ndarray = None, factor: float = 0.5) -> np.ndarray:
          
        rng = np.random.default_rng(seed)

        mappa = np.zeros(hp.nside2npix(self.nside))

        if plm is not None:
        
            klm = self.phi_lm_to_kappa_lm(plm)
            kmap = hp.alm2map(klm, self.nside, verbose = False)
            positions = np.where(rng.poisson(abs(kmap)*factor) > 0)[0]
            nsources = len(positions)

        else:
            
            nsources = rng.poisson(nsrc)
            positions = np.random.randint(0, len(mappa), nsources)
            
        amplitudes = rng.poisson(amp, nsources)
        mappa[positions] = amplitudes

        return mappa
