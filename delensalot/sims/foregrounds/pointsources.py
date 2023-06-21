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
    
    @staticmethod
    def randomized_map(self, mappa: np.ndarray, nside: int):
        """
        Randomizes the phase of the input map, preserving the amplitude.
        """
        alm = hp.map2alm(mappa)
        alm = self.randomizing_fg(alm)
        return hp.alm2map(alm, nside)


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
     
    
    def generate_ps(self, nsrc: int, amp: float = 200, seed: int = 0, plm: np.ndarray = None, factor: float = 2.) -> np.ndarray:
          
        rng = np.random.default_rng(seed)


        if plm is not None:
            klm = self.phi_lm_to_kappa_lm(plm)
            kmap = hp.alm2map(klm, self.nside, verbose = False)
            positions = np.where(rng.poisson(abs(kmap)*factor) > 0)[0]
            nsources = len(positions)
            mappa = np.zeros(hp.nside2npix(self.nside))

        else:
            nsources = rng.poisson(nsrc)
            mappa = np.zeros(hp.nside2npix(self.nside))
            positions = np.random.randint(0, len(mappa), nsources)
            
        amplitudes = rng.poisson(amp, nsources)
        mappa[positions] = amplitudes

        return mappa
