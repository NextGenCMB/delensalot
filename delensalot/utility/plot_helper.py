import numpy as np
import os, sys
from matplotlib.colors import ListedColormap

import healpy as hp
import delensalot

planck_cmap = ListedColormap(np.loadtxt(os.path.dirname(delensalot.__file__)+"/data/Planck_Parchment_RGB.txt")/255.)
planck_cmap.set_bad("gray")
planck_cmap.set_under("white") 

def movavg(data, window=20):
    y = np.asarray(data)
    smoothed = np.full_like(y, np.nan, dtype=np.float64)  # Initialize output with NaNs
    for i in range(len(y)):
        w = window if i < 100 else 5 * window  # Use different window sizes
        w_half = w // 2
        start = max(0, i - w_half)
        end = min(len(y), i + w_half + 1)
        smoothed[i] = np.mean(y[start:end])  # Compute mean over valid range
    return smoothed


def bandpass_alms(alms, lmin, lmax=None):
    """
    lmin: minimum multipole to keep in alms
    lmax: maximimum multipole to keep in alms
    """
    if len(alms) == 3:
        out = np.zeros(alms.shape, dtype=complex)
        for idx, _alms in enumerate(alms):
            out[idx] = bandpass_alms(_alms, lmin, lmax=lmax)
        return out
    
    lmax_in_alms = hp.Alm.getlmax(len(alms))
    if lmax is None:
        lmax = lmax_in_alms
    else:
        assert isinstance(lmax, int), "lmax should be int: {}".format(lmax)
        assert lmax <= lmax_in_alms, "lmax exceeds lmax in alms: {} > {}".format(lmax, lmax_in_alms)
    
    fl = np.zeros(lmax_in_alms + 1, dtype=float)
    fl[lmin:lmax+1] = 1
    
    return hp.almxfl(alms, fl)