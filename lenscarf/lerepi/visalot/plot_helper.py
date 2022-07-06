import numpy as np
import abc
import os, sys
import hashlib

import importlib.util as iu

import matplotlib
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import healpy as hp

import lenscarf.lerepi as lerepi
from lenscarf.lerepi.core import handler

from lenscarf.utils import read_map, cli
from lenscarf.iterators.statics import rec

from component_separation.cs_util import Config

from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.core.transformer.lerepi2dlensalot import l2T_Transformer, transform
from lenscarf.lerepi.visalot import plot_helper as ph

ll = np.arange(0,200+1,1)
scale_uk = (2 * ll + 1) * ll**2 * (ll + 1)**2
scale_ps = ll*(ll+1)/(2*np.pi)
label_scale_ps = r'$\frac{\ell(\ell+1)}{2 \pi}$'
label_scale_lp = r'$\frac{\ell^2(\ell+1)^2}{2 \pi}$'
scale_lp = ll**2 * (ll + 1)**2 * 1e7 / (2 * np.pi)

psl = r'$\frac{l(l+1)}{2\pi}C_\ell \/ [\mu K^2]$'

CB_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", 
"#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"]
CB_color_cycle_lighter = ["#68ACCE", "#AC4657", "#ADAC57", "#005713", "#130268", "#8A2479", 
"#248A79", "#797913", "#680235", "#460000", "#4679AC", "#686868"]

def load_paramfile(directory, descriptor):
    """Load parameterfile

    Args:
        directory (_type_): _description_
        descriptor (_type_): _description_

    Returns:
        _type_: _description_
    """
    spec = iu.spec_from_file_location('paramfile', directory)
    p = iu.module_from_spec(spec)
    sys.modules[descriptor] = p
    spec.loader.exec_module(p)

    return p


def load_config():

    return 

def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))

import matplotlib.colors as mcolors
colors1 = plt.cm.Greys(np.linspace(0., .5, 128))
colors2 = [plt.cm.Blues(np.linspace(0.6, 1., 128)), plt.cm.Reds(np.linspace(0.8, 1., 128)), plt.cm.Wistia(np.linspace(0.4, 1., 128)), plt.cm.Greens(np.linspace(0.6, 1., 128))]
mymap = []
nlevels_loc = [1.2, 2., 5.0, 50.0]
for ni, n in enumerate(nlevels_loc):
    colors2[ni][:,-1] = 0.5
    colors_loc = np.vstack((colors1, colors2[ni]))
    mymap.append(mcolors.LinearSegmentedColormap.from_list('my_colormap', colors_loc))
    
    
def get_ms(dat, binspace=5, bin_multipole=False):
    if bin_multipole:
        return get_weighted_avg(np.mean(dat, axis=0), np.std(dat, axis=0), binspace=binspace)
    else:
        return np.mean(dat, axis=0), np.std(dat, axis=0)

def get_weighted_avg(mean, std, binspace):
    lscan = np.arange(0,len(mean),binspace)
    w_average = np.zeros(shape=int(round((len(mean)/binspace))))
    w_variance = np.zeros(shape=int(round((len(mean)/binspace))))
    for n in range(len(w_average)):
        w_average[n] = np.average(mean[n*binspace:(n+1)*binspace], weights=std[n*binspace:(n+1)*binspace])
        w_variance[n] = np.average(std[n*binspace:(n+1)*binspace])
        
        # w_variance[n] = np.average((mean-w_average[n])[n*binspace:(n+1)*binspace]**2, weights=std[n*binspace:(n+1)*binspace])
    return lscan, w_average, w_variance

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