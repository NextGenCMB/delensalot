import numpy as np
import os, sys

import importlib.util as iu
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import healpy as hp
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


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


def movavg(data, window=20):
    y = data
    average_y = []
    for i in range(int((window - 1)/2)):
        average_y.insert(0, np.nan)
        # average_y.insert(-1, np.nan)
        
    for i in range(len(y) - window + 1):
        average_y.append(np.mean(y[i : i + window]))
    for i in range(int((window - 1)/2)+1):
        average_y.append(np.nan)
    return np.array(average_y)


def phi2kappa_bp(data, bpl=2, bpu=4000):
    return ph.bandpass_alms(hp.almxfl(data, np.sqrt(ll * (ll + 1))), bpl, bpu)


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
        return np.mean(dat, axis=0), np.std(dat, axis=0), np.var(dat, axis=0)

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


def get_rotlonlat_mollview(area = 'SPDP'):
    '''mollview coordinates for different sky patches'''
    rotation, lonra, latra = None, None, None
    if area == 'SPDP':
        rotation = np.array([40,-60])
        lonra = np.array([-35,35])+rotation[0]
        latra = np.array([-15,27])+rotation[1]
    elif area == 'CDP':
        print("Not yet defined")
    else:
        print("Do not understand your input")
    return rotation, lonra, latra


def get_planck_cmap():
    cmap = ListedColormap(np.loadtxt("/global/homes/s/sebibel/plots/Planck_Parchment_RGB.txt")/255.)
    cmap.set_bad("gray") # color of missing pixels
    cmap.set_under("white") # color of background, necessary if you want to use
    # this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
    return cmap

def get_custom_cmap():

    size_g = 4
    grey_map = cm.get_cmap('Greys', 32)

    size_bl = 4
    blue_map = cm.get_cmap('Blues', 32)
    blues_dld = np.vstack(
        (blue_map(np.linspace(0.1, 0.8, size_bl)),
        blue_map(np.linspace(0.8, 0.1, size_bl))))

    greys_light_dld = np.vstack(
        (grey_map(np.linspace(.8, 0.1, size_g)),
        grey_map(np.linspace(0.1, .8, size_g))))

    size_o = 8
    orange_map = cm.get_cmap('Oranges', 32)
    oranges_dark_dld = np.vstack(
        (orange_map(np.linspace(0.2, 0.5, size_o)),
        orange_map(np.linspace(0.5, 0.2, size_o))))


    size_gr = 8
    green_map = cm.get_cmap('Greens', 32)
    greens_dld = np.vstack(
        (green_map(np.linspace(0.2, 0.5, size_gr)),
        green_map(np.linspace(0.5, 0.2, size_gr))))

    size_w = 1
    white_map = cm.get_cmap('Greys', 32)
    white = white_map(np.linspace(0.0, 0.1, size_w))

    gb = np.vstack((
        greens_dld[int(size_o):],
        blues_dld[int(size_bl):],
        white[:size_w],
        greys_light_dld[int(size_g):],
        oranges_dark_dld[:int(size_o)]
    ))
    cmap = ListedColormap(gb)
    return cmap


def plot_cmap(cmap, minmax, ticks=[-0.30,-0.15,0,0.15,0.30], fs=18, label='$\mu $K'):
    matplotlib.rcParams.update({'font.size': fs})
    a = np.array([[minmax[0],minmax[1]]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([-0, 1, 1.0, 0.35])
    nticks = 5
    ticks = np.arange(minmax[0], minmax[1]+(minmax[1]-minmax[0])/nticks, (minmax[1]-minmax[0])/nticks)
    cbar = plt.colorbar(orientation="horizontal", cax=cax, ticks=ticks)
    plt.xlabel(label)