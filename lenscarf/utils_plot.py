import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np 


def set_mpl():
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['font.size'] = 20
    # mpl.rcParams['figure.figsize'] = 6.4, 4.8
    mpl.rcParams['figure.figsize'] = 8.5, 5.5

    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rc('text', usetex=True)
    # mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    mpl.rcParams['errorbar.capsize'] = 4
    mpl.rc('legend', fontsize=15)


def pp2kk(ls):
    """Returns the normalisation to convert phi specrum into kappa spectrum
     
     :math: `C_L^{\kappa\kappa} = (L(L+1))^2 / 4 C_L^{\phi\phi}`
     
      """
    return ls**2*(ls+1)**2/4


def plot_bnd(bndcl, ax=None, marker=None, *argv, **kwargs):
    if ax is None:
        ax = pl.gca()
    if marker is None: marker = '.'
    p = ax.errorbar(bndcl[0], bndcl[1], yerr=bndcl[2], ls='', marker=marker,  *argv, **kwargs)
    return p


def bnd(cl, lmin, lmax, edges, weight=lambda ell: np.ones(len(ell), dtype=float)):
    """Binning for weight(ell)*Cl"""
    bl = edges[:-1];bu = edges[1:]
    ellb = 0.5 * bl + 0.5 * bu
    # lmax = len(cl)-1
    ells = np.arange(lmin, lmax+1)
    cl_bnd, err = binned(cl[lmin :lmax+1]*weight(ells), lmin, lmax,  bl, bu, reterr=True,)
    return ellb, cl_bnd, err


def binned(arr, lmin, lmax, bls, bus, reterr=True):
    """
        lmin: first entry of arr is lmin
        lmax: just consistency check for what you're doing

    """
    assert arr.size == lmax - lmin + 1
    ret = np.zeros(len(bls))
    err = np.zeros(len(bls))
    for i, (bl, bu) in enumerate(zip(bls, bus)):
        bu_ = min(lmax, bu)
        bl_ = max(lmin, bl)
        ret[i] = np.mean(arr[bl_ - lmin:bu_ + 1 - lmin])
        err[i] = np.std(arr[bl_ - lmin:bu_ - lmin + 1]) / np.sqrt((bu_ + 1 - bl_))
    return (ret, err) if reterr else ret



# def binned(Cl, nzell, bins_l, bins_u, w=lambda ell: np.ones(len(ell), dtype=float), return_err=False, meanorsum='mean', error='ste'):
#     """Bins a cl array according to bin edges and multipole to consider

#     """
#     assert error in ['ste', 'std']
#     assert meanorsum in ['mean', 'sum']
#     if meanorsum == 'sum': assert not return_err, 'not implemented'
#     sumfunc = np.mean if meanorsum == 'mean' else np.sum
#     ellmax = np.max(bins_u)
#     ell = np.arange(ellmax + 1, dtype=int)
#     Nbins = bins_l.size
#     assert (Nbins == bins_u.size), "incompatible limits"
#     # enlarge array if needed
#     ret = np.zeros(Nbins)
#     arr = w(ell)
#     err = np.zeros(Nbins)
#     # This should work for ist.cl and arrays
#     arr[0: min(len(Cl), ellmax + 1)] *= Cl[0:min(len(Cl), ellmax + 1)]
#     for i in range(Nbins):
#         if (bins_u[i] < arr.size) and (len(arr[bins_l[i]:bins_u[i] + 1]) >= 1):
#             ii = np.where((nzell >= bins_l[i]) & (nzell <= bins_u[i])) # TODO: the <= means that we use two times the boundary values, is it ok?
#             ret[i] = sumfunc(arr[nzell[ii]])
#             if error=='ste':
#                 # Standard error (to get confidence interval on the unknown mean)
#                 err[i] = np.std(arr[nzell[ii]]) / np.sqrt(max(1, len(ii[0])))
#             elif error=='std':
#                 # Standard deviation (to get std of values inside the bin)
#                 err[i] = np.std(arr[nzell[ii]])
#     if not return_err:
#         return ret
#     return ret, err
