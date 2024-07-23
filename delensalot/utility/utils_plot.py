import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np 


def set_mpl(usetex=True):
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['font.size'] = 20
    # mpl.rcParams['figure.figsize'] = 6.4, 4.8
    mpl.rcParams['figure.figsize'] = 8.5, 5.5

    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    mpl.rc('legend', fontsize=15)
    mpl.rcParams['errorbar.capsize'] = 4
    if usetex:
        mpl.rc('text', usetex=True)
        mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


def pp2kk(ls):
    """Returns the normalisation to convert phi specrum into kappa spectrum
     
     :math: `C_L^{\kappa\kappa} = (L(L+1))^2 / 4 C_L^{\phi\phi}`
     
      """
    return ls**2*(ls+1)**2/4


def plot_bnd(bndcl, ax=None, marker=None, dx=0, ls='', *argv, **kwargs):
    if ax is None:
        ax = pl.gca()
    if marker is None: marker = '.'
    p = ax.errorbar(bndcl[0]+dx, bndcl[1], yerr=bndcl[2], ls=ls, marker=marker,  *argv, **kwargs)
    return p


def bnd(cl, lmin, lmax, edges, weight=lambda ell: np.ones(len(ell), dtype=float)):
    """Binning for weight(ell)*Cl
    
        Args: 
            cl: array to bin, index 0 should be ell=0
            lmin, lmax: range of multipoles to consider in the binning
            edges: boundaries of the bins 
            weight: function of ell, weighting of the binning 

        Returns: 
            ellb: center of the bins
            cl_bnd: binned value
            err: standard deviation of the Cl in the bin 
    
    """
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


def cov2corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def binning_opti(ckk, ckk_fid, ls, edges, resp, fsky=1., ckk_mc=None):
    """
    Optimal weitghing of the Cl kk in the bin with the response
    ckk array always start at ell=0, but we bin using only the ells defined in ls
    """
    nbins = len(edges) - 1
    ellb = np.zeros(nbins)
    ckk_bin = np.zeros(nbins)
    err = np.zeros(nbins)

    ellmax = np.max(edges)
    ell = np.arange(ellmax+1, dtype=int)        
    bin_weight = np.zeros(ellmax+1)

    arr = ckk[0:min(len(ckk), ellmax + 1)]
    inv_weight = np.zeros(ellmax+1)
    inv_weight[1:] = (2*ell[1:] + 1) * fsky * resp[1:ellmax+1]**2 / (2 * (ell*1.)**4 * ((ell*1.)+1)**4)[1:]
    for ibin in range(nbins):
        edge_low = edges[ibin]
        edge_up = edges[ibin+1]
        if (edge_up <= arr.size) and (len(arr[edge_low:edge_up + 1]) >= 1):
            ii = np.where((ls >= edge_low) & (ls < edge_up))[0]
            
            bin_weight[ls[ii]] = ckk_fid[ls[ii]] * inv_weight[ls[ii]] / np.sum(ckk_fid[ls[ii]]**2 * inv_weight[ls[ii]])
            ellb[ibin] = (np.sum(ls[ii] * bin_weight[ls[ii]]) / np.sum(bin_weight[ls[ii]]))
            bin_weight[ls[ii]] *= ckk_fid[int(ellb[ibin])]
                
            if ckk_mc is not None:
                mc_corr = np.sum(bin_weight[ls[ii]] * ckk_fid[ls[ii]]) / np.sum(bin_weight[ls[ii]] * ckk_mc[ls[ii]])
                bin_weight[ls[ii]] *= mc_corr
            
            ckk_bin[ibin] = np.sum(bin_weight[ls[ii]] * arr[ls[ii]])
            # If divide by Nl, it is not variance in the bin but error on the mean 
            err[ibin] = np.sqrt(np.sum(bin_weight[ls[ii]] * (arr[ls[ii]] - ckk_bin[ibin])**2))  /  np.sqrt(max(1, len(ii)))
    return ellb, ckk_bin, err, bin_weight

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
