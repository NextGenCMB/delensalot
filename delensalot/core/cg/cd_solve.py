"""Flexible conjugate directions solver module.

"""
import numpy as np


def PTR(p, t, r):
    return lambda i: max(0, i - max(p, int(min(t, np.mod(i, r)))))


tr_cg = (lambda i: i - 1)
tr_cd = (lambda i: 0)

from plancklens.qcinv.util_alm import eblm, teblm

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
import healpy as hp

def plot_stuff(residual, residualdata, bdata, fwddata, xdata, precondata, searchdirs, searchfwds, weights, x):
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    def get_rainbow_colors(num_items):
        cmap = plt.cm.rainbow  # Choose the rainbow colormap
        return [cmap(i / (num_items - 1)) for i in range(num_items)]
    
    if isinstance(residual, (eblm, teblm)):
        residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual.elm)])
        # fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(searchfwds[0])])
        xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x.elm)])
        # precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs[0])])
    else:
        residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual)])
        # fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(searchfwds[0])])
        xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x)])
        # precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs[0])])

    colors = get_rainbow_colors(len(residualdata)+1)
    clear_output(wait=True)

    plt.figure(figsize=(10, 6))
    for linei, line2 in enumerate(residualdata):
        for resi, res in enumerate(line2):
            plt.plot(res, label='iter %d'%(linei+1), color=colors[linei], ls='-' if resi else '-')
    # plt.legend(title='CG search')
    plt.ylabel(r'$C_\ell^{\rm residual}$')
    plt.xlabel(r'$\ell$')
    plt.yscale('log')
    plt.show()

    plt.figure(figsize=(10, 6))
    for linei, line2 in enumerate(xdata):
        for res in line2:
            plt.plot(res, label='iter %d'%(linei+1), color=colors[linei], ls='-' if resi else '-')
    # plt.legend(title='CG search')
    plt.ylabel(r'$C_\ell^{\rm x}$')
    plt.xlabel(r'$\ell$')
    plt.yscale('log')
    plt.show()

    # plt.figure(figsize=(10, 6))
    # for linei, line2 in enumerate(bdata):
    #     for resi, res in enumerate(line2):
    #         plt.plot(res, label='iter %d'%(linei+1), color=colors[linei], ls='-' if resi else '-')
    # # plt.legend(title='CG search')
    # plt.ylabel(r'$C_\ell^{\rm b}$')
    # plt.xlabel(r'$\ell$')
    # plt.yscale('log')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # for linei, line2 in enumerate(fwddata):
    #     for res in line2:
    #         plt.plot(res, label='iter %d'%(linei+1), color=colors[linei], ls='-' if resi else '-')
    # # plt.legend(title='CG search')
    # plt.ylabel(r'$C_\ell^{\rm fwd(x)}$')
    # plt.xlabel(r'$\ell$')
    # plt.yscale('log')
    # plt.show()

    # plt.close('all')

    # plt.figure(figsize=(10, 6))
    # for linei, line2 in enumerate(precondata):
    #     for res in line2:
    #         plt.plot(res, label='iter %d'%(linei+1))
    # # plt.legend(title='CG search')
    # plt.ylabel(r'$C_\ell^{\rm precon}$')
    # plt.xlabel(r'$\ell$')
    # plt.yscale('log')
    # plt.show()

class cache_mem(dict):
    def __init__(self):
        pass

    def store(self, key, data):
        [dTAd_inv, searchdirs, searchfwds] = data
        self[key] = [dTAd_inv, searchdirs, searchfwds]

    def restore(self, key):
        return self[key]

    def remove(self, key):
        del self[key]

    def trim(self, keys):
        assert (set(keys).issubset(self.keys()))
        for key in (set(self.keys()) - set(keys)):
            del self[key]


def cd_solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr, cache=cache_mem(), roundoff=25, maxiter=200):
    """customizable conjugate directions loop for x=[fwd_op]^{-1}b.

    Args:
        x (array-like)              :Initial guess of linear problem  x =[fwd_op]^{-1}b.  Contains converged solution
                                at the end (if successful).
        b (array-like)              :Linear problem  x =[fwd_op]^{-1}b input data.
        fwd_op (callable)           :Forward operation in x =[fwd_op]^{-1}b.
        pre_ops (list of callables) :Pre-conditioners.
        dot_op (callable)           :Scalar product for two vectors.
        criterion (callable)        :Decides convergence.
        tr                          :Truncation / restart functions. (e.g. use tr_cg for conjugate gradient)
        cache (optional)            :Cacher for search objects. Defaults to cache in memory 'cache_mem' instance.
        roundoff (int, optional)    :Recomputes residual by brute-force every *roundoff* iterations. Defaults to 25.

    Note:
        fwd_op, pre_op(s) and dot_op must not modify their arguments!

    """

    residualdata, bdata, fwddata, xdata, precondata = [], [], [], [], []

    n_pre_ops = len(pre_ops)

    residual = b - fwd_op(x)
    # print('x is', x[10000:10010])
    # print('fwd(x) is', fwd_op(x))
    # print('going to calculate pre_op')
    searchdirs = [op(residual) for op in pre_ops]


    if isinstance(residual, (eblm, teblm)):
        lmax = np.max([Alm.getlmax(r.size, None) for r in residual.elm])
        ell = np.arange(0, lmax + 1)
        weights = 2 * ell + 1
        residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual.elm)])
        # bdata.append([hp.alm2cl(b_) for b_ in np.atleast_2d(b)])
        # fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(fwd_op(x).elm)])
        xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x.elm)])
        # precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs)])
    else:
        lmax = np.max([Alm.getlmax(r.size, None) for r in residual])
        ell = np.arange(0, lmax + 1)
        weights = 2 * ell + 1
        residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual)])
        # bdata.append([hp.alm2cl(b_) for b_ in np.atleast_2d(b)])
        # fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(fwd_op(x))])
        xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x)])
        # precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs)])

    iter = 0
    while not criterion(iter, x, residual) and iter <= maxiter:
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]
        deltas = [dot_op(searchdir, residual) for searchdir in searchdirs]

        # calculate (D^T A D)^{-1}
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1 + 1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2])
        dTAd_inv = np.linalg.inv(dTAd)

        # search.
        alphas = np.dot(dTAd_inv, deltas)
        for (searchdir, alpha) in zip(searchdirs, alphas):
            x += searchdir * alpha

        # append to cache.
        cache.store(iter, [dTAd_inv, searchdirs, searchfwds])

        # update residual
        iter += 1
        if np.mod(iter, roundoff) == 0:
            residual = b - fwd_op(x)
        else:
            for (searchfwd, alpha) in zip(searchfwds, alphas):
                residual -= searchfwd * alpha

        # plot_stuff(residual, residualdata, bdata, fwddata, xdata, precondata, searchdirs, searchfwds, weights, x)

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # orthogonalize w.r.t. previous searches.
        prev_iters = range(tr(iter), iter)

        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(titer)

            for searchdir in searchdirs:
                proj = [dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds]
                betas = np.dot(prev_dTAd_inv, proj)

                for (beta, prev_searchdir) in zip(betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cache.trim(range(tr(iter + 1), iter))

    return iter
