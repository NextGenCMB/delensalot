import sys
import numpy as np

import logging
log = logging.getLogger(__name__)

import healpy as hp

from delensalot.core.cg import cd_monitors

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy


def plot_stuff(residual, residualdata, bdata, fwddata, xdata, precondata, searchdirs, searchfwds, weights, x):
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    from IPython.display import clear_output
    def get_rainbow_colors(num_items):
        cmap = plt.cm.rainbow  # Choose the rainbow colormap
        return [cmap(i / (num_items - 1)) for i in range(num_items)]
    residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual)])
    fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(searchfwds[0])])
    xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x)])
    precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs[0])])
    colors = get_rainbow_colors(len(residualdata)+1)
    clear_output(wait=True)
    plt.figure(figsize=(10, 6))
    for linei, line2 in enumerate(residualdata):
        for resi, res in enumerate(line2):
            color = colors[linei] if linei < len(residualdata) - 1 else 'black'
            plt.plot(res, label='iter %d'%(linei+1), color=color, ls='-' if resi else '-')
    # plt.legend(title='CG search')
    plt.ylabel(r'$C_\ell^{\rm residual}$')
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

    plt.figure(figsize=(10, 6))
    for linei, line2 in enumerate(xdata):
        for res in line2:
            plt.plot(res, label='iter %d'%(linei+1), color=colors[linei], ls='-' if resi else '-')
    # plt.legend(title='CG search')
    plt.ylabel(r'$C_\ell^{\rm x}$')
    plt.xlabel(r'$\ell$')
    plt.yscale('log')
    plt.show()

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

class MultigridStage(object):
    def __init__(self, ids, pre_ops_descr, lmax, nside, iter_max, eps_min, tr):
        self.depth = ids
        self.pre_ops_descr = pre_ops_descr
        self.lmax = lmax
        self.nside = nside
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.tr = tr
        self.pre_ops = []


#NOTE this is actually not a multigrid anymore. pure diagonal here.
class ConjugateGradient:
    def __init__(self, pre_op_diag, chain_descr, s_cls, debug_log_prefix=None, plogdepth=0):
        self.debug_log_prefix = debug_log_prefix
        self.plogdepth = plogdepth
        self.chain_descr = chain_descr
        self.s_cls = s_cls
        stages = {}
        for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr] in self.chain_descr:
            stages[id] = MultigridStage(id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr)
            for pre_op_descr in pre_ops_descr:  # recursively add all stages to stages[0]
                stages[id].pre_ops.append(pre_op_diag)
        self.bstage = stages[0]  # these are the pre_ops called in cd_solve
        self.logger = (lambda iter, eps, stage=self.bstage, **kwargs: self.log(stage, iter, eps, **kwargs))


    def solve(self, soltn, tpn_alm, fwd_op):
        self.watch = cd_monitors.stopwatch()
        self.iter_tot = 0
        self.prev_eps = None

        if len(tpn_alm) == 3:
            dot_op = self.dot_op3d
        else:
            dot_op = self.dot_op

        monitor = cd_monitors.monitor_basic(dot_op, logger=self.logger, iter_max=self.bstage.iter_max, eps_min=self.bstage.eps_min, d0=dot_op(tpn_alm, tpn_alm))
        solve(soltn, tpn_alm, fwd_op, self.bstage.pre_ops, dot_op, monitor, tr=self.bstage.tr, cacher=cache_mem())


    def dot_op(self, elm1, elm2):
        lmax = Alm.getlmax(elm1.size, None)
        ell = np.arange(0, lmax + 1)
        weight = 2 * ell + 1
        ret =  np.sum(alm2cl(elm1, elm2, lmax, lmax, None) * weight)
        return ret
    

    def dot_op3d(self, teblm1, teblm2):
        lmaxs = [Alm.getlmax(te.size, None) for te in teblm1]
        ells = [np.arange(0, lmax + 1) for lmax in lmaxs]
        weights = [2 * ell + 1 for ell in ells]
        tlm1, elm1, blm1 = teblm1
        tlm2, elm2, blm2 = teblm2
        
        ret =  np.sum(alm2cl(tlm1, tlm2, lmaxs[0], lmaxs[0], None)[0:] * weights[0])
        ret += np.sum(alm2cl(elm1, elm2, lmaxs[1], lmaxs[1], None)[0:] * weights[1])
        ret += np.sum(alm2cl(blm1, blm2, lmaxs[2], lmaxs[2], None)[0:] * weights[2])
        
        return ret
    

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth > self.plogdepth:
            return

        log_str = '   ' * stage.depth + '(%4d, %04d) [%s] (%d, %1.2e)' % (
        stage.nside, stage.lmax, str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

        if self.debug_log_prefix is not None:
            log = open(self.debug_log_prefix + 'stage_all.dat', 'a')
            log.write(log_str)
            log.close()

            if stage.depth == 0:
                f_handle = self.debug_log_prefix + 'stage_soltn_' + str(stage.depth) + '_%04d'%iter +'.npy'
                np.save(f_handle,  kwargs['soltn'])

                #f_handle = self.debug_log_prefix + 'stage_resid_' + str(stage.depth) + '.npy'
                #np.save(f_handle, kwargs['resid']]])
                #f_handle.close()

            log_str = '%05d %05d %10.6e %05d %s\n' % (self.iter_tot, int(elapsed), eps, iter, str(elapsed))
            log = open(self.debug_log_prefix + 'stage_' + str(stage.depth) + '.dat', 'a')
            log.write(log_str)
            log.close()

            if (self.prev_eps is not None) and (self.prev_stage.depth > stage.depth):
                log_final_str = '%05d %05d %10.6e %s\n' % (
                self.iter_tot - 1, int(self.prev_elapsed), self.prev_eps, str(self.prev_elapsed))

                log = open(self.debug_log_prefix + 'stage_final_' + str(self.prev_stage.depth) + '.dat', 'a')
                log.write(log_final_str)
                log.close()

            self.prev_stage = stage
            self.prev_eps = eps
            self.prev_elapsed = elapsed


def PTR(p, t, r):
    return lambda i: max(0, i - max(p, int(min(t, np.mod(i, r)))))


tr_cg = (lambda i: i - 1)
tr_cd = (lambda i: 0)


def solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr, cacher, roundoff=25):
    maxiter = 35
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
    searchdirs = [op(residual) for op in pre_ops]

    lmax = np.max([Alm.getlmax(r.size, None) for r in residual])
    ell = np.arange(0, lmax + 1)
    weights = 2 * ell + 1
    residualdata.append([hp.alm2cl(res)*weights for res in np.atleast_2d(residual)])
    bdata.append([hp.alm2cl(b_) for b_ in np.atleast_2d(b)])
    fwddata.append([hp.alm2cl(fwd_) for fwd_ in np.atleast_2d(fwd_op(x))])
    xdata.append([hp.alm2cl(x_) for x_ in np.atleast_2d(x)])
    precondata.append([hp.alm2cl(precon_) for precon_ in np.atleast_2d(searchdirs)])
    iter = 0

    lmax = hp.Alm.getlmax(residual[0].size)
    ell = np.arange(0, lmax + 1)
    
    while not criterion(iter, x, residual) and iter <= maxiter:
        
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]
        deltas = [dot_op(searchdir, residual) for searchdir in searchdirs]

        # calculate (D^T A D)^{-1}
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1 + 1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2])
        dTAd_inv = np.linalg.inv(dTAd)


        searchfwd_alm = searchfwds[0][1] # NOTE zeroth grid direction of elm  # Assuming single search direction
        # Compute per-ell power spectrum
        cl_dTAd = hp.alm2cl(searchfwd_alm)  # This gives a power spectrum over ell-modes
        cl_dTAd[cl_dTAd <= 0] = np.min(cl_dTAd[cl_dTAd > 0]) * 1e-6
        # Compute per-ell condition number
        cond_num_ell = np.zeros(lmax + 1)
        for ell in range(1, lmax + 1):  # Avoid ell=0
            cond_num_ell[ell] = cl_dTAd[ell] / np.min(cl_dTAd[ell:])  # Condition number per ell

        # Compute global condition number (max over all ell)
        global_cond_num = np.max(cond_num_ell)


        # search.
        alphas = np.dot(dTAd_inv, deltas)
        for (searchdir, alpha) in zip(searchdirs, alphas):
            x += searchdir * alpha

        # append to cache.
        cacher.store(iter, [dTAd_inv, searchdirs, searchfwds])
        
        # update residual
        iter += 1
        if np.mod(iter, roundoff) == 0:
            residual = b - fwd_op(x)
        else:
            for (searchfwd, alpha) in zip(searchfwds, alphas):
                residual -= searchfwd * alpha
        if log.getEffectiveLevel() in [logging.INFO, logging.DEBUG]:
            plot_stuff(residual, residualdata, bdata, fwddata, xdata, precondata, searchdirs, searchfwds, weights, x)
            import matplotlib.pyplot as plt
            ell = np.arange(len(cond_num_ell))
            plt.plot(ell[20:51], cond_num_ell[20:51])
            print(f"Iteration {iter}: Global Condition Number = {global_cond_num:.2f}")
            plt.show()

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # orthogonalize w.r.t. previous searches.
        prev_iters = range(tr(iter), iter)

        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cacher.restore(titer)

            for searchdir in searchdirs:
                proj = [dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds]
                betas = np.dot(prev_dTAd_inv, proj)

                for (beta, prev_searchdir) in zip(betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cacher.trim(range(tr(iter + 1), iter))

    return iter