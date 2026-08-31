import sys
import numpy as np

import logging
log = logging.getLogger(__name__)

import healpy as hp

from delensalot.core.cg import cd_monitors

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy


import os

def _plots_enabled():
    if os.environ.get('DELENSALOT_CG_PLOT', '').lower() not in ('1', 'true', 'yes'):
        return False
    if any(v in os.environ for v in ('SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID')):
        return False
    try:
        import matplotlib
        return matplotlib.get_backend().lower() not in ('agg', 'pdf', 'ps', 'svg', 'template')
    except ImportError:
        return False


class CGDiagnostics:
    def __init__(self, bands=((2, 30), (30, 300), (300, None))):
        self.on = _plots_enabled()
        self.bands = bands
        self.residual, self.x = [], []

    def record(self, residual, x):
        if not self.on:
            return
        self.residual.append([hp.alm2cl(r) for r in np.atleast_2d(residual)])
        self.x.append([hp.alm2cl(a) for a in np.atleast_2d(x)])

    def band_eps(self, residual, b, dot_op):
        out = {}
        for lo, hi in self.bands:
            num = _banded_dot(residual, residual, lo, hi)
            den = _banded_dot(b, b, lo, hi)
            out[(lo, hi)] = np.sqrt(num / den) if den > 0 else np.nan
        return out

    def show(self):
        if not self.on or not self.residual:
            return
        import matplotlib.pyplot as plt
        n = len(self.residual)
        colors = [plt.cm.rainbow(i / max(1, n - 1)) for i in range(n)]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for i, (res, xs) in enumerate(zip(self.residual, self.x)):
            c = 'black' if i == n - 1 else colors[i]
            for r in res:
                ax[0].plot(r, color=c, lw=1)
            for a in xs:
                ax[1].plot(a, color=c, lw=1)
        for a, lab in zip(ax, [r'$C_\ell^{\rm residual}$', r'$C_\ell^{x}$']):
            a.set_yscale('log'); a.set_xscale('log')
            a.set_xlabel(r'$\ell$'); a.set_ylabel(lab); a.grid(alpha=.3)
        fig.suptitle(f'CG diagnostics ({n} iterations, black = last)')
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def _banded_dot(teblm1, teblm2, lmin, lmax):
    lm = Alm.getlmax(teblm1[1].size, None)
    hi = lm if lmax is None else min(lmax, lm)
    ell = np.arange(lm + 1)
    w = (2 * ell + 1) * ((ell >= lmin) & (ell <= hi))
    tot = 0.
    for a, b in [(teblm1[0], teblm2[0]), (teblm1[1], teblm2[1])]:
        tot += np.sum(alm2cl(a, b, lm, lm, None) * w)
    return tot

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


    def solve(self, soltn, tpn_alm, fwd_op, maxiter=200):
        self.watch = cd_monitors.stopwatch()
        self.iter_tot = 0
        self.prev_eps = None

        if len(tpn_alm) == 3:
            dot_op = self.dot_op3d
        else:
            dot_op = self.dot_op

        monitor = cd_monitors.monitor_basic(dot_op, logger=self.logger, iter_max=self.bstage.iter_max, eps_min=self.bstage.eps_min, d0=dot_op(tpn_alm, tpn_alm))
        solve(soltn, tpn_alm, fwd_op, self.bstage.pre_ops, dot_op, monitor, tr=self.bstage.tr, cacher=cache_mem(), maxiter=maxiter)


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
        # ret += np.sum(alm2cl(blm1, blm2, lmaxs[2], lmaxs[2], None)[0:] * weights[2])
        # print(ret)
        
        return ret
    

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth > self.plogdepth:
            return

        log_str = '   ' * stage.depth + '(%4d, %04d) [%s] (%d, %1.4e)' % (
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

def solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr, cacher, roundoff=25, maxiter=200, diag=None):
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
        diag (CGDiagnostics)        :Per-iteration spectra collector. A no-op unless plotting is
                                explicitly enabled, so it costs nothing in batch runs.

    Note:
        fwd_op, pre_op(s) and dot_op must not modify their arguments!

    """
    diag = diag if diag is not None else CGDiagnostics()

    n_pre_ops = len(pre_ops)
    residual = b - fwd_op(x)
    searchdirs = [op(residual) for op in pre_ops]
    diag.record(residual, x)

    iter = 0
    while not criterion(iter, x, residual) and iter <= maxiter:
        # Forward ops on search directions
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]

        # RHS = D^T r
        deltas = np.array([dot_op(searchdir, residual) for searchdir in searchdirs])

        # Build D^T A D
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(n_pre_ops):
            for ip2 in range(ip1 + 1):
                v = dot_op(searchdirs[ip1], searchfwds[ip2])
                dTAd[ip1, ip2] = v
                dTAd[ip2, ip1] = v

        # small diagonal jitter for safety
        jitter = 1e-12 * np.trace(dTAd) / max(1, dTAd.shape[0])
        dTAd_reg = dTAd + jitter * np.eye(dTAd.shape[0])
        alphas = np.linalg.solve(dTAd_reg, deltas)
        # Update solution
        for searchdir, alpha in zip(searchdirs, alphas):
            x += searchdir * alpha

        # Cache objects needed for orthogonalization
        cacher.store(iter, [np.linalg.inv(dTAd_reg), searchdirs, searchfwds])

        # Update residual
        iter += 1
        if iter % roundoff == 0:
            residual = b - fwd_op(x)
        else:
            for searchfwd, alpha in zip(searchfwds, alphas):
                residual -= searchfwd * alpha

        # New search directions from preconditioner
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # Orthogonalize against previous directions
        prev_iters = range(tr(iter), iter)
        for titer in prev_iters:
            prev_dTAd_inv, prev_searchdirs, prev_searchfwds = cacher.restore(titer)

            for searchdir in searchdirs:
                proj = np.array([
                    dot_op(searchdir, prev_searchfwd)
                    for prev_searchfwd in prev_searchfwds
                ])
                betas = prev_dTAd_inv @ proj
                for beta, prev_searchdir in zip(betas, prev_searchdirs):
                    searchdir -= beta * prev_searchdir

        cacher.trim(range(tr(iter + 1), iter))
        diag.record(residual, x)

    # NOTE the global eps is a (2l+1)-weighted sum over all multipoles, so the
    # low-L block contributes little to the norm and can be entirely unconverged
    # without moving it.
    if iter > maxiter:
        try:
            bands = diag.band_eps(residual, b, dot_op)
            bandstr = ', '.join(f'[{lo},{hi}]={v:.2e}' for (lo, hi), v in bands.items())
        except Exception:
            bandstr = 'unavailable'
        log.warning(f'CG stopped at maxiter={maxiter} without meeting eps_min; '
                    f'per-band relative residual: {bandstr}')

    diag.show()
    return iter