import os, sys
from os.path import join as opj

import logging
log = logging.getLogger(__name__)
import numpy as np

from delensalot.core import cachers

from delensalot.utils import cli
from delensalot.utility.utils_hp import almxfl, alm2cl, Alm
from delensalot.core.MAP.context import get_computation_context

import matplotlib.pyplot as plt

class BFGSHessian(object):
    """
    Class to evaluate the update to inverse Hessian matrix in the L-BFGS scheme.
    (see wikipedia article if nothing else).


    H is $B^-1$ form that article.
    $$B_k+1 = B  + yy^t / (y^ts) - B s s^t B / (s^t Bk s))$$   (all k on the RHS)
    $$H_k+1 = (1 - sy^t / (y^t s) ) H (1 - ys^t / (y^ts))) + ss^t / (y^t s)$$.

    Determinant of B:
    $$ln det Bk+1 = ln det Bk + ln( s^ty / s^t B s)$$.
    For quasi Newton, $$s_k = x_k1 - x_k = - alpha_k Hk grad_k with alpha_k$$ newton step-length.
        --> $$s^t B s at k is alpha_k^2 g_k H g_k$$
            $$s^t y is  - alpha_k (g_k+1 - g_k) H g_k$$
    This leads to $$ln|B_k + 1| = ln |B_k| + ln(1 - 1/alpha_k g_k+1 H g_k / (gk H gk))$$

    """

    def __init__(self, h0:np.array=np.array([0]), applyH0k:callable=None, applyB0k:callable = None, paths2ys:dict={}, paths2ss:dict={}, dot_op:callable = None,
                        L=100, subs_layout=[], verbose=True, cacher:cachers.cacher=None):
        """
            Args:
                apply_H0k: user supplied function(x,k), applying a zeroth order estimate of the inverse Hessian to x atiter k.
                paths2ys: list of paths to the y vectors. y_k = grad_k+1 - grad_k
                paths2ss: list of paths to the s vectors. s_k = x_k+1 - xk_k
                dot_op: callable with 2 arguments giving scalar product between two vector (e.g. np.sum)
        H is inverse Hessian, not Hessian.
        """

         # this is the 1D-solution. For 2D, I will pass apply_h0k and apply_b0k
        if len(h0) == 1: self.lmax_qlm = h0[0]
        if applyH0k is None: apply_H0k = lambda rlm, kr: almxfl(rlm, h0, self.lmax_qlm, False)
        if applyB0k is None: applyB0k = lambda rlm, kr: almxfl(rlm, cli(h0), self.lmax_qlm, False)
        self.applyH0k = applyH0k
        self.applyB0k = applyB0k

        self.cacher = cacher
        self.paths2ys = paths2ys
        self.paths2ss = paths2ss
        self.L = L

        self.verbose = verbose
        if dot_op is None:
            dot_op = np.sum
        self.dot_op = dot_op

        self.use_powell_damping = False
        self.subs_layout = subs_layout

    def y(self, n):
        return self.cacher.load(self.paths2ys[n])

    def s(self, n):
        return self.cacher.load(self.paths2ss[n])

    def add_ys(self, path2y, path2s, k):
        assert self.cacher.is_cached(path2y), path2y
        assert self.cacher.is_cached(path2s), path2s
        self.paths2ys[k] = path2y
        self.paths2ss[k] = path2s
        if self.verbose:
            log.debug('Linked y vector {} to Hessian'.format(str(path2y)))
            log.debug('Linked s vector {} to Hessian'.format(str(path2s)))

    def _save_alpha(self, alpha, i):
        ctx, isnew = get_computation_context()
        fname = 'temp_alpha_%s_%s'%(i, ctx.idx)
        self.cacher.cache(fname, alpha)
        return

    def _load_alpha(self, i):
        """Loads, and remove, bfgs alpha from disk.

        """
        ctx, isnew = get_computation_context()
        fname = 'temp_alpha_%s_%s'%(i, ctx.idx)
        assert self.cacher.is_cached(fname), fname
        ret = self.cacher.load(fname)
        self.cacher.remove(fname)
        return ret

    def applyH(self, x, k, _depth=0):
        """
        Recursive calculation of H_k x, for any x.
        This uses the product form update H_new = (1 - rho s y^t) H (1 - rho y s^t) + rho ss^t
        :param x: vector to apply the inverse Hessian to
        :param k: iter level. Output is H_k x.
        :param _depth : internal, for internal bookkeeping.
        :return:
        """
        if k <= 0 or _depth >= self.L or self.L == 0: return self.applyH0k(x, k)
        s = self.s(k - 1)
        y = self.y(k - 1)
        rho = 1. / self.dot_op(s, y)
        Hv = self.applyH(x - rho * y * self.dot_op(x, s), k - 1, _depth=_depth + 1)
        return Hv - s * (rho * self.dot_op(y, Hv)) + rho * s * self.dot_op(s, x)

    def get_gk(self, k, alpha_k0):
        """
        Reconstruct gradient at xk, given the first newton step length at step max(0,k-L)
        ! this is very badly behaved numerically.
        """
        assert self.applyB0k is not None
        ret = -self.applyB0k(self.s(max(0, k - self.L)),max(0,k-self.L)) / alpha_k0
        for j in range(max(0, k - self.L), k):
            ret += self.y(j)
        return ret

    def get_sBs(self, k, alpha_k, alpha_k0):
        """
        Reconstruct s^Bs at x_k, given the first newton step length at step max(0,k-L) and current step alpha_k.
        """
        return - alpha_k * self.dot_op(self.s(k), self.get_gk(k, alpha_k0))

    def get_lndet_update(self, k, alpha_k, alpha_k0):
        """
        Return update to B log determinant, lndet B_k+1 = lndet B_k + output.
        """
        return np.log(self.dot_op(self.y(k), self.s(k)) / self.get_sBs(k, alpha_k, alpha_k0))

    def sample_Gaussian(self, k, x_0, rng_state=None):
        """
        sample from a MV zero-mean Gaussian with covariance matrix H, at iteration level k,
        given input x_0 random vector with covariance H_0.
        Since H is the inverse Hessian, then H is roughly the covariance matrix of the parameters in a line search.
        :param k:
        :param x_0:
        :return:
        """
        ret = x_0.copy()
        rho = lambda j: 1. / self.dot_op(self.s(j), self.y(j))
        if rng_state is not None: np.random.set_state(rng_state)
        eps = np.random.standard_normal((len(range(np.max([0, k - self.L]), k)), 1))

        for idx, i in enumerate(range(np.max([0, k - self.L]), k)):
            ret = ret - self.s(i) * self.dot_op(self.y(i), ret) * rho(i) + np.sqrt(rho(i)) * self.s(i) * eps[idx]
        return ret

    def get_mHkgk(self, gk, k, output_fname=None):
        """
        Computes −Hₖ gₖ using the standard L-BFGS two-loop recursion.
        Optionally applies scale-dependent Powell damping.
        """
        q = gk.copy()
        rho = lambda i: 1.0 / self.dot_op(self.s(i), self.y(i))

        # backward pass
        for i in range(k - 1, max(-1, k - self.L - 1), -1):
            s0, y0 = self.s(0), self.y(0)
            si, yi = self.s(i), self.y(i)
            sy = self.dot_op(si, yi)

            if self.use_powell_damping:
                # modify yi in-place according to Powell criterion
                yi, sy = self._apply_scale_dependent_damping(s0, y0, si, yi, k)
                self.visualize_powell_damping(si, yi, self.applyB0k(si,k))
                # self.visualize_powell_damping_curvatureFromFirstIncrement(s0, y0, self.applyB0k(s0, 0), si, yi, self.applyB0k(si,k))

            # skip degenerate pairs
            if sy <= 0 or not np.isfinite(sy):
                print('could skip pair %d with s^T y = %.5e'%(i, sy))
                # NOTE if i really skip, need to start tracking which indices I actually skipped, to pass this info to the forward pass below
                # continue

            alpha_i = rho(i) * self.dot_op(si, q)
            q -= alpha_i * yi
            self._save_alpha(alpha_i, i)

        # 1. apply H0
        r = self.applyH0k(q, k)
        if self.use_powell_damping:
            r = self._attenuate_lowL(r, L0=10, L1=30, sharp=3.0)
           
        # forward pass
        for i in range(max(0, k - self.L), k):
            si, yi = self.s(i), self.y(i)
            sy = self.dot_op(si, yi)
            if sy <= 0 or not np.isfinite(sy):
                continue
            beta = rho(i) * self.dot_op(yi, r)
            r += si * (self._load_alpha(i) - beta)

        if output_fname is None:
            return -r
        self.cacher.cache(output_fname, -r)
        return
    
    def visualize_powell_damping(self, s, y, yB, L0=10, L1=30, show_damped=True):
        """
        Visualize strong low-L damping (smooth taper between L0–L1) consistent with
        the current _apply_scale_dependent_damping() implementation.

        Parameters
        ----------
        s, y, yB : 1D complex arrays
            Concatenated alm arrays of all subfields (same layout).
            yB = B0·s (baseline curvature prediction).
        L0, L1 : int
            Full damping for L<=L0, smooth transition to zero by L1.
        show_damped : bool
            If True, also plot damped |sᵀy′|(L).
        """
        import matplotlib.pyplot as plt
        subs_layout = self.subs_layout
        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(subs_layout)))
        off = 0

        for fi, (lmax, mmax) in enumerate(subs_layout):
            size = Alm.getsize(lmax, mmax)
            s_blk  = s[off:off+size]
            y_blk  = y[off:off+size]
            yB_blk = yB[off:off+size]
            off += size

            # --- strong taper window: full damping below L0, none above L1
            Ls = np.arange(lmax + 1)
            x = (Ls - L0) / max(1, (L1 - L0))
            fL = 0.5 * (1.0 - np.tanh(3.0 * x))  # sharp low-L transition
            wL = fL

            # --- curvature spectra ---
            cl_sy  = alm2cl(s_blk, y_blk,  lmax, mmax, lmax)
            cl_sBs = alm2cl(s_blk, yB_blk, lmax, mmax, lmax)

            if show_damped:
                y_damped = almxfl(y_blk, 1.0 - wL, lmax, False) + almxfl(yB_blk, wL, lmax, False)
                cl_sy_damped = alm2cl(s_blk, y_damped, lmax, mmax, lmax)
            else:
                cl_sy_damped = None

            c = colors[fi]
            label_base = f"Field {fi}"

            plt.loglog(Ls, np.abs(cl_sy),  color=c, lw=1.2, label=f"{label_base} |sᵀy| (original)")
            plt.loglog(Ls, np.abs(cl_sBs), color=c, ls='--', alpha=0.6, label=f"{label_base} |sᵀB₀s|")
            if show_damped and cl_sy_damped is not None:
                plt.loglog(Ls, np.abs(cl_sy_damped), color=c, lw=1.8, alpha=0.8,
                        linestyle='-.', label=f"{label_base} |sᵀy′| (damped)")

            # show the damping window on secondary y-axis
            ax2 = plt.gca().twinx()
            ax2.plot(Ls, wL, color=c, lw=1.0, alpha=0.3)
            ax2.set_ylabel("Damping weight w(L)", color="gray", fontsize=8)
            ax2.set_ylim(-0.05, 1.05)

            print(f"Field {fi}: strong damping up to L≈{L1}, full below L≈{L0}")

        plt.xlabel(r'Multipole $L$')
        plt.ylabel(r'Power-like curvature $|s^T y|(L)$')
        plt.title("Strong low-L damping profile (consistent with tanh taper)")
        plt.ylim(1e-8, 1e6)
        plt.xlim(1, max(subs_layout, key=lambda x: x[0])[0])
        plt.legend(frameon=False, fontsize='x-small', ncol=2)
        plt.tight_layout()
        plt.show()

    def get_curvature_spectra(self, grad_tot, k=None, tau0=1e0, L0=10, L1=30):
        """
        Compute measured and expected curvature spectra for all stored (s, y) pairs.
        Automatically links them into the BFGS memory if missing.

        Parameters
        ----------
        k : int or None
            Maximum iteration index to include (default = all cached pairs).
        tau0, L0, L1 : float
            Damping parameters for consistency (only affect returned tau_L).

        Returns
        -------
        curvature_data : list of dict
            Each entry has keys:
            'iter', 'field', 'Ls', 'cl_sy', 'cl_sBs', 'mask_high', 'tau_L'
        """
        curvature_data = []

        subs_layout = self.subs_layout
        s0 = self.s(0)
        yB0 = self.applyB0k(self.s(0), 0)
        for i in range(len(self.paths2ys)):
            si, yi = self.s(i), self.y(i)
            yB = self.applyB0k(si, i)

            off = 0
            for fi, (lmax, mmax) in enumerate(subs_layout):
                size = Alm.getsize(lmax, mmax)
                s_blk  = si[off:off+size]
                s0_blk = s0[off:off+size]
                y_blk  = yi[off:off+size]
                yB_blk = yB[off:off+size]
                yB0_blk = yB0[off:off+size]
                off += size

                Ls = np.arange(lmax + 1)
                fL = 0.5 * (1.0 + np.tanh((L1 - Ls) / max(1, (L1 - L0))))
                tau_L = tau0 * fL

                cl_sy  = alm2cl(s_blk, y_blk,  lmax, mmax, lmax)
                cl_sBs = alm2cl(s_blk, yB_blk, lmax, mmax, lmax)
                cl_sBs0 = alm2cl(s0_blk, yB0_blk, lmax, mmax, lmax)
                mask_high = np.abs(cl_sy) > (1.0 / tau_L) * np.abs(cl_sBs0)

                curvature_data.append({
                    'iter': i,
                    'field': fi,
                    'Ls': Ls,
                    'cl_sy': cl_sy,
                    'cl_sBs': cl_sBs,
                    'cl_sBs0': cl_sBs0,
                    'mask_high': mask_high,
                    'tau_L': tau_L,
                    'gamma_k': None, #estimate_gamma_from_step(i),
                    's0': s0_blk,
                    'si': s_blk,
                    'yi': y_blk,
                })

        return curvature_data

    def visualize_powell_damping_curvatureFromFirstIncrement(self, s0, y0, yB0, s, y, yB, tau0=1e0, L0=10, L1=30, show_damped=True):
        """
        Visualize Powell damping (high-curvature capping) for concatenated alm vectors.

        Parameters
        ----------
        s, y, yB : 1D complex arrays
            Concatenated alm arrays of all subfields (same layout).
            yB = B0·s (baseline curvature prediction).
        tau0, L0, L1 : float
            Damping parameters. tau_L = tau0 * f_L with tanh taper.
        show_damped : bool
            If True, compute and overlay damped |sᵗy′|(L).
        """
        import matplotlib.pyplot as plt
        subs_layout = self.subs_layout
        plt.figure(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(subs_layout)))
        off = 0

        for fi, (lmax, mmax) in enumerate(subs_layout):
            size = Alm.getsize(lmax, mmax)
            s_blk  = s[off:off+size]
            s0_blk  = s0[off:off+size]
            y_blk  = y[off:off+size]
            y0_blk  = y0[off:off+size]
            yB_blk = yB[off:off+size]
            yB0_blk = yB0[off:off+size]
            off += size

            # --- per-L damping profile ---
            Ls = np.arange(lmax + 1)
            fL = 0.5 * (1.0 + np.tanh((L1 - Ls) / max(1, (L1 - L0))))
            tau_L = tau0 * fL  # stronger damping at low-L

            # --- Curvature spectra ---
            cl_sy  = alm2cl(s_blk, y_blk,  lmax, mmax, lmax)
            cl_sBs = alm2cl(s_blk, yB_blk, lmax, mmax, lmax)
            cl_sBs0 = alm2cl(s0_blk, yB0_blk, lmax, mmax, lmax)

            # --- Identify modes where curvature is too strong ---
            mask = np.abs(cl_sy) > (1.0 / tau_L) * np.abs(cl_sBs0)

            # --- Compute damped curvature if requested ---
            if show_damped:
                theta_L = np.ones_like(Ls, dtype=float)
                if np.any(mask):
                    idx = np.where(mask)[0]
                    # smooth blend back to baseline curvature (reduce y amplitude)
                    num = (1.0 - tau_L[idx]) * np.abs(cl_sBs0[idx])
                    den = np.maximum(np.abs(cl_sy[idx]) - np.abs(cl_sBs0[idx]), 1e-30)
                    theta_L[idx] = np.clip(num / den, 0.0, 1.0)
                y_damped = (
                    almxfl(y_blk,  theta_L, lmax, False)
                    + almxfl(yB_blk, 1.0 - theta_L, lmax, False)
                )
                cl_sy_damped = alm2cl(s_blk, y_damped, lmax, mmax, lmax)
            else:
                cl_sy_damped = None

            # --- Plot per-field ---
            c = colors[fi]
            label_base = f"Field {fi}"
            plt.loglog(Ls, np.abs(cl_sy), color=c, lw=1.2, label=f"{label_base} |sᵗy|")
            plt.loglog(Ls, np.abs(cl_sBs0) / tau_L, color=c, ls='--', alpha=0.6, label=f"{label_base} (1/τₗ)|sᵗBs| cap")
            plt.loglog(Ls, np.abs(cl_sBs) / tau_L, color=c, ls='--', alpha=0.2)
            if show_damped and cl_sy_damped is not None:
                plt.loglog(Ls, np.abs(cl_sy_damped), color=c, lw=1.8, alpha=0.8,
                        linestyle='-.', label=f"{label_base} |sᵗy′| (damped)")

            # --- Shade region where damping active ---
            if np.any(mask):
                plt.fill_between(Ls, 1e-12, 1e12, where=mask, color=c, alpha=0.12, edgecolor=None)
                frac = np.mean(mask)
                print(f"Field {fi}: damping active in {frac*100:.1f}% of L-modes")

        plt.xlabel(r'Multipole $L$')
        plt.ylabel(r'Power-like curvature')
        plt.title("High-curvature Powell damping per field")
        plt.ylim(1e-8, 1e6)   # expanded for your large low-L curvature
        plt.xlim(1, max(subs_layout, key=lambda x: x[0])[0])
        plt.legend(frameon=False, fontsize='x-small', ncol=2)
        plt.tight_layout()
        plt.show()
