from __future__ import print_function

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from lenscarf import cachers

class BFGS_Hessian(object):
    """
    Class to evaluate the update to inverse Hessian matrix in the L-BFGS scheme.
    (see wikipedia article if nothing else).
    H is B^-1 form that article.
    B_k+1 = B  + yy^t / (y^ts) - B s s^t B / (s^t Bk s))   (all k on the RHS)
    H_k+1 = (1 - sy^t / (y^t s) ) H (1 - ys^t / (y^ts))) + ss^t / (y^t s).

    Determinant of B:
    ln det Bk+1 = ln det Bk + ln( s^ty / s^t B s).

    For quasi Newton, s_k = x_k1 - x_k = - alpha_k Hk grad_k with alpha_k newton step-length.
        --> s^t B s at k is alpha_k^2 g_k H g_k
            s^t y is  - alpha_k (g_k+1 - g_k) H g_k
    This leads to ln|B_k + 1| = ln |B_k| + ln(1 - 1/alpha_k g_k+1 H g_k / (gk H gk))    
    """

    def __init__(self, cacher:cachers.cacher, apply_H0k:callable, paths2ys, paths2ss, dot_op:callable,
                        L=100000, apply_B0k=None, verbose=True):
        """
            Args:
                apply_H0k: user supplied function(x,k), applying a zeroth order estimate of the inverse Hessian to x atiter k.
                paths2ys: list of paths to the y vectors. y_k = grad_k+1 - grad_k
                paths2ss: list of paths to the s vectors. s_k = x_k+1 - xk_k
                dot_op: callable with 2 arguments giving scalar product between two vector (e.g. np.sum)
        H is inverse Hessian, not Hessian.
        """
        self.cacher = cacher
        self.paths2ys = paths2ys
        self.paths2ss = paths2ss
        self.L = L
        self.applyH0k = apply_H0k
        self.applyB0k = apply_B0k
        self.verbose = verbose
        if dot_op is None:
            dot_op = np.sum
        self.dot_op = dot_op

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
            log.info('Linked y vector ' + str(path2y) + ' to Hessian')
            log.info('Linked s vector ' + str(path2s) + ' to Hessian')

    def _save_alpha(self, alpha, i):
        fname = 'temp_alpha_%s'%i
        self.cacher.cache(fname, alpha)
        return

    def _load_alpha(self, i):
        """Loads, and remove, bfgs alpha from disk.

        """
        fname = 'temp_alpha_%s'%i
        assert self.cacher.is_cached(fname)
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

    def get_mHkgk(self, gk, k, output_fname=None):
        """
        Obtains - H_k g_k with L-BFGS two-loop recursion.
        :param gk: grad f(x_k)
        :param k: iterate index
        :return: - H_k g_k according to L-BFGS.
        If output_fname is set then output is saved in file and nothing is returned.
        Should be fine with k == 0
        """
        q = gk.copy()
        rho = lambda i: 1. / self.dot_op(self.s(i), self.y(i))
        for i in range(k - 1, np.max([-1, k - self.L - 1]), -1):
            alpha_i = rho(i) * self.dot_op(self.s(i), q)
            q -= alpha_i * self.y(i)
            self._save_alpha(alpha_i, i)

        r = self.applyH0k(q, k)
        for i in range(np.max([0, k - self.L]), k):
            beta = rho(i) * self.dot_op(self.y(i), r)
            r += self.s(i) * (self._load_alpha(i) - beta)
        if output_fname is None: return -r
        self.cacher.cache(output_fname, -r)
        return

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
