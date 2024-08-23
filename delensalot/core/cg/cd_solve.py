"""Flexible conjugate directions solver module.

"""
import numpy as np


def PTR(p, t, r):
    return lambda i: max(0, i - max(p, int(min(t, np.mod(i, r)))))


tr_cg = (lambda i: i - 1) # Conjugate gradient
tr_cd = (lambda i: 0) # Conjugate descent ? 


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


def cd_solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr, cache=cache_mem(), roundoff=25):
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
        
    LL comments:
        Is the preconditioner updated in the loops? 
        Does the calls to pre_op will also make cd_solve?
        
        Adding comments following notations of wikipedia article on Conjugate Gradient method.

    """
    n_pre_ops = len(pre_ops)

    # r_0 = b - A x_0, where A is the fwd operation, r_0 is the initial residual, and x_0 the inital guess
    residual = b - fwd_op(x)

    # z_0 = M^-1 r_0, where M is the preconditioner (dense or diag)
    searchdirs = [op(residual) for op in pre_ops] # If the pre_op is multigrid, this will call cd_solve recursively
    # p_0 = z_0, where p_0 is the initial search direction

    iter = 0
    while not criterion(iter, x, residual):
        #TODO This combines all the preconditioned search directions into a single search direction ?
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs] # A p_k
        deltas = [dot_op(searchdir, residual) for searchdir in searchdirs] # \delta_{k} = r_k^T z_k

        # calculate (p_{k}^T A p_k)^{-1} 
        dTAd = np.zeros((n_pre_ops, n_pre_ops))
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1 + 1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2]) # p_{k}^T A p_k
        dTAd_inv = np.linalg.inv(dTAd) # (p_{k}^T A p_k)^{-1} 

        # search.
        alphas = np.dot(dTAd_inv, deltas) # alpha_{k} = r_k^T z_k / (p_{k}^T A p_k)
        for (searchdir, alpha) in zip(searchdirs, alphas):
            x += searchdir * alpha # x_{k+1} = x_k + \alpha_k p_k

        # append to cache.
        cache.store(iter, [dTAd_inv, searchdirs, searchfwds])

        # update residual
        iter += 1
        if np.mod(iter, roundoff) == 0:
            # In this case compute exact residual 
            residual = b - fwd_op(x)    # r_{k+1 } = b - A x_{k+1}
        else:
            for (searchfwd, alpha) in zip(searchfwds, alphas):
                residual -= searchfwd * alpha # r_{k+1} = r_k - \alpha_k A p_k

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops] # z_{k+1} = M^{-1} r_{k+1}

        # orthogonalize w.r.t. previous searches.
        prev_iters = range(tr(iter), iter)
        # For CG we have only one previous search direction, but for CD we have multiple previous search directions.

        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(titer)

            for searchdir in searchdirs:
                proj = [dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds] # z_{k+1}^T A p_k
                betas = np.dot(prev_dTAd_inv, proj) # beta_{k} = z_{k+1}^T A p_k / (p_{k}^T A p_k)

                for (beta, prev_searchdir) in zip(betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta # p_{k+1} = z_{k+1} - \beta_k p_k

        # clear old keys from cache
        cache.trim(range(tr(iter + 1), iter))

    return iter
