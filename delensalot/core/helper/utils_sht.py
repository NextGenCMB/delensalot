import numpy as np

def st2mmax(spin, tht, lmax):
    r"""Converts spin, tht and lmax to a maximum effective m, according to libsharp paper polar optimization formula Eqs. 7-8

        For a given mmax, one needs then in principle 2 * mmax + 1 longitude points for exact FFT's


    """
    T = max(0.01 * lmax, 100)
    b = - 2 * spin * np.cos(tht)
    c = -(T + lmax * np.sin(tht)) ** 2 + spin ** 2
    mmax = 0.5 * (- b + np.sqrt(b * b - 4 * c))
    return mmax


def lowprimes(n:np.ndarray):
    """Finds approximations of integer array n from above built exclusively of low prime numbers 2,3,5.

        Python but still ok here for reasonable n

     """
    if np.isscalar(n):
        n = [n]
        scal = True
    else:
        scal = False
    # --- first builds all candidates powers of low primes, sorted
    nmax = 2 ** int(np.ceil(np.log2(np.max(n)))) # at the worst power of two larger will do
    grid = [0]
    n2 = 1
    while n2 <= nmax:
        n3 = 1
        while n2 * n3 <= nmax:
            n_ = n2 * n3
            while n_ <= nmax:
                grid.append(n_)
                n_ *= 5
            n3 *= 3
        n2 *= 2
    grid = np.sort(grid)
    # --- then loop over them to find the smallest larger integer
    unique_ns = np.unique(np.sort(n))
    nuniq = len(unique_ns)
    sols = {}
    i_unsolved = 0
    for n_ in grid:
        while n_ >= unique_ns[i_unsolved]:
            sols[unique_ns[i_unsolved]] = n_
            i_unsolved += 1
            if i_unsolved >= nuniq:
                return sols[n[0]] if scal else np.array([sols[i] for i in n])
    assert 0