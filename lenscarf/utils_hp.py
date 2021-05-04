import numpy as np

def getlmax(s, mmax=None):
    """Returns the lmax corresponding to a given healpy array size.

    Parameters
    ----------
    s : int
      Size of the array
    mmax : None or int, optional
      The maximum m, defines the alm layout. Default: lmax.

    Returns
    -------
    lmax : int
      The maximum l of the array, or -1 if it is not a valid size.
    """
    if mmax is not None and mmax >= 0:
        x = (2 * s + mmax ** 2 - mmax - 2) / (2 * mmax + 2)
    else:
        x = (-3 + np.sqrt(1 + 8 * s)) / 2
    if x != np.floor(x):
        return -1
    else:
        return int(x)