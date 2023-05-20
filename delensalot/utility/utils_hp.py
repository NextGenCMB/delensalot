import numpy as np
from numpy.random import default_rng
rng = default_rng()


def almxfl(alm:np.ndarray, fl:np.ndarray, mmax:int or None, inplace:bool):
    """Multiply alm by a function of l.

    Parameters
    ----------
    alm : array
      The alm to multiply
    fl : array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax : None or int
      The maximum m defining the alm layout. Default: lmax.
    inplace : bool
      If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
    alm : array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.

    """
    lmax = Alm.getlmax(alm.size, mmax)
    if mmax is None or mmax < 0:
        mmax = lmax
    assert fl.size > lmax, (fl.size, lmax)
    if inplace:
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            alm[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return
    else:
        ret = np.copy(alm)
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            ret[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return ret


def gauss_beam(fwhm:float, lmax:int):
    """Gaussian beam

    Parameters
    ----------
    fwhm : float
        The full-width half-maximum in radians of the beam
    lmax : int
        Maximum multipole of the beam

    Returns
    -------
    bl: ndarray
        The beam transfer function from multipole 0 to lmax


    """
    l = np.arange(lmax + 1)
    bl = np.exp(-0.5 * l * (l + 1) * (fwhm / np.sqrt(8.0 * np.log(2.0))) ** 2)
    return bl

def synalm(cl:np.ndarray, lmax:int, mmax:int or None):
    """Creates a Gaussian field alm from input cl array

    Parameters
    ----------
    cl : ndarray
        The power spectrum of the map
    lmax : int
        Maximum multipole simulated
    mmax: int
        Maximum m defining the alm layout, defaults to lmax if None or < 0

    Returns
    -------
    alm: ndarray
        harmonic coefficients of Gaussian field with lmax, mmax parameters

    """
    assert lmax + 1 <= cl.size
    if mmax is None or mmax < 0:
        mmax = lmax
    alm_size = Alm.getsize(lmax, mmax)
    alm = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)
    almxfl(alm, np.sqrt(cl[:lmax+1] * 0.5), mmax, True)
    real_idcs = Alm.getidx(lmax, np.arange(lmax + 1, dtype=int), 0)
    alm[real_idcs] = alm[real_idcs].real * np.sqrt(2.)
    return alm

def alm2cl(alm:np.ndarray, blm:np.ndarray or None, lmax:int or None, mmax:int or None, lmaxout:int or None):
    """Auto- or cross-power spectrum between two alm arrays

    Parameters
    ----------
    alm : ndarray
        First alm harmonic coefficient array
    blm : ndarray or None
        Second alm harmonic coefficient array, can set this to same alm object or to None if same as alm
    lmax : int or None
        Maximum multipole defining the alm layout
    mmax: int or None
        Maximum m defining the alm layout, defaults to lmax if None or < 0
    lmaxout: the spectrum is calculated down to this multipole (defaults to lmax is None)

    Returns
    -------
    cl: ndarray
        (cross-)power of the input alm and blm arrays

    """
    if lmax is None: lmax = Alm.getlmax(alm.size, mmax)
    if lmaxout is None: lmaxout = lmax
    if mmax is None: mmax = lmax
    assert lmax == Alm.getlmax(alm.size, mmax), (lmax, Alm.getlmax(alm.size, mmax))
    lmaxout_ = min(lmaxout, lmax)
    if blm is not alm: # looks like twice faster than healpy implementation... ?!
        assert lmax == Alm.getlmax(blm.size, mmax), (lmax, Alm.getlmax(blm.size, mmax))
        cl = 0.5 * alm[:lmaxout_ + 1].real * blm[:lmaxout_ + 1].real
        for m in range(1, min(mmax, lmaxout_) + 1):
            m_idx = Alm.getidx(lmax,  m, m)
            a = alm[m_idx:m_idx + lmaxout_ - m + 1]
            b = blm[m_idx:m_idx + lmaxout_ - m + 1]
            cl[m:] += a.real * b.real + a.imag * b.imag
    else:
        a = alm[:lmaxout_ + 1].real
        cl = 0.5 * a.real * a.real
        for m in range(1, min(mmax, lmaxout_) + 1):
            m_idx = Alm.getidx(lmax,  m, m)
            a = alm[m_idx:m_idx + lmaxout_ - m + 1]
            cl[m:] += a.real * a.real + a.imag * a.imag
    cl *= 2. / (2 * np.arange(len(cl)) + 1)
    if lmaxout > lmaxout_:
        ret = np.zeros(lmaxout + 1, dtype=float)
        ret[:lmaxout_ + 1] = cl
        return ret
    return cl

def alm_copy(alm:np.ndarray, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax

        Parameters
        ----------
        alm :ndarray
            healpy alm array to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax


    """
    lmaxin = Alm.getlmax(alm.size, mmaxin)
    if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
    if (lmaxin == lmaxout) and (mmaxin == mmaxout):
        ret = np.copy(alm)
    else:
        ret = np.zeros(Alm.getsize(lmaxout, mmaxout), dtype=alm.dtype)
        lmax_min = min(lmaxout, lmaxin)
        for m in range(0, min(mmaxout, mmaxin) + 1):
            idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
            idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
            ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
    return ret

class Alm:
    """alm arrays useful statics. Directly from healpy but excluding keywords


    """
    @staticmethod
    def getsize(lmax:int, mmax:int):
        """Number of entries in alm array with lmax and mmax parameters

        Parameters
        ----------
        lmax : int
          The maximum multipole l, defines the alm layout
        mmax : int
          The maximum quantum number m, defines the alm layout

        Returns
        -------
        nalm : int
            The size of a alm array with these lmax, mmax parameters

        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)

    @staticmethod
    def getidx(lmax:int, l:int or np.ndarray, m:int or np.ndarray):
        """Returns index corresponding to (l,m) in an array describing alm up to lmax.

        In HEALPix C++ and healpy, :math:`a_{lm}` coefficients are stored ordered by
        :math:`m`. I.e. if :math:`\ell_{max}` is 16, the first 16 elements are
        :math:`m=0, \ell=0-16`, then the following 15 elements are :math:`m=1, \ell=1-16`,
        then :math:`m=2, \ell=2-16` and so on until the last element, the 153th, is
        :math:`m=16, \ell=16`.

        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        l : int
          The l for which to get the index
        m : int
          The m for which to get the index

        Returns
        -------
        idx : int
          The index corresponding to (l,m)
        """
        return m * (2 * lmax + 1 - m) // 2 + l

    @staticmethod
    def getlmax(s:int, mmax:int or None):
        """Returns the lmax corresponding to a given healpy array size.

        Parameters
        ----------
        s : int
          Size of the array
        mmax : int
          The maximum m, defines the alm layout

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