import numpy as np

from lenscarf.utils_scarf import scarfjob
from lenscarf.utils_hp import almxfl, Alm

def get_spin_raise(s:int, lmax:int):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s:int, lmax:int):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret


def dlm2kggo(job:scarfjob, dlm:np.ndarray, dclm:np.ndarray or None=None):
    """Returns convergence (kappa), shear maps (gamma1, gamma2) and rotation map (omega) from lensing potentials healpy arrays

        :math:`\kappa = -\frac 12 \Delta \phi`
        :math:`\gamma_1 + i \gamma_2 = spin-2 transform of -1/2 raise * raise (plm + i olm)
        :math:`\omega = -\frac 12 \Delta \Omega`

    #FIXME: not too sure about the sign of olm
    """
    lmax = Alm.getlmax(dlm.size, job.mmax)
    assert lmax == job.lmax, (Alm.getlmax(dlm.size, job.mmax), job.lmax)
    # We further have p2d = get_spin_raise(0, lmax) = sqrt(0, lmax + 1) * sqrt(1, lmax + 2)
    d2k = -0.5 *  get_spin_lower(1, lmax)  # For k = -1/2 Delta
    d2g = -0.5 *  get_spin_raise(1, lmax)
    # coefficient for spin lowering is the same. g1 +i g2 is -1/2 spin-2 transform of raise ** 2 plm.
    k = job.alm2map(almxfl(dlm, d2k, job.mmax, False))
    if dclm is None:
        g1, g2 = job.alm2map_spin([almxfl(dlm, d2g, job.mmax, False), np.zeros_like(dlm)], 2)
        o = 0.
    else:
        g1, g2 = job.alm2map_spin([almxfl(dlm, d2g, job.mmax, False), almxfl(dclm, d2g, job.mmax, False)], 2)
        o = job.alm2map(almxfl(dclm, d2k, job.mmax, False))
    return k, (g1, g2), o

def dlm2M(job:scarfjob, dlm:np.ndarray, dclm: np.ndarray or None):
    """Returns determinant of magnification matrix corresponding to input deflection field

        Args:
            job: scarfjob definining the geometry and other SHTs parameters (lmax, mmax, nthreads)
            dlm: alm array for lensing deflection gradient
            dclm: alm array for lensing deflection curl (treated as zero if None)

        Returns:
            determinant of magnification matrix. Array of size input scarfjob pixelization gemoetry

    #FIXME: not too sure about the signs for non-zero curl deflection
    """
    lmax = Alm.getlmax(dlm.size, job.mmax)
    assert lmax == job.lmax, (Alm.getlmax(dlm.size, job.mmax), job.lmax)
    if dclm is None:
        dclm = np.zeros_like(dlm)
    assert lmax == Alm.getlmax(dclm.size, job.mmax), (Alm.getlmax(dclm.size, job.mmax), Alm.getlmax(dlm.size, job.mmax))
    d2k = -0.5 *  get_spin_lower(1, lmax)  # For k = -1/2 Delta
    d2g = -0.5 *  get_spin_raise(1, lmax)
    M = (1. - job.alm2map(almxfl(dlm, d2k, job.mmax, False))) ** 2 # (1 - k) ** 2
    M -= np.sum(job.alm2map_spin([almxfl(dlm, d2g, job.mmax, False), almxfl(dclm, d2g, job.mmax, False)], 2) ** 2, axis=0)  # - g1 ** 2 - g2 ** 2
    if np.any(dclm):
        M += job.alm2map(almxfl(dclm, d2k, job.mmax, False)) ** 2 # + w ** 2
    return M