import numpy as np

from lenscarf.utils_scarf import scarfjob
from lenscarf.utils_hp import almxfl, Alm

def get_spin_raise(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret


def plm2kggo(job:scarfjob, plm:np.ndarray, olm:np.ndarray=None):
    """Returns convergence (kappa), shear maps (gamma1, gamma2) and rotation map (omega) from lensing potentials healpy arrays

        :math:`\kappa = -\frac 12 \Delta \phi`
        :math:`\gamma_1 + i \gamma_2 = spin-2 transform of -1/2 raise * raise (plm + i olm)
        :math:`\omega = -\frac 12 \Delta \Omega`

    #FIXME: not too sure about the sign of olm
    """
    lmax = Alm.getlmax(plm.size, job.mmax)
    assert lmax == job.lmax, (Alm.getlmax(plm.size, job.mmax), job.lmax)
    ftl_k =  0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)  # For k = -1/2 Delta
    ftl_g = -0.5 * get_spin_raise(0, lmax) * get_spin_raise(1, lmax)
    # coefficient for spin lowering is the same. g1 +i g2 is -1/2 spin-2 transform of raise ** 2 plm.
    k = job.alm2map(almxfl(plm, ftl_k, job.mmax, False))
    if olm is None:
        g1, g2 = job.alm2map_spin([almxfl(plm, ftl_g, job.mmax, False), np.zeros_like(plm)], 2)
        o = 0.
    else:
        g1, g2 = job.alm2map_spin([almxfl(plm, ftl_g, job.mmax, False), almxfl(olm, ftl_g, job.mmax, False)], 2)
        o = job.alm2map(almxfl(olm, ftl_k, job.mmax, False))
    return k, (g1, g2), o

def plm2M(job:scarfjob, plm:np.ndarray, olm:np.ndarray):
    """Returns determinant of magnification matrix corresponding to input deflection field potentials

        Args:
            job: scarfjob definining the geometry and other SHTs parameters (lmax, mmax, nthreads)
            plm: alm array for lensing gradient potential
            olm: alm array for lensing curl potential

        Returns:
            determinant of magnification matrix. Array of size input scarfjob pixelization gemoetry
            
    #FIXME: not too sure about the sign of olm
    """
    lmax = Alm.getlmax(plm.size, job.mmax)
    assert lmax == job.lmax, (Alm.getlmax(plm.size, job.mmax), job.lmax)
    ftl_k =  0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)  # For k = -1/2 Delta
    ftl_g = -0.5 * get_spin_raise(0, lmax) * get_spin_raise(1, lmax)
    M = (1. - job.alm2map(almxfl(plm, ftl_k, job.mmax, False))) ** 2 # (1 - k) ** 2
    M -= np.sum(job.alm2map_spin([almxfl(plm, ftl_g, job.mmax, False), almxfl(olm, ftl_g, job.mmax, False)], 2) ** 2, axis=0)  # - g1 ** 2 - g2 ** 2
    if np.any(olm):
        M += job.alm2map(almxfl(olm, ftl_k, job.mmax, False)) ** 2 # + w ** 2
    return M