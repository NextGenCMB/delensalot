import numpy as np

from lenscarf.utils_scarf import scarfjob
from lenscarf.utils_hp import getlmax

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
    assert lmax == job.lmax, (getlmax(plm.size, mmax=job.mmax), job.lmax)
    ftl_k =  0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)  # For k = -1/2 Delta
    ftl_g = -0.5 * get_spin_raise(0, lmax) * get_spin_raise(1, lmax)
    # coefficient for spin lowering is the same. g1 +i g2 is -1/2 spin-2 transform of raise ** 2 plm.
    k = job.alm2map(hp.almxfl(plm, ftl_k, mmax=job.mmax))
    if olm is None:
        g1, g2 = job.alm2map_spin([hp.almxfl(plm, ftl_g, mmax=job.mmax), np.zeros_like(plm)], 2)
        o = 0.
    else:
        g1, g2 = job.alm2map_spin([hp.almxfl(plm, ftl_g, mmax=job.mmax), hp.almxfl(olm, ftl_g, mmax=job.mmax)], 2)
        o = job.alm2map(hp.almxfl(olm, ftl_k), mmax=job.mmax)
    return k, (g1, g2), o