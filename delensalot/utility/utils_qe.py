import os
import numpy as np
from scarf import Geometry
from delensalot.utility.utils_hp import Alm, almxfl

class qeleg_multi:
    """Quadratic estimator leg instance

    """
    def __init__(self, spins_in, spin_out, cls):
        assert isinstance(spins_in, list) and isinstance(cls, list) and len(spins_in) == len(cls)

        self.spins_in = spins_in
        self.cls = cls
        self.spin_ou = spin_out

    def __iadd__(self, qeleg):
        """Adds one spin_in/cl tuple.

        """
        assert qeleg.spin_ou == self.spin_ou, (qeleg.spin_ou, self.spin_ou)
        self.spins_in.append(qeleg.spin_in)
        self.cls.append(np.copy(qeleg.cl))
        return self

    def __call__(self, get_alm, geom:Geometry, sht_tr:int, mmax:int or None=None):
        """Returns the spin-weighted real-space map of the estimator.

            Args:
                get_alm: callable with 'e' ,'b' or 't' depending on instance spins
                geom:scarf geometry used to build the position-space map
                sht_tr: number of openmp threads for transform
                mmax: set this if inputs alms mmaxes are non-standard

            We first build X_{lm} in the wanted _{si}X_{lm} _{so}Y_{lm} and then convert this alm2map_spin conventions.

        """
        lmax = self.get_lmax()
        if mmax is None: mmax = lmax
        glm = np.zeros(Alm.getsize(lmax, mmax), dtype=complex)
        clm = np.zeros(Alm.getsize(lmax, mmax), dtype=complex) # X_{lm} is here glm + i clm
        for i, (si, cl) in enumerate(zip(self.spins_in, self.cls)):
            assert si in [0, -2, 2], str(si) + ' input spin not implemented'
            gclm = [get_alm('e'), get_alm('b')] if abs(si) == 2 else [-get_alm('t'), 0.]
            assert len(gclm) == 2
            sgn_g = -(-1) ** si if si < 0 else -1
            sgn_c = (-1) ** si if si < 0 else -1
            glm += almxfl(gclm[0], sgn_g * cl, mmax, False)
            if np.any(gclm[1]):
                clm += almxfl(gclm[1], sgn_c * cl, mmax, False)
        glm *= -1
        if self.spin_ou > 0: clm *= -1
        Red, Imd = geom.alm2map_spin([glm, clm],  abs(self.spin_ou), lmax, mmax, sht_tr, [-1., 1.])
        if self.spin_ou < 0 and self.spin_ou % 2 == 1: Red *= -1
        if self.spin_ou < 0 and self.spin_ou % 2 == 0: Imd *= -1
        return Red + 1j * Imd


    def get_lmax(self):
        return np.max([len(cl) for cl in self.cls]) - 1

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