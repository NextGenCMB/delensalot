"""Module to handle steps in abstract 'gradient' format


"""
from __future__ import annotations
from lenspyx.utils_hp import Alm, almxfl, alm2cl
import numpy as np


class gradient:
    """Class to abstract away the handling of the joint reconstruction of several fields

    """
    def __init__(self, componentlist:list[np.ndarray], mmax_list:list[int], labels:tuple[str]=('p', 'x')):
        """
            Args:
                componentlist: list of the gradient components, all healpy-like alm arrays
                mmax_list: mmax's of the arrays
                labels: strings to distinguish components if needed


        """
        self.comps = componentlist
        self.mmaxs = mmax_list
        self.lmaxs = [Alm.getlmax(alm.size, mmax) for alm, mmax in zip(componentlist, mmax_list)]
        self.labels = labels[:len(self.comps)]

    def almxfl(self, cl_list:list[np.ndarray], inplace):
        assert len(cl_list) == len(self.comps), (len(cl_list), len(self.comps))
        if inplace:
            for alm, cl, mmax in zip(self.comps, cl_list, self.mmaxs):
                almxfl(alm, cl, mmax, True)
        else:
            comps = [almxfl(alm, cl, mmax, False) for alm, cl, mmax in zip(self.comps, cl_list, self.mmaxs)]
            return gradient(comps, self.mmaxs, labels=self.labels)

    def alm2cl(self):
        return [alm2cl(alm, alm, None, mmax, None) for alm, mmax in zip(self.comps, self.mmaxs)]

    def get_comp(self, label:str):
        if label in self.labels:
            return self.comps[self.labels.index(label)]
        elif label[0] == 'x' and 'p' + label[1:] in self.labels:
            return np.zeros_like(self.get_comp('p' + label[1:]))
        else:
            assert 0, 'no ' + label + ' component in this gradient'

    def getarray(self):
        return np.concatenate(self.comps)

    @staticmethod
    def fromarray(arr, lmaxs:list[int], mmaxs:list[int], labels=None):
        """Builds instance from its array 'getarray' """
        N = 0
        comps = []
        for lmax, mmax in zip(lmaxs, mmaxs):
            size = Alm.getsize(lmax, mmax)
            comps.append(arr[N:N+size])
            N += size
        return gradient(comps, mmaxs, labels=labels)

    def copy(self):
        return gradient([np.copy(alm) for alm in self.comps], self.mmaxs, labels=self.labels)

    def __mul__(self, other):
        if np.isscalar(other):
            return gradient([alm * other for alm in self.comps], self.mmaxs, labels=self.labels)
        else:
            assert 0, 'not implemented'

    def __truediv__(self, other):
        if np.isscalar(other):
            return gradient([alm / other for alm in self.comps], self.mmaxs, labels=self.labels)
        else:
            assert 0, 'not implemented'

    def __sub__(self, other):
        if isinstance(other, gradient):
            assert other.lmaxs == self.lmaxs # not necessary but hard to think when this should not be a bug
            assert other.mmaxs == self.mmaxs
            return gradient([alm - blm for alm, blm in zip(self.comps, other.comps)], self.mmaxs, labels=self.labels)
        else:
            assert 0, 'not implemented'

    def __add__(self, other):
        if isinstance(other, gradient):
            assert other.lmaxs == self.lmaxs # not necessary but hard to think when this should not be a bug
            assert other.mmaxs == self.mmaxs
            return gradient([alm + blm for alm, blm in zip(self.comps, other.comps)], self.mmaxs, labels=self.labels)
        else:
            assert 0, 'not implemented'

    def __iadd__(self, other):
        if isinstance(other, gradient):
            assert other.lmaxs == self.lmaxs # not necessary but hard to think when this should not be a bug
            assert other.mmaxs == self.mmaxs
            for alm, blm in zip(self.comps, other.comps):
                alm += blm
            return self
        else:
            assert 0, 'not implemented'

    def __isub__(self, other):
        if isinstance(other, gradient):
            assert other.lmaxs == self.lmaxs # not necessary but hard to think when this should not be a bug
            assert other.mmaxs == self.mmaxs
            for alm, blm in zip(self.comps, other.comps):
                alm -= blm
            return self
        else:
            assert 0, 'not implemented'

    def __imul__(self, other):
        if np.isscalar(other):
            for alm in self.comps:
                alm *= other
            return self
        else:
            assert 0, 'not implemented'

    def __itruediv__(self, other):
        if np.isscalar(other):
            for alm in self.comps:
                alm /= other
            return self
        else:
            assert 0, 'not implemented'

    def __neg__(self):
        return gradient([-alm for alm in self.comps], self.mmaxs, labels=self.labels)

    def __pos__(self):
        return self.copy()

def gradient_dotop(g1:gradient, g2:gradient):
    assert g1.mmaxs == g2.mmaxs
    assert g1.lmaxs == g2.lmaxs
    ret = 0.
    for alm1, alm2, mmax in zip(g1.comps, g2.comps, g1.mmaxs):
        cl = alm2cl(alm1, alm2, None, mmax, None)
        ret += np.sum(cl * (2 * np.arange(len(cl)) + 1 ))
    return ret

class nrstep(object):
    def __init__(self, val=1.):
        self.val = val


    def build_incr(self, incrlm:gradient, itr:int):
        return incrlm * self.val

class harmonicbump(nrstep):
    def __init__(self, xa=400, xb=1500, a=0.5, b=0.1, scale=50):
        """Harmonic bumpy step that were useful for s06b and s08b

        """
        self.scale = scale
        self.bump_params = (xa, xb, a, b)

    def steplen(self, lmax_qlm):
        xa, xb, a, b = self.bump_params
        return self.bp(np.arange(lmax_qlm + 1),xa, a, xb, b, scale=self.scale)


    def build_incr(self, incrlm:gradient, itr:int):
        incrlm.almxfl([self.steplen(lmax) for lmax in  incrlm.lmaxs], True)
        return incrlm

    @staticmethod
    def bp(x, xa, a, xb, b, scale=50):
            """Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale

            """
            x0 = (xa + xb) * 0.5
            r = lambda x_: np.arctan(np.sign(b - a) * (x_ - x0) / scale) + np.sign(b - a) * np.pi * 0.5
            return a + r(x) * (b - a) / r(xb)

