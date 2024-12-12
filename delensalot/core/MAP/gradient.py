import numpy as np

from delensalot.core import cachers

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utils import cli

from . import filter

class base:
    def __init__(self, field, gradient_desc, filter_desc, **kwargs):
        self.field = field
        self.filter = filter(filter_desc)
        self.det_fn = gradient_desc['det_fn']
        self.pri_fn = gradient_desc['pri_fn']
        self.quad_fn = gradient_desc['quad_fn']
        self.increment_fns = gradient_desc['increment_fns']
        self.cacher = cachers.cacher_npy(gradient_desc['id'])


    def load_gradient(self, it):
        """Loads the total gradient at iteration iter.
        All necessary alm's must have been calculated previously
        Compared to formalism of the papers, this returns -g_LM^{tot}
        """
        if it == 0:
            for component in self.field.components:
                g  = self.load_grad_prior(it)
                g += self.load_grad_det(it)
                g += self.load_grad_quad(it)
            return g
        return self._build(it)


    def load_det(self, it):
        return self.cacher.load(self.det_fn.format(it=it))


    def load_prior(self, it):
        """Compared to formalism of the papers, this returns -g_LM^{PR}"""
        ret = self.field.get_klm(it)
        almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        return ret


    def load_quad(self, itr, key):
        fn = '%slm_grad%slik_it%03d' % (self.h, key.lower(), itr)
        return self.cacher.load(fn)


    def _build(self, it):
        rlm = self.load_gradient(0)
        for i in range(it):
            rlm += self.hess_cacher.load(self.increment_fns(i))
        return rlm
    

    def calc_gradient_quad(self, field):
        ivf = self.filter.get_ivf(field)
        inner = self.gradient.get_inner(field)
        XWF = self.filter.get_XWF(field)
        qlms = 0
        for n in [0,1,2]: # need to sum over spins here
            qlms += ivf * inner * XWF


    def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE
            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.
        """
        assert elm_wf.ndim == 1
        assert Alm.getlmax(elm_wf.size, self.mmax_sol) == self.lmax_sol
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = np.atleast_2d(almxfl(elm_wf, fl, self.mmax_sol, False))
        ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)
    

    def calc_gradient_meanfield(self):
        pass


    def calc_gradient_prior(self):
        pass


    def update_gradient(self):
        pass


