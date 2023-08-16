import os
import numpy as np
from plancklens.helpers import cachers

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

alm2rlm = lambda x : x.copy()
rlm2alm = lambda x : x.copy()


from plancklens import shts
from delensalot.utility.utils_hp import Alm, almxfl
from plancklens.utils import cli

def kappa_to_phi(k_lm):
    lmax = Alm.getlmax(k_lm.size, None)
    ells = np.arange(0, lmax+1, 1)
    factor = ells*(ells+1)/2
    return almxfl(k_lm, cli(factor), lmax, False)

def kappa_shifted(kappa, kappa0):
    return kappa+kappa0

def get_kappa(ylm, kappa0, lmax):
    y = shts.alm2map(ylm, lmax)
    return kappa_shifted(np.exp(y), -kappa0)

def get_plm(ylm, kappa0, lmax):
    kappa = get_kappa(ylm, kappa0, lmax)
    return kappa_to_phi(shts.map2alm(kappa, lmax))


def transform(rlm, kappa0):
    if kappa0 is not None:
        print('kappa0 is not None, transforming to phi')
        lmax = Alm.getlmax(rlm.size, None)
        rlm = get_plm(rlm, kappa0, lmax)
    return rlm

#TODO this looks like a 'query' class to me. May be refactored.
class rec:
    """Static methods to reach for iterated lensing maps etc


    """

    @staticmethod
    def maxiterdone(lib_dir):
        lib_dir = os.path.abspath(lib_dir)
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = rec.is_iter_done(lib_dir, itr + 1)
        return itr

    @staticmethod
    def load_plms(lib_dir, itrs, kappa0 = None):
        """Loads plms for the requested itrs"""
        lib_dir = os.path.abspath(lib_dir)
        cacher = cachers.cacher_npy(lib_dir)
        itmax = np.max(itrs)
        sk_fname = lambda k: os.path.join(lib_dir, 'hessian', 'rlm_sn_%s_%s' % (k, 'p'))
        rlm = alm2rlm(cacher.load(os.path.join(lib_dir, 'phi_plm_it000')))

        """if kappa0 is not None:
            print('kappa0 is not None, transforming to phi')
            lmax = Alm.getlmax(rlm.size, None)
            rlm = get_plm(rlm, kappa0, lmax)
        """

        ret = [] if 0 not in itrs else [rlm2alm(transform(rlm, kappa0))]
        for i in range(itmax):
            if cacher.is_cached(sk_fname(i)):
                rlm += cacher.load(sk_fname(i))

                if (i + 1) in itrs:
                    print("Doing for kappa0", kappa0)
                    ret.append(rlm2alm(transform(rlm, kappa0)))
            else:
                log.info("*** Could only build up to itr number %s"%i)
                return ret

        return ret

    @staticmethod
    def load_elm(lib_dir, itr):
        """Load delensing E-map at iteration 'itr'

        """
        lib_dir = os.path.abspath(lib_dir)
        cacher = cachers.cacher_npy(lib_dir)
        e_fname = os.path.join(lib_dir, 'wflms', 'wflm_%s_it%s' % ('p', itr))
        assert cacher.is_cached(e_fname), 'cant load ' + e_fname
        return cacher.load(e_fname)

    @staticmethod
    def is_iter_done(lib_dir, itr):
        """Returns True if the iteration 'itr' has been performed already and False if not

        """
        lib_dir = os.path.abspath(lib_dir)
        if not os.path.exists(lib_dir): return False
        cacher = cachers.cacher_npy(lib_dir)
        if itr <= 0:
            return cacher.is_cached(os.path.join(lib_dir, '%s_plm_it000' % ({'p': 'phi', 'o': 'om'}['p'])))
        sk_fname = lambda k: os.path.join(lib_dir, 'hessian', 'rlm_sn_%s_%s' % (k, 'p'))
        return cacher.is_cached(sk_fname(itr - 1))

    @staticmethod
    def load_grad(lib_dir, itr):
        #FIXME: load gradient at zero
        assert 0, 'fix gradient load at 0'
        lib_dir = os.path.abspath(lib_dir)
        cacher = cachers.cacher_npy(lib_dir)
        yk_fname = lambda k: os.path.join(lib_dir, 'hessian','rlm_yn_%s_%s' % (k, 'p'))
        rlm = alm2rlm(load_gradient(0, 'p'))
        for i in range(itr):
            rlm += cacher.load(yk_fname(i))
        return rlm2alm(rlm)