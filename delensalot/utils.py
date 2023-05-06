import sys
from time import time
import numpy as np
import hashlib
import json

class timer:
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time()
        self.ti = self.t0
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix
        self.keys = {}

    def __iadd__(self, othertimer):
        for k in othertimer.keys:
            if not k in self.keys:
                self.keys[k] = othertimer.keys[k]
            else:
                self.keys[k] += othertimer.keys[k]
        return self

    def reset(self):
        self.t0 = time()
        self.ti = self.t0
        self.keys = {}

    def reset_t0(self):
        self.t0 = time()

    def __str__(self):
        if len(self.keys) == 0:
            return r""
        s = ""
        for k in self.keys:
            dt = self.keys[k]
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dms = 1000 *  np.mod(dt,1.)
            #s +=  "%24s: %.1f"%(k, self.keys[k]) + '\n'
            s += "%40s:  [" % k + ('%02dh:%02dm:%02ds:%03dms' % (dh, dm, ds, dms)) + "] " + "\n"
        dt = time() - self.ti
        dh = np.floor(dt / 3600.)
        dm = np.floor(np.mod(dt, 3600.) / 60.)
        ds = np.floor(np.mod(dt, 60))
        dms = 1000 *np.mod(dt, 1.)
        s += "%45s :  [" %(self.prefix + ' Total') + ('%02dh:%02dm:%02ds:%03dms' % (dh, dm, ds, dms)) + "] "
        return s

    def add(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time()
        self.keys[label] += t0  - self.t0
        self.t0 = t0

    def add_elapsed(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time()
        self.keys[label] += t0  - self.ti

    def dumpjson(self, fn):
        json.dump(self.keys, open(fn, 'w'))

    def checkpoint(self, msg):
        dt = time() - self.t0
        self.t0 = time()

        if self.verbose:
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dms = 100 *np.mod(dt, 1.)

            dhi = np.floor((self.t0 - self.ti) / 3600.)
            dmi = np.floor(np.mod((self.t0 - self.ti), 3600.) / 60.)
            dsi = np.floor(np.mod((self.t0 - self.ti), 60))
            dmsi = 100 *np.mod((self.t0 - self.ti), 1.)

            sys.stdout.write("\r  %s   [" % self.prefix + ('%02d:%02d:%02d:%02d' % (dh, dm, ds,dms )) + "] "
                             + " (total [" + (
                                 '%02d:%02d:%02d:%02d' % (dhi, dmi, dsi, dmsi)) + "]) " + msg + ' %s \n' % self.suffix)

def enumerate_progress(lst:list or np.ndarray, label=''):
    """Simple progress bar.

    """
    t0 = time()
    ni = len(lst)
    for i, v in enumerate(lst):
        yield i, v
        ppct = int(100. * (i - 1) / ni)
        cpct = int(100. * (i + 0) / ni)
        if cpct > ppct:
            dt = time() - t0
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                             label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def clhash(cl, dtype=np.float32):
    """Hash for generic numpy array.

    By default we avoid here double precision checks since this might be machine dependent.

    """
    return hashlib.sha1(np.copy(cl.astype(dtype), order='C')).hexdigest()

def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def read_map(m):
    """Reads a map whether given as (list of) string (with ',f' denoting field f), array or callable

    """
    if callable(m):
        return m()
    if isinstance(m, list):
        ma = read_map(m[0])
        for m2 in m[1:]:
            ma = ma * read_map(m2) # avoiding *= to allow float and full map inputs
        return ma
    if not isinstance(m, str):
        return m
    if '.npy' in m:
        return np.load(m)
    elif '.fits' in m:
        #FIXME
        import healpy as hp
        if ',' not in m:
            return hp.read_map(m)
        m, field = m.split(',')
        return hp.read_map(m, field=int(field))
    else:
        assert 0, 'cant tell what to do with ' + m


def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls