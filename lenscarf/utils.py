import sys
from time import time
import numpy as np

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
            s += "\r  %40s:  [" % k + ('%02dh:%02dm:%02ds:%03dms' % (dh, dm, ds, dms)) + "] " + "\n"
        dt = time() - self.ti
        dh = np.floor(dt / 3600.)
        dm = np.floor(np.mod(dt, 3600.) / 60.)
        ds = np.floor(np.mod(dt, 60))
        dms = 1000 *np.mod(dt, 1.)
        s += "\r  %40s:  [" % 'Total' + ('%02dh:%02dm:%02ds:%03dms' % (dh, dm, ds, dms)) + "] "
        return s

    def add(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time()
        self.keys[label] += t0  - self.t0
        self.t0 = t0

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

