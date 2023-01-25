from dlensalot import cachers, remapping
import numpy as np
from dlensalot import utils_scarf as sj
import healpy as hp

from plancklens.utils import camb_clfile


PBOUNDS = (0., 2 * np.pi)
j = sj.scarfjob()
j.set_ecp_geometry(100, 100,  tbounds=(0.0, np.pi/10))

lmaxin = 3999
lmaxout = 2999
clee = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmaxin + 1]
clpp = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmaxin + 1]

glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)


dlm = hp.almxfl(plm, np.sqrt(np.arange(lmaxin + 1) * np.arange(1, lmaxin + 2)))


d = remapping.deflection(j.geom, 1.7, PBOUNDS, dlm, 8, 8, cacher=cachers.cacher_mem())

def get_mi():
    d._bwd_angles()
    return d._bwd_magn()


if __name__ == '__main__':
    print(d._fwd_magn())
    print(get_mi())