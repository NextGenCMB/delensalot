from lenscarf import cachers, remapping
import numpy as np
from lenscarf import utils_scarf as sj
import healpy as hp

from plancklens.utils import camb_clfile


PBOUNDS = [0., 0.1 * np.pi]
j = sj.scarfjob()
j.set_thingauss_geometry(3999, 2, zbounds=(0.,0.1))

lmax = 3000
clee = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmax + 1]
clpp = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax + 1]

glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)


dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2)))

d = remapping.deflection(j.geom, 1.7, PBOUNDS, dlm, 8, 8, cacher=cachers.cacher_mem())
d._bwd_angles()
