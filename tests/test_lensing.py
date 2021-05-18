from lenscarf import cachers, remapping
import numpy as np
from lenscarf import utils_scarf as sj
import healpy as hp
from lenscarf import utils_config
from plancklens.utils import camb_clfile


# PBOUNDS = (np.pi, 2* np.pi)
#j = sj.scarfjob()
#j.set_thingauss_geometry(3999, 2, zbounds=(0.9, 1.))
j, PBOUNDS = utils_config.cmbs4_08b_healpix()
print(PBOUNDS, np.min(j.geom.cth), np.max(j.geom.cth))

lmaxin = 3999
lmaxout = 2999
clee = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmaxin + 1]
clpp = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmaxin + 1]

glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)


dlm = hp.almxfl(plm, np.sqrt(np.arange(lmaxin + 1) * np.arange(1, lmaxin + 2)))

d = remapping.deflection(j.geom, 1.7, PBOUNDS, dlm, 8, 8, cacher=cachers.cacher_mem())

d.lensgclm(glm, 0, lmaxout)
d.tim.reset()
d.lensgclm(glm, 0, lmaxout)
d.tim.reset()
d.lensgclm(glm, 2, lmaxout)
d.tim.reset()
d.lensgclm(glm, 2, lmaxout)
d.tim.reset()
