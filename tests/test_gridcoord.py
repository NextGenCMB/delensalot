from plancklens import utils
from delensalot.utils_scarf import Geom
import numpy as np
import delensalot
from delensalot import utils_scarf as sj
import pylab as pl
from delensalot import utils_scarf
from delensalot import cachers, remapping
from plancklens.utils import camb_clfile
lmax = 3000
clee = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmax + 1]
clpp = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax + 1]

import healpy as hp
glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)


j = sj.scarfjob()
j.set_thingauss_geometry(3999, 2, zbounds=(0.,0.1))


dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2)))
d = remapping.deflection(j.geom, 1.7, [0., np.pi], dlm, 8, 8, cacher=cachers.cacher_mem())




d1 = d._build_interpolator(d.dlm, 1)
(tht0, t2grid), (phi0, p2grid), (re_f, im_f) = d1.get_spline_info()
PTRUNC = abs(d1.patch.pbounds[1]) < 2 * np.pi


for i, ir in utils.enumerate_progress(range(d.geom.get_nrings())):
   pixs, phis = Geom.pbounds2pix(d.geom, ir, d._pbds)
   thts = np.ones(len(pixs)) * d.geom.get_theta(ir)
   thts_grid = (thts - tht0) * t2grid
   phis_grid = (phis - phi0) %(2 * np.pi) * p2grid
   tmax = np.max(thts_grid)
   tmin = np.min(thts_grid)
   pmax = np.max(phis_grid)
   pmin = np.min(phis_grid)
   if tmax > re_f.shape[0] + 3:
        assert 0
   if tmin <  3:
        assert 0
   if PTRUNC and pmax > re_f.shape[1] - 3:
        print(phis[np.where(phis_grid==pmax)])
        assert 0, (pmax, re_f.shape, ir, d._pbds, d1.patch.pbounds, phi0,(phis[[0, -1]] - phi0) %(2 * np.pi) * p2grid)
   if PTRUNC and pmin <  3:
        assert 0, (pmin, re_f.shape, ir, d1.patch.pbounds, phi0)