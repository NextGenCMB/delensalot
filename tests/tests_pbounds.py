import numpy as np
import dlensalot
from dlensalot import utils_scarf as sj
import pylab as pl
from dlensalot import utils_scarf




j = sj.scarfjob()
j.set_thingauss_geometry(3999, 2, zbounds=(0.6,0.9))
for bounds in [ [-0.1, np.pi * 0.3], [0, np.pi * 0.5], [0., 2*np.pi], [-2* np.pi, 0]]:
    npix = 0
    for ir in range(j.geom.get_nrings()):
        pb=  utils_scarf.pbounds(bounds[0], bounds[1] )
        npix += len(utils_scarf.Geom.pbounds2pix(j.geom, ir, pb)[0])

    assert(npix == utils_scarf.Geom.pbounds2npix(j.geom, pb)), (npix, utils_scarf.Geom.pbounds2npix(j.geom, pb))
