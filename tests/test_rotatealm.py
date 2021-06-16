import pylab as pl
from ducc0.misc import rotate_alm
import numpy as np
import healpy as hp
import os

# --- looks like 3 to 4 minutes to rotate lmax=4000 alm with 8 threads
#--- s06b ctr and range (40, 146.407), lat ctr -55
p = os.environ['ONED'] +'/cmbs4/inputs/ipvmap.fits'
lmax = 800

# I believe ths correct order to remap onto the north pole looks like (eg.g for cmbs4 06b) -40 / 180 * np.pi, -143 / 180 * np.pi, 0 / 180

glm = hp.map2alm(hp.read_map(p, verbose=False, dtype=float), lmax, iter=0)
hp.mollview(hp.alm2map(rotate_alm(glm, lmax, -40 / 180 * np.pi, - (90 + 55) / 180 * np.pi, 0, 8), 512))
pl.show() # On north pole
hp.mollview(hp.alm2map(rotate_alm(glm, lmax, -40 / 180 * np.pi, - (0 + 55) / 180 * np.pi, 0, 8), 512))
pl.show() # On Eq, phi = 0
