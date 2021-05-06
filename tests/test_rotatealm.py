from ducc0.misc import rotate_alm
import numpy as np
import healpy as hp
import os

# --- looks like 3 to 4 minutes to rotate lmax=4000 alm with 8 threads
#--- s06b ctr and range (40, 146.407),
p = os.environ['ONED'] +'/cmbs4/inputs/ipvmap.fits'
lmax = 200

# I believe ths correct order to remap onto the north pole looks like (eg.g for cmbs4 06b) -40 / 180 * np.pi, -143 / 180 * np.pi, 0 / 180

glm = hp.map2alm(hp.read_map(p), lmax, iter=0)
hp.mollview(hp.alm2map(rotate_alm(glm, lmax, 0, 45 / 180 * np.pi, 50 / 180 * np.pi, 8), 1024))