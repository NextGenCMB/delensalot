import numpy as np
from lenspyx.remapping.utils_geom import Geom
from delensalot.core.opfilt import tmodes_ninv
from psutil import cpu_count

lmax = 20
nside = 2048
thread = cpu_count(logical=False)

geom = Geom.get_healpix_geometry(nside)
tpl = tmodes_ninv.template_tfilt(lmax, geom, sht_threads=thread)

NiT = np.ones(geom.npix(), dtype=float)
tpl.build_tnit(NiT)