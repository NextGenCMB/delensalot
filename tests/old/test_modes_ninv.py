import numpy as np
from lenspyx.remapping.utils_geom import Geom
from delensalot.core.opfilt import tmodes_ninv, ebmodes_ninv
from psutil import cpu_count

lmax = 6
nside = 256
thread = cpu_count(logical=False)

geom = Geom.get_healpix_geometry(nside)
tpl = tmodes_ninv.template_tfilt(lmax, geom, sht_threads=thread)

NiT = np.ones(geom.npix(), dtype=float)
mT = tpl.build_tnit(NiT)
print('mT shape', mT.shape)
print(np.diag(mT))

tpl = ebmodes_ninv.template_ebfilt(lmax, geom, sht_threads=thread)

NiT = np.ones(geom.npix(), dtype=float)
mP = tpl.build_tnit(NiT)

print('mP shape', mP.shape)
print(np.diag(mP))

