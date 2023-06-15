from lenspyx.shts import shts as lpyx_shts, fsht
from itercurv import utils_shts as usht
from delensalot.utils import timer
from delensalot.utility.utils_hp import Alm
from lenspyx.bicubic import bicubic as lpyx_bicubic

def lensgclm_band_lpyx(thts, nphi, spin, dx, dy, glm, clm=None):
    tim = timer(True)
    vtm = fsht.vlm2vtm(Alm.getlmax(glm.size, -1), spin, thts, usht.alm2vlm(glm, clm=clm))
    tim.add('glm2vtm')
    ma = lpyx_shts.vtm2map(spin, vtm, nphi, bicubic_prefilt=True)
    tim.add('vtm2fmap')
    re = lpyx_bicubic.deflect(ma.real, dx, dy)
    im = lpyx_bicubic.deflect(ma.imag, dx, dy)
    tim.add('interp')
    print(tim)
    return re, im