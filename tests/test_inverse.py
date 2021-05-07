from lenscarf import cachers, remapping
from lenscarf.utils_scarf import Geom
import numpy as np
from lenscarf import utils_scarf as sj
import healpy as hp
from lenscarf.utils_remapping import d2ang, ang2d

from plancklens.utils import camb_clfile

PBOUNDS = [0., 2 * np.pi]
j = sj.scarfjob()
j.set_thingauss_geometry(3999, 2, zbounds=(0.9, 1))
#j.set_ecp_geometry(100, 100, tbounds=(0.0, np.pi / 10))  # This one is more tricky since has th == 0

lmax = 3000
clee = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmax + 1]
clpp = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax + 1]

glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)

dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2)))

d = remapping.deflection(j.geom, 1.7 * 1., PBOUNDS, dlm, 8, 8, cacher=cachers.cacher_mem())

def fortransolve(defl:remapping.deflection, ir):
    """Emulates fortran solver on a ring

    """
    defl._init_d1()
    (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = defl.d1.get_spline_info()
    pixs = Geom.pbounds2pix(defl.geom, ir, defl._pbds)
    tht = defl.geom.get_theta(ir)
    thts = defl.geom.get_theta(ir) * np.ones(pixs.size)
    phis = Geom.phis(defl.geom, ir)[defl._pbds.contains(Geom.phis(defl.geom, ir))]
    #print(fremap.remapping.solve_pixs(re_f, im_f, thts[0:1], phis[0:1], tht0, phi0, t2grid, p2grid)[0])
    TOLAMIN =1e-10
    ft = (thts - tht0) * t2grid
    fp = (phis - tht0) %(2. *np.pi) * p2grid
    redi, imdi = -np.array(defl.d1.eval_ongrid(ft, fp))
    maxres = 10.
    itr = 0
    ITRMAX=100
    version =  int(np.rint(1 - 2 * tht / np.pi))
    tol = max(TOLAMIN / 180 / 60 * np.pi, 1e-15)
    while  (maxres >= tol)  & (itr <= ITRMAX) :
            itr = itr + 1
            thti, phii =  d2ang(redi, imdi, thts, phis, version)
            ft = (thti - tht0) * t2grid
            fp =  (phii - phi0)%(2 * np.pi) * p2grid
            red, imd = defl.d1.eval_ongrid(ft, fp)
            #print(red[0], imd[0])
            """#=====
            e_t = 2 * np.sin(thti[3] * 0.5) ** 2
            d2 = red[3] * red[3] + imd[3] * imd[3]
            sind_d = 1. + np.poly1d([0., -1 / 6., 1. / 120., -1. / 5040.][::-1])(d2)
            e_d = 2 * np.sin(np.sqrt(red[3] * red[3] + imd[3] * imd[3]) * 0.5) ** 2
            e_tp = e_t + e_d - e_t * e_d + version * red[3] * sind_d * np.sin(thti[3])  # 1 -+ cost'
            """
            #assert (e_tp * (2 - e_tp) > 0.), (e_tp * (2 - e_tp), e_tp, e_t, e_d)
            #=====
            thtn, phin=  d2ang(red, imd, thti, phii, version)
            re_res, im_res = ang2d(thtn, thts, phin - phis) # residual deflection field
            maxres = np.max(np.sqrt(re_res * re_res + im_res * im_res))
            redi = redi - re_res
            imdi = imdi - im_res
            print(maxres / np.pi * 180 * 60)
    print(itr, ITRMAX, maxres / np.pi * 180 * 60)
    return redi, imdi

if __name__ == '__main__':

    d._bwd_angles()


