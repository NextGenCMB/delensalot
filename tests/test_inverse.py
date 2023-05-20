from delensalot import remapping
from delensalot.core import cachers
from delensalot.utils_scarf import Geom, scarfjob
import numpy as np
from delensalot import utils_scarf as sj
import healpy as hp
from delensalot.core.helper.utils_remapping import d2ang, ang2d

from plancklens.utils import camb_clfile

PBOUNDS = (0., 2 * np.pi)
j = sj.scarfjob()
#j.set_thingauss_geometry(3999, smax=2, zbounds=(0.9, 1))
j.set_ecp_geometry(100, 100, tbounds=(0.0, np.pi / 10))  # This one is more tricky since has th == 0

lmax = 3000
clee = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lensedCls.dat')['ee'][:lmax + 1]
clpp = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax + 1]

glm = hp.synalm(clee, new=True)
plm = hp.synalm(clpp, new=True)

dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2)))

d = remapping.deflection(j.geom, 1.7, PBOUNDS, dlm, 8, 8, cacher=cachers.cacher_mem())

def fortransolve(defl:remapping.deflection, ir, ip=None):
    """Emulates fortran solver on a ring

    """
    defl._init_d1()
    (tht0, t2grid), (phi0, p2grid), (re_f, im_f) = defl.d1.get_spline_info()
    phis = Geom.phis(defl.geom, ir)[defl._pbds.contains(Geom.phis(defl.geom, ir))]
    tht = defl.geom.get_theta(ir)

    if ip is not None:
        assert np.isscalar(ip)
        phis = np.array([phis[ip]])
        print('thetaphi', tht, phis)

    thts = defl.geom.get_theta(ir) * np.ones(phis.size)
    #print(fremap.remapping.solve_pixs(re_f, im_f, thts[0:1], phis[0:1], tht0, phi0, t2grid, p2grid)[0])
    TOLAMIN =1e-10
    ft = (thts - tht0) * t2grid
    fp = (phis - phi0) %(2. *np.pi) * p2grid
    redi, imdi = -np.array(defl.d1.eval_ongrid(ft, fp))
    print(redi[0], imdi[0])

    maxres = 10.
    itr = 0
    ITRMAX=30
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
            res = np.sqrt(re_res * re_res + im_res * im_res)
            maxres = np.max(res)
            redi = redi - re_res
            imdi = imdi - im_res
            print(maxres / np.pi * 180 * 60, 'pixel index ' + str(np.argmax(res)))
    print(itr, ITRMAX, maxres / np.pi * 180 * 60)
    return redi, imdi, thti, phii

def pixel_solver(defl:remapping.deflection, ir:int, ip:int):
    """Solves the inversion 'exactly' (wo splines but brute-force SHT) for a single pixel

    """
    defl._init_d1()
    sc_job_pixel = scarfjob()
    sc_job_check = scarfjob()
    sc_job_pixel.set_nthreads(1)
    sc_job_pixel.set_triangular_alm_info(defl.lmax_dlm, defl.mmax_dlm)

    sc_job_check.set_nthreads(8)
    sc_job_check.set_triangular_alm_info(defl.lmax_dlm, defl.mmax_dlm)
    dclm = [defl.dlm, defl.dlm * 0]

    tht = defl.geom.get_theta(ir)
    thts = np.array([tht])
    phi = Geom.phis(defl.geom, ir)[defl._pbds.contains(Geom.phis(defl.geom, ir))][ip]
    sc_job_pixel.set_pixel_geometry(tht, phi)
    sc_job_check.set_ecp_geometry(2, 2, tbounds=(tht, np.pi))

    print('thetaphi', tht, phi,  Geom.phis(sc_job_pixel.geom, 0))
    phis = np.array([phi])
    TOLAMIN =1e-10
    #ft = (thts - tht0) * t2grid
    #fp = (phis - tht0) %(2. *np.pi) * p2grid
    redi, imdi = -sc_job_pixel.alm2map_spin(dclm, 1)[:,0:1]
    redcheck, imdcheck = -sc_job_check.alm2map_spin(dclm, 1)
    print(redi[0], imdi[0], redcheck[0], imdcheck[0])

    maxres = 10.
    itr = 0
    ITRMAX=30
    version =  int(np.rint(1 - 2 * tht / np.pi))
    tol = max(TOLAMIN / 180 / 60 * np.pi, 1e-15)

    while  (maxres >= tol)  & (itr <= ITRMAX) :
            itr = itr + 1
            thti, phii =  d2ang(redi, imdi, thts, phis, version)
            sc_job_pixel.set_pixel_geometry(thti, phii)
            red, imd = sc_job_pixel.alm2map_spin(dclm, 1)[:,0:1]
            thtn, phin=  d2ang(red, imd, thti, phii, version)
            re_res, im_res = ang2d(thtn, thts, phin - phis) # residual deflection field
            maxres = np.max(np.sqrt(re_res * re_res + im_res * im_res))
            redi = redi - re_res
            imdi = imdi - im_res
            print(maxres / np.pi * 180 * 60, sc_job_pixel.geom.theta, Geom.phis(sc_job_pixel.geom, 0))
    print(itr, ITRMAX, maxres / np.pi * 180 * 60)
    return redi, imdi
if __name__ == '__main__':

    d._bwd_angles()


