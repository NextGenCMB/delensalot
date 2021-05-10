from lenscarf import remapping
from lenscarf.utils_scarf import Geom, scarfjob
import numpy as np
from lenscarf.utils_remapping import d2ang, ang2d

def exactpixel_solver(defl:remapping.deflection, ir:int, ip:int):
    """Solves the deflection field inversion 'exactly' (without splines but brute-force SHT) for a single pixel

    """
    defl._init_d1()
    sc_job_pixel = scarfjob()
    sc_job_check = scarfjob()
    sc_job_pixel.set_nthreads(1)
    sc_job_pixel.set_triangular_alm_info(defl.lmax_dlm, defl.mmax_dlm)

    sc_job_check.set_nthreads(1)
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

def fortran_ecp_solver(defl:remapping.deflection, ir, ip=None):
    """Emulates fortran remapping.f95 solver on a ring

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
            thtn, phin=  d2ang(red, imd, thti, phii, version)
            re_res, im_res = ang2d(thtn, thts, phin - phis) # residual deflection field
            res = np.sqrt(re_res * re_res + im_res * im_res)
            maxres = np.max(res)
            redi = redi - re_res
            imdi = imdi - im_res
            print(maxres / np.pi * 180 * 60, 'pixel index ' + str(np.argmax(res)))
    print(itr, ITRMAX, maxres / np.pi * 180 * 60)
    return redi, imdi

