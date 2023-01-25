import os.path

import healpy as hp
import numpy as np
from time import time
from ducc0 import sht
import dlensalot
from dlensalot.utils_scarf import scarfjob

def test_backforth(geom, spin):
    """Tests quadrature accuracy of different pixelization by just going back and forth

    """

    if geom == 'gauss':
        print("*** testing Gauss-Legendre grid")
        nlon = 8192
        nlat = lmax + 1
        job = sht.sharpjob_d()
        job.set_gauss_geometry(nlat, nlon)

    elif geom == 'healpix':
        print("*** testing Healpix grid")
        job = sht.sharpjob_d()
        job.set_healpix_geometry(2048)

    elif geom == 'scarfhealpix':
        print("*** scarf Healpix grid")
        job = scarfjob()
        job.set_healpix_geometry(2048)

    elif geom == 'scarfgauss':
        print("*** scarf gauss grid")
        nlon = 8192
        nlat = lmax + 1
        job = scarfjob()
        job.set_gauss_geometry(nlat, nlon)

    elif geom == 'scarfthingauss':
        print("*** scarf thin gauss grid")
        job = scarfjob()
        job.set_thingauss_geometry(lmax, 2)

    else:
        assert 0, geom

    job.set_nthreads(8)
    job.set_triangular_alm_info(lmax, mmax)
    t0 = time()
    if spin > 0:
        m = job.alm2map_spin([glm, glm * 0], 2)
    else:
        m = job.alm2map(glm)
    print("time for map synthesis: %.2f" % (time() - t0))
    print('root npix', int(np.sqrt(job.n_pix())))

    t0 = time()
    if spin > 0:
        glm2 = job.map2alm_spin(m, 2)
        print("time for alm synthesis: %.2f" % (time() - t0))
        print('lmax', hp.Alm.getlmax(glm2[0].size))
    else:
        glm2 = job.map2alm(m)
        print("time for alm synthesis: %.2f" % (time() - t0))
        print('lmax', hp.Alm.getlmax(glm2.size))
    if spin > 0:
        m2 = job.alm2map_spin(glm2, 2)

    else:
        m2 = job.alm2map(glm2)

    print('real space max dev', np.max(np.abs(m2 - m)), np.std(m2 - m))
    print('harmonic space max dev', np.max(np.abs(glm2 - glm)), np.std(glm2 - glm))
    return m


if __name__ == '__main__':
    from plancklens import utils
    clpath = os.path.dirname(dlensalot.__file__)
    cl_len = utils.camb_clfile(clpath + '/data/cls/FFP10_wdipole_lensedCls.dat')
    lmax = 4000
    glm = hp.synalm(cl_len['ee'][:lmax + 1], new=True)
    mmax = lmax

    m1 = test_backforth('gauss', 0)
    #test_backforth('healpix', 0)
    #test_backforth('scarfhealpix', 0)
    m2 = test_backforth('scarfgauss', 0)
    print(np.max(np.abs(m1 - m2)))
    m3 = test_backforth('scarfthingauss', 0)
    #mgs = test_backforth('scarfthingauss', 0)
    #test_backforth('gauss', 2)
    #test_backforth('healpix', 2)
    #test_backforth('scarfhealpix', 2)
    #mgs = test_backforth('scarfgauss', 2)
    #mgs = test_backforth('scarfthingauss', 2)