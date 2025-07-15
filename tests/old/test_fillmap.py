import numpy as np
import time
from delensalot import remapping
from delensalot.core import cachers
from delensalot.utility import utils_hp
from delensalot.core.helper.utils_scarf import pbdGeometry, pbounds, scarfjob, Geom
from plancklens.utils import camb_clfile
from delensalot.core.helper.utils_remapping import d2ang, ang2d


def func_fwdangle(red, imd, tht, cost, sint, phis):
    thtp_, phip_ = d2ang(red, imd, tht * np.ones(red.size), phis, int(np.round(cost)))

def fwd_polrot(red, imd, tht, cost, sint, phis):
    d = np.sqrt(red * red + imd * imd)
    return np.arctan2(imd, red)-  np.arctan2(imd, d * np.sin(d) * (cost / sint) + red * np.cos(d))

def bwd_detmagn(red, imd, tht, cost, sint, phis):
    """Return redi imdi, (2, npix)-shaped

     """
    # hmm, need here the inverse angles as well
    pass
    #return ang2d(thti[sli], self.geom.theta[it] * np.ones(pixs.size), phii[sli] - phis)


if __name__ == '__main__':
    import pylab as pl

    lmax_dlm, mmax_dlm, targetres_amin, sht_threads, fftw_threads = (3000, 3000, 1.7, 8, 8)
    cacher = cachers.cacher_mem()
    lenjob = scarfjob()
    lenjob.set_healpix_geometry(2048)
    # deflection instance:
    cldd = camb_clfile('../delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax_dlm + 1]
    cldd *= np.sqrt(np.arange(lmax_dlm + 1) *  np.arange(1, lmax_dlm + 2))
    dlm = utils_hp.synalm(cldd, lmax_dlm, mmax_dlm)
    d_geom = pbdGeometry(lenjob.geom, pbounds(np.pi, 2 * np.pi))
    d = remapping.deflection(d_geom, targetres_amin, dlm, mmax_dlm, sht_threads, fftw_threads, cacher=cacher)


    t0 = time.time()
    d._fwd_polrot()
    print('%.1f fwdpolrot'%(time.time() - t0))
    t0 = time.time()
    d._bwd_angles()
    print('%.1f bwd_angles'%(time.time() - t0))
    t0 = time.time()
    d._bwd_polrot()
    print('%.1f bwd_polrot'%(time.time() - t0))
    t0 = time.time()
    d._bwd_magn()
    print('%.1f bwd_magn'%(time.time() - t0))
    t0 = time.time()

