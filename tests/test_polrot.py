import numpy as np
from dlensalot import utils_hp, remapping, cachers
from dlensalot.utils_scarf import pbdGeometry, pbounds, scarfjob, Geom
from plancklens.utils import camb_clfile

def phase_approx(d:remapping.deflection):
    """Challinor & Chon approx to the phase rotation owing to axes rotation

        Its ok but with an inclusion of a factor 1 / sin(tht)

    """
    func1 = lambda red, imd, tht, cos, sin : imd * (cos / sin)
    func2 = lambda red, imd, tht, cos, sin : imd * (cos - 0.5 * red * (1. + cos ** 2 ) /sin) / sin
    return d.fill_map([func1, func2])


if __name__ == '__main__':
    import pylab as pl

    lmax_dlm, mmax_dlm, targetres_amin, sht_threads, fftw_threads = (3000, 3000, 1.7, 8, 8)
    cacher = cachers.cacher_mem()
    lenjob = scarfjob()
    lenjob.set_healpix_geometry(2048)
    # deflection instance:
    cldd = camb_clfile('../lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat')['pp'][:lmax_dlm + 1]
    cldd *= np.sqrt(np.arange(lmax_dlm + 1) *  np.arange(1, lmax_dlm + 2))
    #dlm = hp.synalm(cldd, lmax=lmax_dlm, mmax=mmax_dlm) # get segfault with nontrivial mmax and new=True ?!
    dlm = utils_hp.synalm(cldd, lmax_dlm, mmax_dlm)
    d_geom = pbdGeometry(lenjob.geom, pbounds(np.pi, 2 * np.pi))
    d = remapping.deflection(d_geom, targetres_amin, dlm, mmax_dlm, sht_threads, fftw_threads, cacher=cacher)

    g1 = d._fwd_polrot()
    g2, g3 = phase_approx(d)

    print(np.max(np.abs(g2 / g1 - 1.)), np.std(np.abs(g2 / g1 - 1.)))
    print(np.max(np.abs(g3 / g1 - 1.)), np.std(np.abs(g3 / g1 - 1.)))