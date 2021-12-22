import numpy as np
from lenscarf import utils_hp, remapping, cachers, utils_scarf, utils_dlm
from lenscarf.utils_scarf import pbdGeometry, pbounds, scarfjob, Geom
from plancklens.utils import camb_clfile
from lenscarf.utils_remapping import d2ang

def get_jacobian(d:remapping.deflection):
    """Challinor & Chon approx to the phase rotation owing to axes rotation

        Its ok but with an inclusion of a factor 1 / sin(tht)

    """
    npix = Geom.pbounds2npix(d.geom, d._pbds)
    dclm = np.zeros_like(d.dlm) if d.dclm is None else d.dclm
    red, imd = d.geom.alm2map_spin([d.dlm, dclm], 1, d.lmax_dlm, d.mmax_dlm, d.sht_tr, [-1., 1.])
    dt_sintdp_cotsintdp = np.zeros((3, npix), dtype=float)
    startpix = 0
    for ir in np.argsort(d.geom.ofs):  # We must follow the ordering of scarf position-space map
        pixs = Geom.pbounds2pix(d.geom, ir, d._pbds)
        if pixs.size > 0:
            phis = Geom.phis(d.geom, ir)[pixs - d.geom.ofs[ir]]
            assert phis.size == pixs.size, (phis.size, pixs.size)
            cot = np.cos(d.geom.get_theta(ir)) / np.sin(d.geom.get_theta(ir))
            thts = d.geom.get_theta(ir) * np.ones(pixs.size)
            thtp_, phip_ = d2ang(red[pixs], imd[pixs], thts, phis, int(np.round(np.cos(d.geom.theta[ir]))), sint_dphi=True)
            sli = slice(startpix, startpix + len(pixs))
            dt_sintdp_cotsintdp[0, sli] = thtp_
            dt_sintdp_cotsintdp[1, sli] = phip_
            dt_sintdp_cotsintdp[2, sli] = phip_ * cot

            startpix += len(pixs)
    assert startpix == npix, (startpix, npix)
    print("spin-1 transforming ")
    sjob = utils_scarf.scarfjob()
    sjob.set_geometry(d.geom)
    sjob.set_triangular_alm_info(d.lmax_dlm, d.mmax_dlm)
    sjob.set_nthreads(sht_threads)
    glm = - sjob.map2alm(dt_sintdp_cotsintdp[0])
    clm =   sjob.map2alm(dt_sintdp_cotsintdp[1])

    fl = np.sqrt(np.arange(d.lmax_dlm + 1) * np.arange(1, d.lmax_dlm + 2))
    utils_hp.almxfl(glm, fl, d.mmax_dlm, True)
    utils_hp.almxfl(clm, fl, d.mmax_dlm, True)

    # pm is the isotropic looking thing, close to kappa + i omega (I guess)
    # pp are the analog og the spin 2 components
    k, w = sjob.alm2map_spin(np.array([0.5 * glm, 0.5 * clm]), 1)
    w -=  0.5 * dt_sintdp_cotsintdp[-1]
    g1, g2 = sjob.alm2map_spin(np.array([0.5 * glm, -0.5 * clm]), 1)
    g2 +=  0.5 * dt_sintdp_cotsintdp[-1]
    #k, w   = 0.5 * (sjob.alm2map_spin(np.array([glm,  clm]), 1) - 1j * dt_sintdp_cotsintdp[-1])
    #g1_ig2 = 0.5 * (sjob.alm2map_spin(np.array([glm, -clm]), 1) + 1j * dt_sintdp_cotsintdp[-1])
    return (1. - k) ** 2 - g1 ** 2 - g2 ** 2 + w ** 2

if __name__ == '__main__':
    import pylab as pl
    import healpy as hp
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
    fl = 0.25 * np.sqrt(np.arange(d.lmax_dlm + 1) * np.arange(1, d.lmax_dlm + 2))
    sjob = utils_scarf.scarfjob()
    sjob.set_geometry(d.geom)
    sjob.set_triangular_alm_info(d.lmax_dlm, d.mmax_dlm)
    sjob.set_nthreads(sht_threads)

    M1 = utils_dlm.dlm2M(sjob, dlm, None)  # 'standard' magn mat
    M2 = get_jacobian(d) # New calc.
