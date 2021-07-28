import numpy as np
import scarf

from lenscarf.opfilt import opfilt_ee_wl
from lenscarf import utils_hp, utils_scarf

# FIXME: Put this sd opfilt method? with cachers for EWF and prep (in case rotations are involved) etc?
def get_qlms_wl(qudat:np.ndarray or list, elm_wf:np.ndarray, filt:opfilt_ee_wl.alm_filter_ninv_wl, q_pbgeom:utils_scarf.pbdGeometry):
    """

        Args:
            qudat: input polarization maps (geom must match that of the filter)
            elm_wf: Wiener-filtered CMB maps (alm arrays)
            filt: lenscarf filtering instance
            q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

        #FIXME: all implementation signs are super-weird but end result correct...
    """
    assert len(qudat) == 2
    assert (qudat[0].size == utils_scarf.Geom.npix(filt.ninv_geom)) and (qudat[0].size == qudat[1].size)

    ebwf = (elm_wf, np.zeros_like(elm_wf))
    repmap, impmap = get_irespmap(qudat, ebwf, filt, q_pbgeom)
    Gs, Cs = get_gpmap(ebwf, 3, filt, q_pbgeom)  # 2 pos.space maps
    GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
    Gs, Cs = get_gpmap(ebwf, 1, filt, q_pbgeom)
    GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
    del repmap, impmap, Gs, Cs
    lmax_qlm = filt.ffi.lmax_dlm
    mmax_qlm = filt.ffi.mmax_dlm
    G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, filt.ffi.sht_tr, (-1., 1.))
    del GC
    fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
    utils_hp.almxfl(G, fl, mmax_qlm, True)
    utils_hp.almxfl(C, fl, mmax_qlm, True)
    return G, C

def get_irespmap(qudat:np.ndarray, ebwf:np.ndarray or list, filt:opfilt_ee_wl.alm_filter_ninv_wl, q_pbgeom:utils_scarf.pbdGeometry):
    """Builds inverse variance weighted map to feed into the QE


        :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


    """

    assert len(qudat) == 2 and len(ebwf) == 2
    assert np.all(filt.sc_job.geom.weight == 1.) # sum rather than integrals

    ebwf = filt.ffi.lensgclm(ebwf, filt.mmax_sol, 2, filt.lmax_len, filt.mmax_len, False)
    utils_hp.almxfl(ebwf[0], filt.b_transf, filt.mmax_len, True)
    utils_hp.almxfl(ebwf[1], filt.b_transf, filt.mmax_len, True)
    qu = qudat - filt.sc_job.alm2map_spin(ebwf, 2)
    filt.apply_map(qu)
    ebwf = filt.sc_job.map2alm_spin(qu, 2)
    utils_hp.almxfl(ebwf[0], filt.b_transf * 0.5, filt.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
    utils_hp.almxfl(ebwf[1], filt.b_transf * 0.5, filt.mmax_len, True)
    return q_pbgeom.geom.alm2map_spin(ebwf, 2, filt.lmax_len, filt.mmax_len, filt.ffi.sht_tr, (-1., 1.))

def get_gpmap(eblm_wf:np.ndarray or list, spin:int, filt:opfilt_ee_wl.alm_filter_ninv_wl, q_pbgeom:utils_scarf.pbdGeometry):
    """Wiener-filtered gradient leg to feed into the QE


    :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                   sqrt(l-2 (l+3)) _3 Ylm(n)`

        Output is list with real and imaginary part of the spin 1 or 3 transforms.


    """
    assert len(eblm_wf) == 2
    lmax = utils_hp.Alm.getlmax(eblm_wf[0].size, filt.mmax_sol)
    assert lmax == filt.lmax_sol, (lmax, filt.lmax_sol)
    if spin == 1:
        fl = np.arange( 2, lmax + 3, dtype=float) * np.arange(-1, lmax)
    elif spin == 3:
        fl = np.arange(-2, lmax - 1, dtype=float) * np.arange(3, lmax + 4)
    else:
        assert 0
    fl[:spin] *= 0.
    fl = np.sqrt(fl)
    eblm = [utils_hp.almxfl(eblm_wf[0], fl, filt.mmax_sol, False), utils_hp.almxfl(eblm_wf[1], fl, filt.mmax_sol, False)]
    if q_pbgeom.geom is not filt.ffi.pbgeom:
        ffi = filt.ffi.change_geom(q_pbgeom)
    else:
        ffi = filt.ffi
    return ffi.gclm2lenmap(eblm, filt.mmax_sol, spin, False)
