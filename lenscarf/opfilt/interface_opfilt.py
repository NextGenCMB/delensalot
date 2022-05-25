"""Anything related to opfilt should be made accessible via this module

Returns:
    _type_: _description_
"""

from lenscarf.opfilt import opfilt_ee_wl

def get_ninv_opfilt(sc):
    zbounds, zbounds_len = sc.get_zbounds()
    return opfilt_ee_wl.alm_filter_ninv_wl(
        libdir, pixn_inv, transf, sc.lmax_filt, plm, bmarg_lmax=sc.BMARG_LCUT, _bmarg_lib_dir=sc.BMARG_LIBDIR,
        olm=olm, sc.nside_lens=2048, nbands_lens=1, facres=-1,zbounds=zbounds, zbounds_len=zbounds_len)
