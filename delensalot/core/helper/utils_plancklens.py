import numpy as np
import healpy as hp

from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import data_functions as df


def get_niv_desc(nlev, nivjob_geominfo, nivjob_geomlib, rhits_normalised, mask, niv_map=None, mode='P'):
    """Generate noise inverse variance (NIV) description for temperature ('T') or polarization ('P')."""
    masks, noisemodel_rhits_map = get_masks(nivjob_geomlib, rhits_normalised, mask)
    noisemodel_norm = np.max(noisemodel_rhits_map)

    if niv_map is None:
        if nivjob_geominfo[0] != 'healpix':
            assert 0, 'needs testing, please choose Healpix geom for nivjob for now'
        pixel_area = hp.nside2pixarea(nivjob_geominfo[1]['nside'], degrees=True) * 3600  # Convert to arcmin²
        niv_desc = [np.array([pixel_area / nlev[mode] ** 2]) / noisemodel_norm] + masks
    else:
        niv_desc = [np.load(niv_map)] + masks
    return niv_desc


def get_masks(nivjob_geomlib, rhits_normalised, mask):
    # TODO refactor. This here generates a mask from the rhits map..
    # but this should really be detached from one another
    masks = []
    if rhits_normalised is not None:
        msk = df.get_nlev_mask(rhits_normalised[1], hp.read_map(rhits_normalised[0]))
    else:
        msk = np.ones(shape=nivjob_geomlib.npix())
    masks.append(msk)
    if mask is not None:
        if type(mask) == str:
            _mask = mask
        elif mask[0] == 'nlev':
            noisemodel_rhits_map = msk.copy()
            _mask = df.get_nlev_mask(mask[1], noisemodel_rhits_map)
            _mask = np.where(_mask>0., 1., 0.)
    else:
        _mask = np.ones(shape=nivjob_geomlib.npix())
    masks.append(_mask)

    return masks, msk


def gauss_beamtransferfunction(beam, lm_max, lmin_teb, with_pixwin=False, geominfo=None):
    beam_factor = gauss_beam(df.a2r(beam), lmax=lm_max[0])
    lmin_mask = np.arange(lm_max[0] + 1)[:, None] >= lmin_teb
    if not with_pixwin:
        transf = (beam_factor[:, None] * lmin_mask)
    elif with_pixwin:
        assert geominfo[0] == 'healpix', 'implement non-healpix pixelwindow function'
        pixwin_factor = hp.pixwin(geominfo[1]['nside'], lmax=lm_max[0])
        transf = (beam_factor[:, None] * pixwin_factor[:, None] * lmin_mask)
    return dict(zip('teb', transf.T))