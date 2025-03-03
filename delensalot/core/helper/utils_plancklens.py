import numpy as np
import healpy as hp

from delensalot.utils import cli

from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import data_functions as df


def get_niv_desc(nlev, nivjob_geominfo, nivjob_geomlib, rhits_normalised, mask, niv_map=None, mode='P'):
    """Generate noise inverse variance (NIV) description for temperature ('T') or polarization ('P')."""
    if isinstance(mask, str):
        mask = mask
    if niv_map is None:
        if nivjob_geominfo[0] != 'healpix':
            assert 0, 'needs testing, please choose Healpix geom for nivjob for now'
        pixel_area = hp.nside2pixarea(nivjob_geominfo[1]['nside'], degrees=True) * 3600  # Convert to arcmin²
        rhits_normalised = np.load(rhits_normalised) if rhits_normalised is not None else np.array([1])
        niv_desc = [np.array([pixel_area / nlev[mode] ** 2]*cli(rhits_normalised))]
        niv_desc = niv_desc + [mask] if mask else niv_desc
    else:
        niv_desc = [niv_map] +[mask]
    return niv_desc


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