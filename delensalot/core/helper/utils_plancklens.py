import numpy as np
import healpy as hp

from delensalot.utils import cli

from delensalot.utility.utils_hp import gauss_beam
from delensalot.config.config_helper import data_functions as df


def get_niv_desc(nlev, nivjob_geominfo, nivjob_geomlib, rhits_normalised, mask, niv_map=None, mode='P'):
    """Generate noise inverse variance (NIV) description for temperature ('T') or polarization ('P')."""
    rhits_normalised = None
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
        niv_desc = [niv_map] + [mask]
    return niv_desc


def gauss_beamtransferfunction_logistic(beam, lm_max, lmin_teb, with_pixwin=False, geominfo=None, transition_width=1.5):
    """
    Computes a Gaussian beam transfer function with a smooth transition at lmin_teb.
    
    Parameters:
        beam (float): Beam FWHM in arcmin.
        lm_max (list or tuple): Maximum multipole for each field.
        lmin_teb (int): The lower limit for TEB modes, below which modes are suppressed.
        transition_width (int): The width of the transition region for smooth suppression.
        with_pixwin (bool): Whether to include the pixel window function.
        geominfo (tuple): Geometry information for pixel window function.

    Returns:
        dict: A dictionary containing smoothed transfer functions for 'T', 'E', and 'B'.
    """
    # Compute Gaussian beam factor
    beam_factor = gauss_beam(df.a2r(beam), lmax=lm_max[0])

    # Define smooth transition instead of a hard cutoff
    ell = np.arange(lm_max[0] + 1)[:, None]
    transition_mask = 1 / (1 + np.exp(-(ell - lmin_teb) / transition_width))  # Logistic function

    if not with_pixwin:
        transf = beam_factor[:, None] * transition_mask
    else:
        assert geominfo[0] == 'healpix', 'Implement non-healpix pixelwindow function'
        pixwin_factor = hp.pixwin(geominfo[1]['nside'], lmax=lm_max[0])
        transf = beam_factor[:, None] * pixwin_factor[:, None] * transition_mask

    return dict(zip('teb', transf.T))


def gauss_beamtransferfunction_cosine(beam, lm_max, lmin_teb, with_pixwin=False, geominfo=None, transition_width=5):
    def cosine_taper(ell, lmin, width):
        """Cosine taper function for smooth suppression."""
        taper = 0.5 * (1 + np.cos(np.pi * (ell - lmin) / width))
        taper[ell < lmin - width] = 0  # Strictly zero below lmin - width
        taper[ell > lmin] = 1  # Fully 1 above lmin
        return taper

    # Compute Gaussian beam factor
    beam_factor = gauss_beam(df.a2r(beam), lmax=lm_max[0])

    # Define multipole array
    ell = np.arange(lm_max[0] + 1)[:, None]

    # Apply taper function separately to T, E, B
    transition_masks = [cosine_taper(ell, lmin, transition_width) for lmin in lmin_teb]

    if not with_pixwin:
        transf = [beam_factor[:, None] * mask for mask in transition_masks]
    else:
        assert geominfo[0] == 'healpix', 'Implement non-healpix pixelwindow function'
        pixwin_factor = hp.pixwin(geominfo[1]['nside'], lmax=lm_max[0])
        transf = [beam_factor[:, None] * pixwin_factor[:, None] * mask for mask in transition_masks]


    return dict(zip('teb', [t.squeeze().T for t in transf]))  # Ensure correct orientation


def gauss_beamtransferfunction_sharp(beam, lm_max, lmin_teb, with_pixwin=False, geominfo=None):
    beam_factor = gauss_beam(df.a2r(beam), lmax=lm_max[0])
    lmin_mask = np.arange(lm_max[0] + 1)[:, None] >= lmin_teb
    if not with_pixwin:
        transf = (beam_factor[:, None] * lmin_mask)
    elif with_pixwin:
        assert geominfo[0] == 'healpix', 'implement non-healpix pixelwindow function'
        pixwin_factor = hp.pixwin(geominfo[1]['nside'], lmax=lm_max[0])
        transf = (beam_factor[:, None] * pixwin_factor[:, None] * lmin_mask)
    return dict(zip('teb', transf.T))