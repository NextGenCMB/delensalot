import numpy as np
import healpy as hp

# TODO This might rather belong to /config

def get_nlev_mask(ratio, rhits):
    """Mask built thresholding the relative hit counts map
        Note:
            Same as 06b
    """
    rhits_loc = np.nan_to_num(rhits)
    mask = np.where(rhits/np.max(rhits_loc) < 1. / ratio, 0., rhits_loc)
    
    return np.nan_to_num(mask)


def get_zbounds(rhits, hits_ratio=np.inf):
    """Cos-tht bounds for thresholded mask

    """
    pix = np.where(get_nlev_mask(hits_ratio, rhits))[0]
    tht, phi = hp.pix2ang(2048, pix)
    zbounds = np.cos(np.max(tht)), np.cos(np.min(tht))

    return zbounds


def extend_zbounds(zbounds, degrees=5.):
    """Extends given zbounds by some degrees.

    Args:
        zbounds (_type_): _description_

    Returns:
        _type_: _description_
    """
    zbounds_len = [np.cos(np.arccos(zbounds[0]) + degrees / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - degrees / 180 * np.pi)]
    zbounds_len[0] = max(zbounds_len[0], -1.)
    zbounds_len[1] = min(zbounds_len[1],  1.)

    return zbounds_len