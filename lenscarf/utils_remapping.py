import numpy as np


def _sind_d_m1(d, deriv=False):
    """Series for sind / d  - 1 for small deflections

    """
    assert np.max(d) <= 0.01, (np.max(d), 'CMB Lensing deflections should never be that big')
    d2 = d * d
    if not deriv:
        return np.poly1d([0., -1 / 6., 1. / 120., -1. / 5040.][::-1])(d2)
    else:
        return - 1. / 3. * (1. -  d2 / 10. * ( 1.  - d2 / 28.))

def d2ang(red, imd, tht, phi, version):
    """Builds deflected positions according to deflection field red, imd and undeflected coord. tht and phi.

        This assumes we are close to one of the pole |cos(tht)| ~ 1 and are a bit careful not to lose any precision

        Args:
            red: real part of spin-1 deflection field  (~ dtht on the equator)
            imd: imaginary part of spin-1 deflection field  (~ dphi on the equator)
            tht: undeflected co-latitude coordinate
            phi: undeflected longitude
            version: either 1, 0, -1 whether we are closer to the north pole, equator or south pole

        Returns:
            deflected co-latitudes and longitudes

        Note:

            This uses the following Eqs.after putting them in a form suitable to evaluation close to the poles,
            avoiding clustering of cos tht to 1

            cost' = cost cosd - red/d sind sint
            sint' sin dp =  imd/d * sind
            sint' cos dp =  (cosd - cost cost') / sint = cosd sint + red/d sind cost

    """
    assert version in [1, 0, -1], version
    d = np.sqrt(red ** 2 + imd ** 2)
    assert np.max(d) <= 0.01, (np.max(d), np.max(np.abs(red)), np.max(np.abs(imd)), 'CMB Lensing deflections should never be that big')

    sind_d = 1. + _sind_d_m1(d)  # sin(d) / d avoiding division by zero or near zero, assuming small deflections

    if version == 0:  #---'close' to equator, where cost ~ 0
        cost = np.cos(tht)
        assert np.max(cost < 0.8), ('wrong localization', np.max(cost))
        costp = cost * np.cos(d) - red * sind_d * np.sqrt(1. - cost ** 2) # -- cosd fine here
        dphi = np.arcsin(imd / np.sqrt(1. - costp ** 2) * sind_d) # This is unanbiguous unless d is absurdly high
        thtp = np.arccos(costp)
    else:
        isnorth = version == 1
        sint = np.sin(tht)
        # --- 'e_t' quantities are 1 \mp cos(t) with - if close to 0 and + if close to pi, such that e_t is small and > 0
        e_t = 2 * np.sin(tht * 0.5) ** 2 if isnorth else 2 * np.cos(tht * 0.5) ** 2  # 1 -+ costh with no precision loss
        e_d = 2 * np.sin(d * 0.5) ** 2

        # -- Eq. for new co-latitude (always work fine), here written for 1 - cos tht in order not to lose precision
        e_tp = e_t + e_d - e_t * e_d + version * red * sind_d * sint  # 1 -+ cost'
        sintp = np.sqrt(e_tp * (2 - e_tp))

        # -- deflected coordinates
        if isnorth:
            assert np.max(tht) < np.pi * 0.4, ('wrong localization', np.max(tht))  # -- for the arcsin at the end
            thtp = np.arcsin(sintp)
            dphi = np.arctan2(imd * sind_d, (1. - e_d) * sint + red * sind_d * (1. - e_t))
        else:
            assert np.min(tht) > np.pi * 0.4, ('wrong localization', np.min(tht))  # -- for the arcsin at the end
            thtp = np.pi - np.arcsin(sintp)
            dphi = np.arctan2(imd * sind_d, (1. - e_d) * sint + red * sind_d * (e_t - 1.))
    return thtp, (phi +dphi) % (2. * np.pi)


def ang2d(thtp, tht, dphi):
    """Returns spin-1 deflection components from deflected angles

        Args:
            thtp: deflected co-latitude, in radians
            tht: un-dflected co-latitude, in radians
            dphi: longitude deflection (phip - phi) in radians

        Returns:
            red, imd, arrays with real and imaginary part of spin-1 deflection

    """

    sintp = np.sin(thtp)
    red = np.sin(thtp - tht) - 2 * np.sin(dphi * 0.5) ** 2 * np.cos(tht) * sintp  # red sind / d
    imd = np.sin(dphi) * sintp   # imd sind / d
    sind = np.sqrt(red * red + imd * imd)
    pix = np.where(sind > 0)
    norm = np.arcsin(sind[pix]) / sind[pix]
    red[pix] *= norm[pix]
    imd[pix] *= norm[pix]
    return red, imd