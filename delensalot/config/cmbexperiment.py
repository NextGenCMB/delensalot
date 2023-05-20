"""Helper function to get often used noise level and beam of CMB experiments"""

import numpy as np 
from delensalot.core.helper import utils_scarf as us

def get_config(exp):
    """Returns noise levels, beam size and multipole cuts for some configurations

    """
    sN_uKaminP = None
    if exp == 'Planck':
        sN_uKamin = 35.
        Beam_FWHM_amin = 7.
        ellmin = 10
        ellmax = 2048
    elif exp == 'S4':
        sN_uKamin = 1.5
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S4_opti':
        sN_uKamin = 1.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_opti':
        sN_uKamin = 11.
        Beam_FWHM_amin = 4.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    else:
        sN_uKamin = 0
        Beam_FWHM_amin = 0
        ellmin = 0
        ellmax = 0
        assert 0, '%s not implemented' % exp
    sN_uKaminP = sN_uKaminP or np.sqrt(2.) * sN_uKamin
    return sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax



def cmbs4_06b():
    zbounds_len = [-0.9736165659024625, -0.4721687661208586]
    pbounds_exl = np.array((113.20399439681668, 326.79600560318335)) #These were the pbounds as defined with the old itercurv conv.
    pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
    pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
    return scarf_job, [pb_ctr / 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len, zbounds_len

def cmbs4_08b_healpix():
    zbounds_len = [-0.9736165659024625, -0.4721687661208586]
    pbounds_exl = np.array((113.20399439681668, 326.79600560318335))
    pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
    pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
    return scarf_job, [pb_ctr/ 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len, zbounds_len

def cmbs4_08b_healpix_oneq():
    zbounds_len = np.cos( (90 + 25) / 180 * np.pi), np.cos( (90 - 25) / 180 * np.pi)
    pb_ctr = 0. / 180 * np.pi
    pb_extent = 50. / 180 * np.pi
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
    return scarf_job, [pb_ctr, pb_extent], zbounds_len, zbounds_len

def cmbs4_08b_healpix_onp(): # square of 50 deg by 50 deg
    extent_deg = 40. # 40 deg ensures each and every point at least 6 deg. away from mask
    zbounds_len = (np.cos(extent_deg / 180 * np.pi), 1.)
    zbounds_ninv =  (np.cos(34. / 180 * np.pi), 1.)
    pb_ctr = 0. / 180 * np.pi
    pb_extent = 360 / 180 * np.pi
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_ninv)
    return scarf_job, [pb_ctr, pb_extent], zbounds_len, zbounds_ninv

def full_sky_healpix():
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048)
    return scarf_job, [0, 2 * np.pi], (-1.,1.), (-1.,1)