#!/usr/bin/env python

"""config_helper.py: functions for defining config parameter input variables.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import numpy as np
import healpy as hp


class LEREPI_Constants:
    fs_edges = np.arange(2, 3000, 20)
    fs_edges = np.arange(2, 3000, 20)
    ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
    cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])


class data_functions:

    def a2r(val):
        """arcmin2radian converter

        Args:
            val (_type_): _description_

        Returns:
            _type_: _description_
        """
        return val  / 180 / 60 * np.pi
        

    def get_nlev_mask(ratio, rhits):
        """Mask built thresholding the relative hit counts map
            Note:
                Same as 06b
        """
        mask = np.where(rhits/np.max(rhits) < 1. / ratio, 0., rhits)
        
        return np.nan_to_num(mask)


    def get_zbounds(rhits, hits_ratio=np.inf):
        """Cos-tht bounds for thresholded mask

        """
        pix = np.where(data_functions.get_nlev_mask(hits_ratio, rhits))[0]
        tht, phi = hp.pix2ang(2048, pix)
        zbounds = np.cos(np.max(tht)), np.cos(np.min(tht))

        return zbounds


    def extend_zbounds(zbounds, degrees=5.):
        zbounds_len = [np.cos(np.arccos(zbounds[0]) + degrees / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - degrees / 180 * np.pi)]
        zbounds_len[0] = max(zbounds_len[0], -1.)
        zbounds_len[1] = min(zbounds_len[1],  1.)

        return zbounds_len
