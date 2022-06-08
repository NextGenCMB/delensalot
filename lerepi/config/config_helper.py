import numpy as np
import healpy as hp


class data_functions:

    def get_nlev_mask(ratio, rhits):
        """Mask built thresholding the relative hit counts map
            Note:
                Same as 06b
        """
        rhits_loc = np.nan_to_num(rhits)
        mask = np.where(rhits/np.max(rhits_loc) < 1. / ratio, 0., rhits_loc)
        
        return np.nan_to_num(mask)


    def get_zbounds(self, rhits, hits_ratio=np.inf):
        """Cos-tht bounds for thresholded mask

        """
        pix = np.where(data_functions.get_nlev_mask(hits_ratio, rhits))[0]
        tht, phi = hp.pix2ang(2048, pix)
        zbounds = np.cos(np.max(tht)), np.cos(np.min(tht))

        return zbounds


    def extend_zbounds(self, zbounds, degrees=5.):
        zbounds_len = [np.cos(np.arccos(zbounds[0]) + degrees / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - degrees / 180 * np.pi)]
        zbounds_len[0] = max(zbounds_len[0], -1.)
        zbounds_len[1] = min(zbounds_len[1],  1.)

        return zbounds_len


    def get_noisemodel_mask(self):

        return hp.read_map(self.p2mask, field=0)


    def get_noisemodel_path(self):
        
        return self.p2mask