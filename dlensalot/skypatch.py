import numpy as np
from dlensalot.utils_sht import lowprimes

class skypatch:
    """This contains simple methods in order to build ECP maps centred on a sky location and link them to latitude rings

        Args:
            tbounds: co-latitudes (rad) delimiting the patch (e.g. 0, pi)
            pbounds: longitudes (rad) delimiting the patch. format is (patch_center, patch_extent)
            targetres_amin: desired resolution of the pixelisation
            pole_buffers: adds a number of pixels at the poles if set (for interpolation purposes e.g.)

        Note:
            The effective resolution will be slightly higher as box shapes with low prime numbers will be prefered


    """
    def __init__(self, tbounds:tuple, pbounds:tuple, targetres_amin:float, pole_buffers:int=0, verbose=False):
        assert (0. <= tbounds[0] < tbounds[1] <= np.pi), tbounds
        assert (pbounds[1] > 0), pbounds
        #-- inclusive bounds

        self.tbounds = tbounds
        self.pbounds = pbounds
        self.res_desired_amin = targetres_amin

        colat_bounds = [max(0., tbounds[0]), min(tbounds[1], np.pi)]
        northp = colat_bounds[0] <= 0.
        southp = colat_bounds[1] >= np.pi

        self.colat_bounds = colat_bounds

        self.nt_buffers_n =  pole_buffers *  northp
        self.nt_buffers_s  = pole_buffers *  southp

        self.verbose = verbose

    def ecp_ring_ntphi(self):
        """Number of tht and phi points matching desired resolution and powers of low primes


        """
        cross_eq = ( (np.pi / 2 - self.colat_bounds[0]) * (np.pi / 2 - self.colat_bounds[1])) <= 0
        max_sinth = 1. if cross_eq else np.max(np.sin(self.colat_bounds))
        nt_min =  (self.colat_bounds[1] - self.colat_bounds[0]) / (self.res_desired_amin / 60 / 180 * np.pi)
        np_min = max_sinth * (2 * np.pi)/ (self.res_desired_amin / 60 / 180 * np.pi)
        nt = lowprimes(np.ceil(nt_min + self.nt_buffers_n + self.nt_buffers_s))
        #: add a buffer for poles bicubic spline interp
        nph = lowprimes(np.ceil(np_min))
        rest_amin = (self.colat_bounds[1] - self.colat_bounds[0]) / np.pi * 180 * 60 / (nt - self.nt_buffers_n - self.nt_buffers_s)
        resp_amin = max_sinth * 360 * 60 / nph
        if self.verbose:
            print("achieved nt np %s %s res %.2f (tht) %.2f (phi) amins"%(nt, nph, rest_amin, resp_amin))
        return nt, nph

    def ecp_resize(self, nph_ecp):
        """Builds slices resizing a full 2pi ECP map onto the pbounds


            Note:
                This assumes a patch centred on pi

        """
        # ------|--------|-------
        prange = self.pbounds[1]
        first = max(0, int(np.floor((np.pi - prange * 0.5) * nph_ecp / (2 * np.pi))))
        last = min(nph_ecp - 1, int(np.ceil((np.pi + prange * 0.5) * nph_ecp / (2 * np.pi))))
        nph_ap = lowprimes(last - first + 1)
        if nph_ap < nph_ecp:
            dn = nph_ap - (last - first + 1)
            dhp = dn // 2
            dmp = dn - dhp
            first -= dmp
            last += dhp
        assert first >= 0 and last < nph_ecp, (first, last)
        return first, last
