from __future__ import annotations

import numpy as np
from delensalot.core.helper.utils_scarf import Geom
from plancklens.sims import maps


class ztrunc_sims:
    """From a plancklens-style simlib instance on a healpix pixelization makes one returning only a subset of rings

        Args:
            sims: plancklens-style simulation library (wants a get_sim_pmap and get_sim_tmap method)
            nside: healpix resolution of the maps
            zbounds_list: list of non-overlapping inclusive colat bounds. ( [(-1, 1)] for full map )


    """
    def __init__(self, sims:maps.cmb_maps, nside:int, zbounds_list:[tuple[float, float]]):
        self.sims = sims

        hp_geom  = Geom.get_healpix_geometry(nside)
        slics = []
        slics_m = []
        npix = 0
        for zbounds in zbounds_list:
            hp_trunc = Geom.get_healpix_geometry(nside, zbounds=zbounds)
            hp_start = hp_geom.ofs[np.where(hp_geom.theta == np.min(hp_trunc.theta))[0]][0]
            this_npix = Geom.npix(hp_trunc)
            hp_end = hp_start + this_npix
            slics.append(slice(hp_start, hp_end))
            slics_m.append(slice(npix, npix + this_npix))
            npix += this_npix
        self.slics = slics
        self.slics_m  = slics_m
        self.nside = nside
        self.npix = npix

    def get_sim_pmap(self, idx):
        Q, U = self.sims.get_sim_pmap(idx)
        return self.ztruncify(Q), self.ztruncify(U)

    def get_sim_tmap(self, idx):
        return self.ztruncify(self.sims.get_sim_tmap(idx))

    def ztruncify(self, m:np.ndarray):
        assert m.size == 12 * self.nside ** 2, ('unexpected input size', m.size, 12 * self.nside ** 2)
        if len(self.slics) == 1:
            return m[self.slics[0]]
        ret = np.zeros(self.npix, dtype=float)
        for sli_m, sli in zip(self.slics_m, self.slics):
            ret[sli_m] = m[sli]
        return ret