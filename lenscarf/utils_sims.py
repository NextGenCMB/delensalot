from __future__ import annotations

import numpy as np
from lenscarf.utils_scarf import Geom
from plancklens.sims import maps

class ztrunc_sims:
    """From a plancklens-style simlib instance on a healpix pixelization makes one returning only a subset of rings

        Args:
            sims: plancklens-style simulation library (wants a get_sim_pmap and get_sim_tmap method)
            nside: healpix resolution of the maps
            zbounds: only rings within the cosine colatitude bounds ( (-1, 1) for full maps ) will be returned


    """
    def __init__(self, sims:maps.cmb_maps, nside:int, zbounds:tuple[float, float]):
        self.sims = sims

        hp_geom  = Geom.get_healpix_geometry(nside)
        hp_trunc = Geom.get_healpix_geometry(nside, zbounds=zbounds)
        hp_start = hp_geom.ofs[np.where(hp_geom.theta == np.min(hp_trunc.theta))[0]][0]
        hp_end = hp_start + Geom.npix(hp_trunc).astype(hp_start.dtype)  # Somehow otherwise makes a float out of int64 and uint64 ???

        self.slic = slice(hp_start, hp_end)
        self.nside = nside

    def get_sim_pmap(self, idx):
        Q, U = self.sims.get_sim_pmap(idx)
        return self.ztruncify(Q), self.ztruncify(U)

    def get_sim_tmap(self, idx):
        return self.ztruncify(self.sims.get_sim_tmap(idx))

    def ztruncify(self, m:np.ndarray):
        assert m.size == 12 * self.nside ** 2, ('unexpected input size', m.size, 12 * self.nside ** 2)
        return m[self.slic]