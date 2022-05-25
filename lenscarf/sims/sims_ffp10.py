import os
import numpy as np
from lenscarf import cachers
from lenscarf import utils_scarf, utils_hp
from plancklens.sims.planck2018_sims import cmb_unl_ffp10
from lenscarf.remapping import deflection

class cmb_len_ffp10:
    def __init__(self, cacher:cachers.cacher or None=None, lmax_thingauss= 4 * 4096):
        """FFP10 lensed cmbs, lensed with independent lenscarf code on thingauss geometry

        """
        if cacher is None:
            cacher = cachers.cacher_none()
        self.cacher = cacher

        self.fft_tr = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.sht_tr = int(os.environ.get('OMP_NUM_THREADS', 1))

        self.lmax_len = 4096
        self.mmax_len = 4096

        self.targetres = 0.75

        len_geom = utils_scarf.Geom.get_thingauss_geometry(lmax_thingauss, 2)
        # default value seems overkill, but want here same number of rings than healpix for comp

        pbdGeom = utils_scarf.pbdGeometry(len_geom, utils_scarf.pbounds(np.pi, 2 * np.pi ))
        self.pbdGeom = pbdGeom

    def _get_dlm(self, idx):
        dlm = cmb_unl_ffp10.get_sim_plm(idx)
        lmax_dlm = utils_hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        utils_hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        return dlm, lmax_dlm, mmax_dlm

    def _build_eb(self, idx):
        dlm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
        ffi = deflection(self.pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr)
        unl_elm = cmb_unl_ffp10.get_sim_elm(idx)
        unl_blm = cmb_unl_ffp10.get_sim_blm(idx)
        lmax_elm = utils_hp.Alm.getlmax(unl_elm.size, -1)
        mmax_elm = lmax_elm
        assert lmax_elm == utils_hp.Alm.getlmax(unl_blm.size, -1)
        len_elm, len_blm = ffi.lensgclm([unl_elm, unl_blm], mmax_elm, 2, self.lmax_len, self.mmax_len)
        return len_elm, len_blm

    def get_sim_tlm(self, idx):
        fn = 'tlm_%04d'%idx
        if not self.cacher.is_cached(fn):
            dlm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
            ffi = deflection(self.pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr)
            unl_tlm = cmb_unl_ffp10.get_sim_tlm(idx)
            lmax_tlm = utils_hp.Alm.getlmax(unl_tlm.size, -1)
            mmax_tlm = lmax_tlm
            len_tlm = ffi.lensgclm(unl_tlm, mmax_tlm, 0, self.lmax_len, self.mmax_len)
            self.cacher.cache(fn, len_tlm)
            return len_tlm
        return self.cacher.load(fn)

    def get_sim_elm(self, idx):
        fn_e = 'elm_%04d'%idx
        fn_b = 'blm_%04d'%idx
        if not self.cacher.is_cached(fn_e):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_b, len_blm)
            self.cacher.cache(fn_e, len_elm)
            return len_elm
        return self.cacher.load(fn_e)

    def get_sim_blm(self, idx):
        fn_e = 'elm_%04d'%idx
        fn_b = 'blm_%04d'%idx
        if not self.cacher.is_cached(fn_b):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_e, len_elm)
            self.cacher.cache(fn_b, len_blm)
            return len_blm
        return self.cacher.load(fn_b)