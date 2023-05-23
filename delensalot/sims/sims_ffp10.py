"""This modules contains simulations libraries involving the Planck FFP10 CMBs on NERSC

    The FFP10 lensed CMB's contain a number of lensing-related small defects.
    This module can be used to regenerate customized lensed maps


"""
from __future__ import annotations
import os
import numpy as np
import plancklens.sims.phas

import lenspyx.remapping.utils_geom as utils_geom

from delensalot.core import cachers

from delensalot.utility import utils_hp
from plancklens.sims.planck2018_sims import cmb_unl_ffp10
from plancklens import utils
from lenspyx.remapping import deflection

aberration_lbv_ffp10 = (264. * (np.pi / 180), 48.26 * (np.pi / 180), 0.001234)

class cmb_len_ffp10:
    def __init__(self, aberration:tuple[float, float, float]or None=None, lmin_dlm=0, cacher:cachers.cacher or None=None,
                       lmax_thingauss:int=5120, nbands:int=1, targetres=0.75, verbose:bool=False, plm_shuffle:callable or None=None):

        """FFP10 lensed cmbs, lensed with independent delensalot code on thingauss geometry

            Args:
                aberration: aberration parameters (gal. longitude (rad), latitude (rad) and v/c) Defaults to FFP10 values
                lmin_dlm: Optionally set to zero the deflection field for L<lmin_dlm
                cacher: set this to one of delensalot.cachers in order save maps (nothing saved by default)
                nbands: if set splits the sky into bands to perform the operations (saves some memory but probably a bit slower)
                targetres: main accuracy parameter; target resolution in arcmin to perform the deflection operation.
                           make this smaller for more accurate interpolation
                plm_shuffle: reassigns deflection indices if set to a callable giving new deflection index.
                             Useful to generate sims with independent CMB but same deflection, or vice-versa


        """

        nbands = int(nbands + (1 - int(nbands)%2))  # want an odd number to avoid a split just on the equator
        assert nbands <= 10, 'did not check'
        if cacher is None:
            cacher = cachers.cacher_none() # This cacher saves nothing
        if aberration is None:
            aberration = aberration_lbv_ffp10
        self.cacher = cacher

        self.fft_tr = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.sht_tr = int(os.environ.get('OMP_NUM_THREADS', 1))

        self.lmax_len = 4096 # FFP10 lensed CMBs were designed for this lmax
        self.mmax_len = 4096
        self.lmax_thingauss = lmax_thingauss
        self.lmin_dlm = lmin_dlm

        self.targetres = targetres  # Main accuracy parameter. This crudely matches the FFP10 pipeline's

        zls, zus = self._mkbands(nbands)
        # By construction the central one covers the equator
        len_geoms = [utils_geom.Geom.get_thingauss_geometry(lmax_thingauss, 2)]
        len_geoms[0].restrict(zls[nbands//2], zus[nbands//2])
        for ib in range(nbands//2):
            # and the other ones are symmetric w.r.t. the equator. We merge them to get faster SHTs
            geom_south = utils_geom.Geom.get_thingauss_geometry(lmax_thingauss, 2, zbounds=(zls[ib], zus[ib]))
            geom_north = utils_geom.Geom.get_thingauss_geometry(lmax_thingauss, 2, zbounds=(zls[nbands-ib-1], zus[nbands-ib-1]))
            len_geoms.append(utils_geom.Geom.merge([geom_north, geom_south]))
        pbdGeoms = [utils_geom.pbdGeometry(len_geom, utils_geom.pbounds(np.pi, 2 * np.pi)) for len_geom in len_geoms]

        # Sanity check, we cant have rings overlap
        ref_geom = utils_geom.Geom.get_thingauss_geometry(self.lmax_thingauss, 2)
        tht_all = np.concatenate([pbgeo.geom.theta for pbgeo in pbdGeoms])
        assert np.all(np.sort(ref_geom.theta) == np.sort(tht_all))

        self.pbdGeoms = pbdGeoms

        # aberration: we must add the difference to the FFP10 aberration
        l, b, v = aberration
        l_ffp10, b_ffp10, v_ffp10 = aberration_lbv_ffp10

        # \phi_{10} = - \sqrt{4\pi/3} n_z
        # \phi_{11} = + \sqrt{4\pi / 3} \frac{(n_x - i n_y)}{\sqrt{2}}
        vlm = np.array([0., np.cos(b), - np.exp(-1j * l) * np.sin(b) / np.sqrt(2.)])  # LM = 00, 10 and 11
        vlm_ffp10 = np.array([0., np.cos(b_ffp10), - np.exp(-1j * l_ffp10) * np.sin(b_ffp10) / np.sqrt(2.)])
        vlm       *= (-v * np.sqrt(4 * np.pi / 3))
        vlm_ffp10 *= (-v_ffp10 * np.sqrt(4 * np.pi / 3))
        self.delta_vlm = vlm - vlm_ffp10
        self.vlm = vlm
        if verbose:
            print("Input aberration power %.3e"%(utils_hp.alm2cl(vlm, vlm, 1, 1, 1)[1]))
        self.verbose = verbose

        self.plm_shuffle = plm_shuffle

    @staticmethod
    def _mkbands(nbands: int):
        """Splits the sky in nbands regions with equal numbers of latitude points """
        thts, dt = np.linspace(0, np.pi, nbands + 1), 0. / 180 / 60 * np.pi  # no buffer here
        th_l = thts[:-1] - dt
        th_u = thts[1:] + dt
        th_l[0] = 0.
        th_u[-1] = np.pi
        zu = np.cos(th_l[::-1])
        zl = np.cos(th_u[::-1])
        zl[0] = -1.
        zu[-1] = 1.
        return zl, zu

    def hashdict(self):
        ret = {'sims':'ffp10', 'tres':self.targetres, 'lmaxGL':self.lmax_thingauss, 'lmin_dlm':self.lmin_dlm}
        cl_aber = utils_hp.alm2cl(self.vlm, self.vlm, 1, 1, 1)
        if np.any(cl_aber):
            ret['aberration'] = cl_aber
        if self.plm_shuffle is not None:
            ret['pshuffle'] = [self.plm_shuffle(idx) for idx in range(20)]
        return ret

    def _get_dlm(self, idx):
        if self.plm_shuffle is None:
            shuffled_idx = idx
        else:
            shuffled_idx = self.plm_shuffle(idx)
        dlm = cmb_unl_ffp10.get_sim_plm(shuffled_idx) # gradient mode
        dclm = None # curl mode
        lmax_dlm = utils_hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        dlm[utils_hp.Alm.getidx(lmax_dlm, 1, 0)] += self.delta_vlm[1] # LM=10 aberration
        dlm[utils_hp.Alm.getidx(lmax_dlm, 1, 1)] += self.delta_vlm[2] # LM = 11

        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        p2d[:self.lmin_dlm] = 0

        utils_hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        return dlm, dclm, lmax_dlm, mmax_dlm

    def _build_eb(self, idx, unl_elm=None, unl_blm=None, lmax_len=None, mmax_len=None):
        dlm, dclm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
        if lmax_len is None:
            lmax_len = self.lmax_len
        if mmax_len is None:
            mmax_len = min(lmax_len, self.mmax_len)
        len_eblm = np.zeros((2, utils_hp.Alm.getsize(lmax_len, mmax_len)), dtype=complex)
        if unl_elm is None:
            unl_elm = cmb_unl_ffp10.get_sim_elm(idx)
        if unl_blm is None:
            unl_blm = cmb_unl_ffp10.get_sim_blm(idx)
        lmax_elm = utils_hp.Alm.getlmax(unl_elm.size, -1)
        mmax_elm = lmax_elm
        assert lmax_elm == utils_hp.Alm.getlmax(unl_blm.size, -1)
        for i, pbdGeom in utils.enumerate_progress(self.pbdGeoms, 'collecting bands'):
            ffi = deflection(pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr, verbose=self.verbose, dclm=dclm)
            len_eblm += ffi.lensgclm([unl_elm, unl_blm], mmax_elm, 2, lmax_len, mmax_len)
        return len_eblm

    def get_sim_tlm(self, idx):
        fn = 'tlm_%04d' % idx
        if not self.cacher.is_cached(fn):
            dlm, dclm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
            unl_tlm = cmb_unl_ffp10.get_sim_tlm(idx)
            len_tlm = np.zeros(utils_hp.Alm.getsize(self.lmax_len, self.mmax_len), dtype=complex)
            lmax_tlm = utils_hp.Alm.getlmax(unl_tlm.size, -1)
            mmax_tlm = lmax_tlm
            for i, pbdGeom in utils.enumerate_progress(self.pbdGeoms, 'collecting bands'):
                ffi = deflection(pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr, verbose=self.verbose, dclm=dclm)
                len_tlm += ffi.lensgclm(unl_tlm, mmax_tlm, 0, self.lmax_len, self.mmax_len)
            self.cacher.cache(fn, len_tlm)
            return len_tlm
        return self.cacher.load(fn)

    def get_sim_eblm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_e) or not self.cacher.is_cached(fn_b):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_b, len_blm)
            self.cacher.cache(fn_e, len_elm)
            return len_elm, len_blm
        return self.cacher.load(fn_e), self.cacher.load(fn_b)

    def get_sim_elm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_e):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_b, len_blm)
            self.cacher.cache(fn_e, len_elm)
            return len_elm
        return self.cacher.load(fn_e)

    def get_sim_blm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_b):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_e, len_elm)
            self.cacher.cache(fn_b, len_blm)
            return len_blm
        return self.cacher.load(fn_b)


class cmb_len_ffp10_wcurl(cmb_len_ffp10):
        def __init__(self, clxx:np.ndarray, lib_phas:plancklens.sims.phas.lib_phas, aberration:tuple[float, float, float]or None=None, lmin_dlm=0, cacher:cachers.cacher or None=None,
                       lmax_thingauss:int=5120, nbands:int=1, targetres=0.75, verbose:bool=False, plm_shuffle:callable or None=None):
            """FFP10 lensed CMBs, where lensing including an additional lensing curl potential component

                Args:
                    clxx: lensing curl potential power spectrum
                    lib_phas: random phases of the curl sims (the code will call the '0'th field index of these phases)

                See mother class for other args

                Note:
                    Note: the curl :`Lmax`: is at most the lensing potential :`Lmax`: here

                Note:
                    Deflection is defined as $-\eth (\phi + i \Omega)$ ('x' usually stands in plancklens for $\Omega$)



            """
            super().__init__(aberration=aberration, lmin_dlm=lmin_dlm, cacher=cacher, lmax_thingauss=lmax_thingauss,
                             nbands=nbands, targetres=targetres, verbose=verbose, plm_shuffle=plm_shuffle)

            assert np.all(clxx >= 0.), 'Somethings wrong with the input'

            self.rclxx = np.sqrt(clxx[:lib_phas.lmax+1])
            self.lib_phas = lib_phas
            self.plm_shuffle = plm_shuffle

        def hashdict(self):
            ret = {'sims': 'ffp10', 'tres': self.targetres, 'lmaxGL': self.lmax_thingauss,
                   'lmin_dlm': self.lmin_dlm}
            cl_aber = utils_hp.alm2cl(self.vlm, self.vlm, 1, 1, 1)
            if np.any(cl_aber):
                ret['aberration'] = cl_aber
            ret['rclxx'] = self.rclxx
            ret['xphas'] = self.lib_phas.lib_phas[0].hashdict()
            if self.plm_shuffle is not None:
                ret['pshuffle'] = [self.plm_shuffle(idx) for idx in range(20)]
            return ret

        def _get_dlm(self, idx):
            if self.plm_shuffle is None:
                shuffled_idx = idx
            else:
                shuffled_idx = self.plm_shuffle(idx)
            dlm = cmb_unl_ffp10.get_sim_plm(shuffled_idx)
            lmax_dlm = utils_hp.Alm.getlmax(dlm.size, -1)
            mmax_dlm = lmax_dlm

            dlm[utils_hp.Alm.getidx(lmax_dlm, 1, 0)] += self.delta_vlm[1] # LM = 10 aberration
            dlm[utils_hp.Alm.getidx(lmax_dlm, 1, 1)] += self.delta_vlm[2] # LM = 11

            # curl mode
            dclm = utils_hp.almxfl(self.lib_phas.get_sim(shuffled_idx, idf=0), self.rclxx, None, False)
            dclm = utils_hp.alm_copy(dclm, None, lmax_dlm, mmax_dlm)

            # potentials to deflection
            p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
            p2d[:self.lmin_dlm] = 0

            utils_hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
            utils_hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
            return dlm, dclm, lmax_dlm, mmax_dlm



class cmb_len_ffp10_shuffle_dlm(cmb_len_ffp10):
    def __init__(self, dlm_idxs:dict={}, aberration:tuple[float, float, float]or None=None, lmin_dlm=0, cacher:cachers.cacher or None=None,
                       lmax_thingauss:int=5120, nbands:int=1, targetres=0.75, verbose:bool=False):

        r"""Library of simulations with remaped the defelection field indice.
            Useful for MC-N1 computations.

            Args:
                dlm_idxs : index idx of this instance points to a sim with defelection field index idxs[idx] but CMB fields index idx
        """

        super(cmb_len_ffp10_shuffle_dlm, self).__init__(aberration=aberration, lmin_dlm=lmin_dlm, cacher=cacher,
                       lmax_thingauss=lmax_thingauss, nbands=nbands, targetres=targetres, verbose=verbose)
        self.dlm_idxs = dlm_idxs

    def _get_dlm(self, idx):
        assert idx in self.dlm_idxs.keys(), f"Index {idx} is not assigned in the dlm_idxs remapping"
        return super(cmb_len_ffp10_shuffle_dlm, self)._get_dlm(self.dlm_idxs[idx])