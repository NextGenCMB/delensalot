#!/usr/bin/env python

"""OBD'd version of Plancklens.filt.filt_cinf.cinv_p. This is exactly itercurvs filt.utils_cinv_p

Returns:
    _type_: _description_
"""

# TODO proper integration into .opfilt structure?

import sys, os
import numpy as np
import healpy as hp
import pickle as pk

from plancklens import utils
from plancklens.helpers import mpi
from plancklens.qcinv import cd_solve, opfilt_pp, multigrid
from plancklens.qcinv import util, util_alm
from plancklens.filt import filt_cinv

from lenscarf.opfilt import bmodes_ninv

class cinv_p(filt_cinv.cinv):
    r"""Polarization-only inverse-variance (or Wiener-)filtering instance.

        Args:
            lib_dir: mask and other things will be cached there
            lmax: filtered alm's are reconstructed up to lmax
            nside: healpy resolution of maps to filter
            cl: fiducial CMB spectra used to filter the data (dict with 'tt' key)
            transf: CMB maps transfer function (array)
            ninv: inverse pixel variance maps. Must be a list of either 3 (QQ, QU, UU) or 1 (QQ = UU noise) elements.
                  These element are themselves list of paths or of healpy maps with consistent nside.

        Note:
            this implementation does not support template projection

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, geom,
                 pcf='default', chain_descr=None, _bmarg_lib_dir=None, _bmarg_rescal=1., zbounds=(-1., 1.), bmarg_lmax=0, sht_threads=8):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        super(cinv_p, self).__init__(lib_dir, lmax)
        
        self.nside = nside
        self.cl = cl
        self.transf = transf
        self.ninv = ninv
        
        pcf = os.path.join(lib_dir, "dense.pk") if pcf == 'default' else None
        if chain_descr is None: chain_descr = \
            [[2, ["split(dense(" + pcf + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,cd_solve.cache_mem()],
             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]

        n_inv_filt = util.jit(bmodes_ninv.eblm_filter_ninv, geom, ninv, transf[0:lmax + 1],
                              lmax_marg=bmarg_lmax, zbounds=zbounds, _bmarg_lib_dir=_bmarg_lib_dir, _bmarg_rescal=_bmarg_rescal, sht_threads=sht_threads)
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_pp, chain_descr, cl, n_inv_filt)

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(os.path.join(lib_dir, "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir, "fbl.dat")):
                fel, fbl = self._calc_febl()
                np.savetxt(os.path.join(self.lib_dir, "fel.dat"), fel)
                np.savetxt(os.path.join(self.lib_dir, "fbl.dat"), fbl)

            if not os.path.exists(os.path.join(self.lib_dir, "tal.dat")):
                np.savetxt(os.path.join(self.lib_dir, "tal.dat"), self._calc_tal())

            if not os.path.exists(os.path.join(self.lib_dir,  "fmask.fits.gz")):
                hp.write_map(os.path.join(self.lib_dir,  "fmask.fits.gz"),  self._calc_mask())

        mpi.barrier()
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'clee': utils.clhash(self.cl.get('ee', np.array([0.]))),
                'cleb': utils.clhash(self.cl.get('eb', np.array([0.]))),
                'clbb': utils.clhash(self.cl.get('bb', np.array([0.]))),
                'transf':utils.clhash(self.transf),
                'ninv': self._ninv_hash()}


    def apply_ivf(self, tmap, soltn=None):
        if soltn is not None:
            assert len(soltn) == 2
            assert hp.Alm.getlmax(soltn[0].size) == self.lmax, (hp.Alm.getlmax(soltn[0].size), self.lmax)
            assert hp.Alm.getlmax(soltn[1].size) == self.lmax, (hp.Alm.getlmax(soltn[1].size), self.lmax)
            talm = util_alm.eblm([soltn[0], soltn[1]])
        else:
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            talm = util_alm.eblm([telm, tblm])

        assert len(tmap) == 2
        self.chain.solve(talm, [tmap[0], tmap[1]])

        return talm.elm, talm.blm

    def _calc_febl(self):
        assert not 'eb' in self.chain.s_cls.keys()

        if len(self.chain.n_inv_filt.n_inv) == 1:
            ninv = self.chain.n_inv_filt.n_inv[0]
            npix = len(ninv)
            NlevP_uKamin = np.sqrt(
                4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
            # 4. * np.pi/np.sum(ninv) *  len(np.where(ninv != 0.0)[0])) / len(ninv) *  * 180. * 60. / np.pi
        else:
            assert len(self.chain.n_inv_filt.n_inv) == 3
            ninv = self.chain.n_inv_filt.n_inv
            NlevP_uKamin= 0.5 * np.sqrt(
                4. * np.pi / len(ninv[0]) / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
            NlevP_uKamin += 0.5 * np.sqrt(
                4. * np.pi / len(ninv[2]) / np.sum(ninv[2]) * len(np.where(ninv[2] != 0.0)[0])) * 180. * 60. / np.pi


        print("cinv_p::noiseP_uk_arcmin = %.3f"%NlevP_uKamin)

        s_cls = self.chain.s_cls    
        b_transf = self.chain.n_inv_filt.b_transf
        fel = 1.0 / (s_cls['ee'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)
        fbl = 1.0 / (s_cls['bb'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)

        fel[0:2] *= 0.0
        fbl[0:2] *= 0.0

        return fel, fbl

    def _calc_tal(self):
        return utils.cli(self.transf)

    def _calc_mask(self):
        mask = np.ones(hp.nside2npix(self.nside), dtype=float)
        for ninv in self.chain.n_inv_filt.n_inv:
            assert hp.npix2nside(len(ninv)) == self.nside
            mask *= (ninv > 0.)
        return mask

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv[0]:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return [ret]