"""Generic cmb-only sims module

"""

import os

import numpy as np, healpy as hp
import pickle as pk

import logging
log = logging.getLogger(__name__)

from delensalot.core import mpi
from plancklens.sims import cmbs, phas
from plancklens import utils


verbose = False

def _get_fields(cls):
    if verbose: print(cls.keys())
    fields = ['p', 't', 'e', 'b', 'o']
    ret = ['p', 't', 'e', 'b', 'o']
    for _f in fields:
        if not ((_f + _f) in cls.keys()): ret.remove(_f)
    for _k in cls.keys():
        for _f in _k:
            if _f not in ret: ret.append(_f)
    return ret

class sims_cmb_len(object):
    """Lensed CMB skies simulation library.

        Note:
            To produce the lensed CMB, the package lenspyx is mandatory

        Note:
            These sims do not contain aberration or modulation

        Args:
            lib_dir: lensed cmb alms will be cached there
            lmax: lensed cmbs are produced up to lmax
            cls_unl(dict): unlensed cmbs power spectra
            lib_pha(optional): random phases library for the unlensed maps (see *plancklens.sims.phas*)
            offsets_plm: offset lensing plm simulation index (useful e.g. for MCN1), tuple with block_size and offsets
            offsets_cmbunl: offset unlensed cmb (useful e.g. for MCN1), tuple with block_size and offsets
            dlmax(defaults to 1024): unlensed cmbs are produced up to lmax + dlmax, for accurate lensing at lmax
            nside_lens(defaults to 4096): healpy resolution at which the lensed maps are produced
            epsilon(defaults to 1e-5): sets the interpolation resolution in lenspyx

            verbose(defaults to True): lenspyx timing info printout

    """
    def __init__(self, lib_dir, lmax, cls_unl, lib_pha=None, offsets_plm=None, offsets_cmbunl=None,
                 dlmax=1024, nside_lens=4096, epsilon=1e-7, nbands=8, cache_plm=True, verbose=True):

        fields = _get_fields(cls_unl)

        if lib_pha is None:
            lib_pha = phas.lib_phas(lib_dir, len(fields), lmax + dlmax)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lmax == lmax + dlmax


        self.lmax = lmax
        self.dlmax = dlmax
        self.lmax_plm = lmax + dlmax
        # lenspyx parameters:
        self.nside_lens = nside_lens
        self.cache_plm = True
        self.epsilon = epsilon

        self.unlcmbs = cmbs.sims_cmb_unl(cls_unl, lib_pha)
        self.lib_dir = lib_dir
        self.fields = _get_fields(cls_unl)

        self.offset_plm = offsets_plm if offsets_plm is not None else (1, 0)
        self.offset_cmb = offsets_cmbunl if offsets_cmbunl is not None else (1, 0)

        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        first_rank = mpi.bcast(mpi.rank)
        if first_rank == mpi.rank:
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            for n in range(mpi.size):
                if n != mpi.rank:
                    mpi.send(1, dest=n)
        else:
            mpi.receive(None, source=mpi.ANY_SOURCE)
        utils.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))
        try:
            import lenspyx
        except ImportError:
            log.info("Could not import lenspyx module")
            lenspyx = None
        self.lens_module = lenspyx
        self.verbose=verbose

    @staticmethod
    def offset_index(idx, block_size, offset):
        """Offset index by amount 'offset' cyclically within blocks of size block_size

        """
        return (idx // block_size) * block_size + (idx % block_size + offset) % block_size

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,
                'offset_plm':self.offset_plm, 'offset_cmb':self.offset_cmb,
                'nside_lens':self.nside_lens, 'facres':self.epsilon}

    def _is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_alm(self, idx, field, ret=True):
        if field == 't':
            return self.get_sim_tlm(idx, ret)
        elif field == 'e':
            return self.get_sim_elm(idx, ret)
        elif field == 'b':
            return self.get_sim_blm(idx, ret)
        elif field == 'p':
            return self.get_sim_plm(idx, ret)
        elif field == 'o':
            return self.get_sim_olm(idx)
        else :
            assert 0, (field,self.fields)

    def get_sim_plm(self, idx, ret=True):
        fn = os.path.join(self.lib_dir, 'plm_in_%04d_lmax%s.fits'%(idx, self.lmax_plm))
        if not os.path.exists(fn):
            plm = self.unlcmbs.get_sim_plm(self.offset_index(idx, self.offset_plm[0], self.offset_plm[1]))
            if self.cache_plm:
                hp.write_alm(fn, plm)
        if ret:
            return hp.read_alm(fn)

    def get_sim_olm(self, idx):
        if 'o' in self.fields:
            return self.unlcmbs.get_sim_olm(idx)
        else:
            return np.zeros_like(self.get_sim_plm(idx))

    def _get_dlm(self, idx):
        dlm = self.get_sim_plm(idx)
        dclm = self.get_sim_olm(idx) # curl mode
        lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        # potentials to deflection
        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
        return dlm, dclm, lmax_dlm, mmax_dlm

    def _cache_eblm(self, idx):
        elm = self.unlcmbs.get_sim_elm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
        blm = None if 'b' not in self.fields else self.unlcmbs.get_sim_blm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
        dlm, dclm, _, _ = self._get_dlm(idx)
        assert 'o' not in self.fields, 'not implemented'

        Qlen, Ulen = self.lens_module.alm2lenmap_spin(np.array([elm, blm]), np.array([dlm, dclm]), 2, epsilon=self.epsilon, verbose=self.verbose)
        elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=self.lmax)
        del Qlen, Ulen
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx), elm)
        del elm
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx), blm)

    def get_sim_tlm(self, idx, ret=True):
        fname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits' % idx)
        if not os.path.exists(fname):
            tlm = self.unlcmbs.get_sim_tlm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
            dlm = self.get_sim_plm(idx)
            assert 'o' not in self.fields, 'not implemented'

            lmaxd = hp.Alm.getlmax(dlm.size)
            hp.almxfl(dlm, np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2)), inplace=True)
            Tlen = self.lens_module.alm2lenmap(np.array(tlm), [dlm, None], epsilon=self.epsilon, verbose=self.verbose)
            hp.write_alm(fname, hp.map2alm(Tlen, lmax=self.lmax, iter=0))
        if ret:
            return hp.read_alm(fname)

    def get_sim_elm(self, idx, ret=True):
        fname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        if ret:
            return hp.read_alm(fname)

    def get_sim_blm(self, idx, ret=True):
        fname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        if ret:
            return hp.read_alm(fname)