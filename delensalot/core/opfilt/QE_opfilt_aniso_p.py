"""Lenspyx-geometry based inverse-variance filter, without any lensing remapping


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from lenspyx.remapping import utils_geom

from delensalot.utils import timer, cli, clhash, read_map
from delensalot.utility.utils_hp import almxfl, Alm, alm2cl
from delensalot.core.opfilt import bmodes_ninv as bni



class alm_filter_ninv(object):
    def __init__(self, ninv_geom:utils_geom.Geom, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,
                 transf_b:np.ndarray or None=None,
                 tpl:bni.template_dense or None=None, verbose=False):
        r"""CMB inverse-variance and Wiener filtering instance, to use for cg-inversion

            Args:
                ninv_geom: lenspyx geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (itself can be (a list of) string, or array, or ...)
                transf: CMB transfer function (assumed to be the same in E and B)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for lenspyx SHTs
                verbose: some printout if set, defaults to False


        """
        transf_elm = transf
        transf_blm = transf_b if transf_b is not None else transf
        assert transf_blm.size == transf_elm.size, 'check if not same size OK'

        self.n_inv = ninv
        self.transf_elm = transf_elm
        self.transf_blm = transf_blm
        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = max(len(transf), len(transf_blm)) - 1
        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)
        self.lmax_sol = lmax_unl
        self.mmax_sol = min(lmax_unl, mmax_unl)

        self.sht_threads = sht_threads
        self.ninv_geom = ninv_geom

        self.verbose=verbose

        self._nlevp = None
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed



    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.transf_elm),
                'unalm':(self.lmax_sol, self.mmax_sol),
                'lenalm':(self.lmax_len, self.mmax_len) }

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.n_inv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return ret

    def get_febl(self):
        if self._nlevp is None:
            if len(self.n_inv) == 1:
                nlev_febl =  10800. / np.sqrt(np.sum(read_map(self.n_inv[0])) / (4.0 * np.pi)) / np.pi
            elif len(self.n_inv) == 3:
                nlev_febl = 10800. / np.sqrt( (0.5 * np.sum(read_map(self.n_inv[0])) + np.sum(read_map(self.n_inv[2]))) / (4.0 * np.pi))  / np.pi
            else:
                assert 0
            self._nlevp = nlev_febl
        fel = self.transf_elm ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        fbl = self.transf_blm ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return fel, fbl

    def apply_alm(self, eblm:np.ndarray):
        """Applies operator B^T N^{-1} B

        """
        assert self.lmax_sol == self.lmax_len, (self.lmax_sol, self.lmax_len) # not implemented wo lensing
        assert self.mmax_sol == self.mmax_len, (self.mmax_sol, self.mmax_len)

        tim = timer(True, prefix='opfilt_pp')
        lmax_unl = Alm.getlmax(eblm[0].size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')

        qumap = self.ninv_geom.synthesis(eblm, 2, self.lmax_len, self.mmax_len, self.sht_threads)
        tim.add('alm2map_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.ninv_geom.theta.size))

        self.apply_map(qumap)  # applies N^{-1}
        tim.add('apply ninv')

        self.ninv_geom.adjoint_synthesis(qumap, 2, self.lmax_sol, self.mmax_sol, self.sht_threads,
                                                   apply_weights=False, alm=eblm)
        tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.ninv_geom.theta.size))

        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')
        if self.verbose:
            print(tim)

    def apply_map(self, qumap):
        """Applies pixel inverse-noise variance maps


        """
        if len(self.n_inv) == 1:  #  QQ = UU
            qumap *= self.n_inv[0]
            if self.template is not None:
                ts = [self.template] # Hack, this is only meant for one template
                coeffs = np.concatenate(([t.dot(qumap) for t in ts]))
                coeffs = np.dot(ts[0].tniti(), coeffs)
                pmodes = np.zeros_like(qumap)
                im = 0
                for t in ts:
                    t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                    im += t.nmodes
                pmodes *= self.n_inv[0]
                qumap -= pmodes

        elif len(self.n_inv) == 3:  # QQ, QU, UU
            assert self.template is None
            qmap, umap = qumap
            qmap_copy = qmap.copy()
            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap
            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy
            del qmap_copy
        else:
            assert 0

    def get_qlms(self, qudat: np.ndarray or list, eblm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, lmax_qlm, mmax_qlm):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                eblm_wf: Wiener-filtered CMB maps (alm arrays)
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs
                lmax_qlm: maximum multipole of output
                mmax_qlm: maximum m of lm output

        """
        assert len(qudat) == 2 and len(eblm_wf)
        assert (qudat[0].size == self.geom_.npix()) and (qudat[0].size == qudat[1].size)

        repmap, impmap = self._get_irespmap(qudat, eblm_wf, q_pbgeom)
        Gs, Cs = self._get_gpmap(eblm_wf, 3, q_pbgeom)  # 2 pos.space maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(eblm_wf, 1, q_pbgeom)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.sht_threads, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_gpmap(self, eblm_wf:np.ndarray or list, spin:int, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert len(eblm_wf) == 2
        assert  Alm.getlmax(eblm_wf[0].size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(eblm_wf[0].size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(eblm_wf[0].size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        eblm = [almxfl(eblm_wf[0], fl, self.mmax_sol, False), almxfl(eblm_wf[1], fl, self.mmax_sol, False)]
        return q_pbgeom.geom.synthesis(eblm, spin, lmax, self.mmax_sol, self.sht_threads)

    def _get_irespmap(self, qu_dat:np.ndarray, eblm_wf:np.ndarray or list, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE

                :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert len(qu_dat) == 2 and len(eblm_wf) == 2, (len(eblm_wf), len(qu_dat))
        ebwf = np.copy(eblm_wf)
        almxfl(ebwf[0], self.transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], self.transf_blm, self.mmax_len, True)
        qu = qu_dat - self.ninv_geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads)
        self.apply_map(qu)
        self.ninv_geom.adjoint_synthesis(qu, 2, self.lmax_sol, self.mmax_sol, self.sht_threads,
                                                   apply_weights=False, alm=ebwf)
        almxfl(ebwf[0], self.transf_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.transf_blm * 0.5, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads)

pre_op_dense = None # not implemented

def calc_prep(maps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv, sht_threads:int=0):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            maps: input polarisation maps
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert ninv_filt.lmax_sol == ninv_filt.lmax_len, (ninv_filt.lmax_sol, ninv_filt.lmax_len)  # not implemented wo lensing
    assert ninv_filt.mmax_sol == ninv_filt.mmax_len, (ninv_filt.mmax_sol, ninv_filt.mmax_len)
    qumap = np.copy(maps)
    ninv_filt.apply_map(qumap)
    eblm = ninv_filt.ninv_geom.adjoint_synthesis(np.array(qumap), 2,  ninv_filt.lmax_len,  ninv_filt.mmax_len, sht_threads, apply_weights=False)
    lmax_tr = ninv_filt.lmax_len
    almxfl(eblm[0], ninv_filt.transf_elm * (s_cls['ee'][:lmax_tr+1] > 0.), ninv_filt.mmax_len, inplace=True)
    almxfl(eblm[1], ninv_filt.transf_blm * (s_cls['bb'][:lmax_tr+1] > 0.), ninv_filt.mmax_len, inplace=True)
    return eblm

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass

class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
        ninv_fel, ninv_fbl = ninv_filt.get_febl()  # (N_lev * transf) ** 2 basically
        lmax_sol = ninv_filt.lmax_sol
        flmat = {}
        for fl, clk in zip([ninv_fel, ninv_fbl], ['ee', 'bb']):
            assert len(s_cls[clk]) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls[clk]))
            if len(fl) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
                log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(fl)-1, lmax_sol))
                assert np.all(fl > 0)
                spl_sq = spl(np.arange(len(ninv_fel), dtype=float), np.log(fl), k=2, ext='extrapolate')
                flmat[clk] = cli(s_cls[clk][:lmax_sol + 1]) + np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
            else:
                flmat[clk] = cli(s_cls[clk][:lmax_sol + 1]) + fl

        self.flmat = {k: cli(flmat[k]) * (s_cls[k][:lmax_sol +1] > 0.) for k in ['ee', 'bb']}
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, eblm):
        return self.calc(eblm)

    def calc(self, eblm):
        assert Alm.getsize(self.lmax, self.mmax) == eblm[0].size, (self.lmax, self.mmax, Alm.getlmax(eblm[0].size, self.mmax))
        assert Alm.getsize(self.lmax, self.mmax) == eblm[1].size, (self.lmax, self.mmax, Alm.getlmax(eblm[1].size, self.mmax))
        ret = np.copy(eblm)
        almxfl(ret[0], self.flmat['ee'], self.mmax, True)
        almxfl(ret[1], self.flmat['bb'], self.mmax, True)
        return ret

class dot_op:
    def __init__(self, lmax:int, mmax:int or None):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


        """
        if mmax is None or mmax < 0: mmax = lmax
        self.lmax = lmax
        self.mmax = min(mmax, lmax)

    def __call__(self, eblm1, eblm2):
        assert eblm1[0].size == Alm.getsize(self.lmax, self.mmax), (eblm1[0].size, Alm.getsize(self.lmax, self.mmax))
        assert eblm2[0].size == Alm.getsize(self.lmax, self.mmax), (eblm2[0].size, Alm.getsize(self.lmax, self.mmax))
        assert eblm1[1].size == Alm.getsize(self.lmax, self.mmax), (eblm1[1].size, Alm.getsize(self.lmax, self.mmax))
        assert eblm2[1].size == Alm.getsize(self.lmax, self.mmax), (eblm2[1].size, Alm.getsize(self.lmax, self.mmax))
        ret  = np.sum(alm2cl(eblm1[0], eblm2[0], self.lmax, self.mmax, None) * (2 * np.arange(self.lmax + 1) + 1))
        ret += np.sum(alm2cl(eblm1[1], eblm2[1], self.lmax, self.mmax, None) * (2 * np.arange(self.lmax + 1) + 1))
        return ret

class fwd_op:
    """Forward operation for polarization-only, no primordial B power cg filter


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
        self.icls = {'ee':cli(s_cls['ee']), 'bb':cli(s_cls['bb'])}
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'iclee': clhash(self.icls['ee']),'iclbb': clhash(self.icls['bb']),
                'n_inv_filt': self.ninv_filt.hashdict()}

    def __call__(self, eblm):
        return self.calc(eblm)

    def calc(self, eblm):
        nlm = np.copy(eblm)
        self.ninv_filt.apply_alm(nlm)
        nlm[0] = almxfl(nlm[0] + almxfl(eblm[0], self.icls['ee'], self.mmax_sol, False), self.icls['ee'] > 0., self.mmax_sol, False)
        nlm[1] = almxfl(nlm[1] + almxfl(eblm[1], self.icls['bb'], self.mmax_sol, False), self.icls['bb'] > 0., self.mmax_sol, False)
        return nlm
