"""This module contains the definitions for a Wiener-filter of CMB polarization.

    In this module, the data is distributed on arbitrary pixels, in contrast to the standard ones where
    the data is distributed on isolatitude rings.

"""

import numpy as np
import psutil
from scipy.interpolate import UnivariateSpline as spl

from lenspyx.remapping import utils_geom, deflection_029
from delensalot.utils import timer, cli, clhash, read_map
from lenspyx.utils_hp import almxfl, Alm, alm2cl, synalms
from delensalot.core.opfilt import bmodes_ninv as bni
from duccjc.sht import synthesis_general_cap as syng, adjoint_synthesis_general_cap as syng_adj # FIXME
from ducc0.misc import get_deflected_angles, lensing_rotate
from copy import deepcopy

rtype = {np.dtype(np.complex128):np.dtype(np.float64), np.dtype(np.complex64):np.dtype(np.float32)}
ctype = {rtype[ctyp]:ctyp for ctyp in rtype}

class operator(object):
    def __init__(self, *args, **kwargs):
        pass
    def apply_inplace(self, vec_a, vec_b):
        assert 0, 'implement this'
    def apply_adjoin_inplace(self, vec_a, vec_b):
        assert 0, 'implement this'
    def eval_speed(self):
        assert 0, 'implement this'

class transfer_harmonic_simple(operator):
    def __init__(self, bl:np.ndarray[np.float64], lmmax:tuple[int]):
        """Simple transfer function operator in harmonic space


        """
        super(transfer_harmonic_simple).__init__(self)
        assert bl.ndim == 2
        self.ncomp = bl.shape[0]
        self.bl = bl
        self.mmax = lmmax[1]
        self.alm_size = Alm.getsize(*lmmax)

    def apply_inplace(self, alms:np.ndarray[np.complex128]):
        assert alms.shape == (self.ncomp, self.alm_size)
        for bl, alm in zip(self.bl, alms):
            almxfl(alm, bl, self.mmax, True)

class transfer_BMD(operator):
    def __init__(self, thtcap, dgclm:np.ndarray, lmmax_len:tuple[int, int], lmmax_unl:tuple[int, int], Lmmax:tuple[int, int],
                 nthreads=0, epsilon=1e-7, verbose=False):
        """Polarized transfer operator with lensing


                This implements the operator Y M D, where D is lensing, M some masking window, and B a beam

                The input are skyalms, the output are lensed skyalms multiplied by bl

                #TODO: maybe the bl's should be kicked out, so that multifrequencies can then be used

        """
        super(transfer_BMD).__init__(self)


        self._loc = None
        self._dl_7 = 20

        self.lmmax_len = lmmax_len
        self.lmmax_unl = lmmax_unl
        self.Lmmax = Lmmax

        self.thtcap = thtcap

        self.geom = self._prepare_geom()
        self.dgclm = dgclm

        self.nthreads = nthreads or psutil.cpu_count(logical=False)

        self._loc   = None
        self._gamma = None
        self._mapsc = None
        self._mapsr = None

        lmax_unl, mmax_unl = lmmax_unl
        lensing_syng_params = {'epsilon':epsilon,
                       'thtcap':thtcap,
                       'eps_apo': np.sqrt(self._dl_7 / lmax_unl * np.pi / thtcap),
                        'spin':2, 'lmax':lmax_unl, 'mmax':mmax_unl,
                        'nthreads':self.nthreads, 'mode':'STANDARD', 'verbose':verbose}
        adj_lensing_syng_params = lensing_syng_params.copy()
        adj_lensing_syng_params['lmax'] = lmax_unl  # FIXME
        adj_lensing_syng_params['mmax'] = mmax_unl

        adj_syn_params = {'spin':2, 'lmax':self.lmmax_len[0],
                          'mmax':self.lmmax_len[1], 'nthreads':self.nthreads, 'apply_weights':False}
        syn_params = {'spin':2, 'lmax':self.lmmax_len[0],
                          'mmax':self.lmmax_len[1], 'nthreads':self.nthreads}


        self.lensing_syng_params = lensing_syng_params
        self.adj_lensing_syng_params = adj_lensing_syng_params
        self.adj_syn_params = adj_syn_params
        self.syn_params = syn_params

        self.rtype = np.float64 if epsilon < 1e-7 else np.float32

    def _prepare_geom(self):
        """Prepares geometry that will handle the masking and lensing operations"""
        lmax = self.lmmax_unl[0]
        eps = np.sqrt(self._dl_7 / lmax)
        dlmax = lmax * eps
        thtmax = min(self.thtcap * (1 + eps), np.pi)
        bgeom = utils_geom.Geom.get_thingauss_geometry(int(lmax + dlmax) + 2, 3, False).restrict(0, thtmax * 1.0001, False, True)

        # Build weights for the geometry (apodized window, standard bump)
        x = (bgeom.theta - self.thtcap) / (thtmax - self.thtcap)
        i = np.where((x < 1) & (x > 0))[0]
        fx = np.exp(-2 / x[i])
        f1x = np.exp(-2 / (1 - x[i]))
        wx = f1x / (f1x + fx)
        bgeom.weight[i] *= wx
        bgeom.weight[np.where(x >= 1.)] *= 0
        return bgeom

    def _prepare_ptg(self):
        """Build pointing coordinates for synthesis general


        """
        if self._loc is None:
            Lmax, Mmax = self.Lmmax
            tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
            d1 = self.geom.synthesis(self.dgclm, 1, Lmax, Mmax, self.nthreads)
            tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                  calc_rotation=True, nthreads=self.nthreads)
            self._gamma = tht_phip_gamma[:, 2]
            self._loc   = tht_phip_gamma[:, 0:2]

    def _allocate(self):
        if self._mapsc is None:
            mapsc = np.empty((self.geom.npix(),), dtype=ctype[self.rtype])
            mapsr = mapsc.view(self.rtype).reshape((self.geom.npix(), 2)).T
            self._mapsc = mapsc
            self._mapsr = mapsr


    def deallocate(self):
        self._mapsc = None
        self._mapsr = None

    def apply_inplace(self, alms_unl, alms_len):
        assert alms_unl.ndim == 2 and alms_unl.shape[1] == Alm.getsize(*self.lmmax_unl)
        assert alms_len.ndim == 2 and alms_len.shape[1] == Alm.getsize(*self.lmmax_len)
        assert not self.adj_lensing_syng_params.get('apply_weights')

        self._prepare_ptg()
        self._allocate()
        self.lensing_syng_params['mode'] = 'GRAD_ONLY' if alms_unl.shape[0] == 1 else 'STANDARD'
        syng(alm=alms_unl, map=self._mapsr, loc=self._loc, **self.lensing_syng_params)
        lensing_rotate(self._mapsc, self._gamma, 2, self.nthreads)
        for of, w, npi in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
            self._mapsr[:, of:of + npi] *= w
        self.geom.adjoint_synthesis(m=self._mapsr, alm=alms_len, **self.adj_syn_params)

    def apply_adjoint_inplace(self, alms_unl, alms_len):
        assert alms_unl.ndim == 2 and alms_unl.shape[1] == Alm.getsize(*self.lmmax_unl)
        assert alms_len.ndim == 2 and alms_len.shape[1] == Alm.getsize(*self.lmmax_len)
        assert not self.adj_lensing_syng_params.get('apply_weights')

        self._prepare_ptg()
        self._allocate()
        self.geom.synthesis(m=self._mapsr, alm=alms_len, **self.syn_params)
        for of, w, npi in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
            self._mapsr[:, of:of + npi] *= w
        lensing_rotate(self._mapsc, -self._gamma, 2, self.nthreads)
        self.adj_lensing_syng_params['mode'] = 'GRAD_ONLY' if alms_unl.shape[0] == 1 else 'STANDARD'
        syng_adj(alm=alms_unl, map=self._mapsr, loc=self._loc, **self.adj_lensing_syng_params)

class alm_filter_ninv(object):
    def __init__(self, loc:np.ndarray, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, dgclm:np.ndarray[complex] or None=None,
                 transf_b:np.ndarray or None=None, epsilon=1e-7, nlevp_iso=None,
                 tpl:bni.template_dense or None=None, verbose=False, maskbeam=False):
        r"""CMB inverse-variance and Wiener filtering instance, to use for cg-inversion

            Args:
                loc: co-latitude and longitude of the pixels. Must be something readable to give a (npix, 2) array
                ninv: list of inverse-pixel noise variance maps (itself can be (a list of) string, or array, or ...)
                transf: CMB transfer function (assumed to be the same in E and B)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                epsilon: accuracy parameter of ducc synthesis_general and its adjoint
                sht_threads: number of threads for lenspyx SHTs
                verbose: some printout if set, defaults to False
                nlevp_iso: some reference value of the isotropic white noise level in uK-amin (preconditioner etc)
                maskbeam: the data model includes a area-masking operation prior to the beam convolution

            #FIXME: make an operator class and abstract away the beam at least

        """
        assert nlevp_iso is not None, 'need nlevp_iso, or come up with an idea of the pixel size given the loc'
        transf_elm = transf
        transf_blm = transf_b if transf_b is not None else transf
        assert transf_blm.size == transf_elm.size, 'check if not same size OK'


        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = max(len(transf), len(transf_blm)) - 1

        self.n_inv = ninv
        self.transf_elm = transf_elm
        self.transf_blm = transf_blm

        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)

        self.lmax_sol = lmax_unl
        self.mmax_sol = min(lmax_unl, mmax_unl)
        self.ncomp_sol = 2  # here E and B alms
        self.ncomp_dat = 2  # here Q and U

        self.sht_threads = sht_threads
        self.verbose=verbose

        self._nlevp = nlevp_iso
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed
        self.loc = loc

        # Build some syng_cap params
        # When synthesis_general_cap will be stable we will adapt this
        rloc = read_map(loc)
        npix = rloc.shape[0]
        thtcap = min(np.max(rloc[:, 0]) * 1.0001, np.pi)

        # syng and adj_syng helpers, since not API not stable just now
        dl_7 = 19 # might want to tweak this
        dl = int(np.round(dl_7 * ((- np.log10(epsilon) + 1) / (7 + 1)) ** 2))
        # all syng params except of loc
        syng_params = {'epsilon':epsilon,
                       'thtcap':thtcap,
                       'eps_apo': np.sqrt(dl / lmax_unl * np.pi / thtcap),
                        'spin':2, 'lmax':lmax_len, 'mmax':mmax_len,
                        'nthreads':sht_threads, 'mode':'STANDARD', 'verbose':verbose}
        adj_syng_params = syng_params.copy()
        adj_syng_params['lmax'] = lmax_unl
        adj_syng_params['mmax'] = mmax_unl

        if verbose:
            print("general QE p filter setup with thtcap = %.2f deg"%(thtcap/np.pi*180))
            print('eps apo %.3f'%syng_params.get('eps_apo', 0))
        self.syng_params = syng_params
        self.adj_syng_params = adj_syng_params
        self._dl_7 = dl_7
        self.thtcap = thtcap

        self.npix = npix
        self._qu = None

        if maskbeam:
            # The beam operator includes a masking operation
            self.maskbeam = True
            self._beamgeom = self._get_sky_geom(lmax_unl, weighted=True)

        else:
            self._beamgeom = None
            self._beamw = None
            self.maskbeam = False

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.transf_elm),
                'unalm':(self.lmax_sol, self.mmax_sol),
                'lenalm':(self.lmax_len, self.mmax_len) }

    def _get_sky_geom(self, lmax, weighted=True):
        """Returns a suitable weighted geometry object for the masked beam operation


        """
        if self.maskbeam:
            # Here we know that the lensing gradients are exactly zero outside of the beam window, se we cut out the geometry
            eps = np.sqrt(self._dl_7 / lmax)
            dlmax = lmax * eps
            thtmax = min(self.thtcap * (1 + eps), np.pi)
            bgeom = utils_geom.Geom.get_thingauss_geometry(int(lmax + dlmax) + 2, 3, False).restrict(0, thtmax * 1.0001, False, True)
            # now build weights using f(1-x) / (f(1-x) + f(x))
            if weighted:
                x = (bgeom.theta - self.thtcap) / (thtmax - self.thtcap)
                i = np.where((x < 1) & (x > 0))[0]
                fx = np.exp(-2 / x[i])
                f1x = np.exp(-2 / (1 - x[i]))
                wx = f1x / (f1x + fx)
                bgeom.weight[i] *= wx
                bgeom.weight[np.where(x >= 1.)] *= 0
            if self.verbose:
                print("using beam geometry with dlmax %d" % int(dlmax))
                print("%s non-zero rings" % (len(bgeom.theta)))
            return bgeom
        else:
            return utils_geom.Geom.get_thingauss_geometry(lmax, 3, False)

    def _test_syng_accuracy(self, dtype=np.float64):
        if 'thtcap' in self.syng_params:
            eblm = synalms({'ee':np.ones(self.lmax_sol+1), 'bb':np.ones(self.lmax_sol+1)}, self.lmax_sol, self.mmax_sol, rlm_dtype=dtype)
            self._allocate_maps(rtype[eblm.dtype])
            loc = read_map(self.loc)
            syng(alm=eblm, map=self._qu, loc=loc, **self.syng_params)
            syng_params = deepcopy(self.syng_params)
            syng_params['thtcap'] = np.pi
            ref = syng(alm=eblm, loc=loc, **syng_params)
            norm = np.sqrt(np.mean(ref ** 2))
            print('syng max  rel dev %.2e'%(np.sqrt(np.max( (ref-self._qu) ** 2) )/ norm))
            print('syng mean rel dev %.2e'%(np.sqrt(np.mean( (ref-self._qu) ** 2) )/ norm))

    def _allocate_maps(self, dtype):
        """This allocate the positions-space maps for cg iterations


        """
        if self._qu is None:
            self._qu = np.empty((self.ncomp_dat, self.npix), dtype=dtype)
        else:
            assert self._qu.dtype == dtype, 'not sure how thats possible, find some way to treat the types properly'

    def deallocate(self):
        """Just cleans some stuff if relevant by deallocating potentially large arrays


        """
        self._qu = None

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.n_inv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return ret

    def get_febl(self):
        assert self._nlevp is not None, 'need to implement some idea of pixel size from the locs if not specified'
        if self._nlevp is None:
            if len(self.n_inv) == 1:
                ni = read_map(self.n_inv[0])
                nlev_febl =  1. / np.sqrt(np.sum(ni * ni) / np.sum(ni)) * 180 * 60 / np.pi  # Ni-weigted noise level
                # Hmm need some idea of pixel size here...
            elif len(self.n_inv) == 3:
                assert 0, 'implement this'
            else:
                assert 0
            self._nlevp = nlev_febl
        fel = self.transf_elm ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        fbl = self.transf_blm ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return fel, fbl

    def apply_beam(self, eblm:np.ndarray, qumap:np.ndarray, loc=None):
        """Action of beam operator. Here defined from harmonic space to data pixel space

            Note:
                This can modify eblm

        """
        if self.maskbeam:
            # FIXME: with lensing just replace here synthesis with synthesis_general_cap
            qu = self._beamgeom.synthesis(gclm=eblm, spin=2, lmax=self.lmax_sol, mmax=self.mmax_sol, nthreads=self.sht_threads)
            self._beamgeom.adjoint_synthesis(m=qu, alm=eblm, spin=2, lmax=self.lmax_len, mmax=self.mmax_len, nthreads=self.sht_threads, apply_weights=True)
        if loc is None:
            loc = read_map(self.loc)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        syng(alm=eblm, map=qumap, loc=loc, **self.syng_params)

    def apply_adjoint_beam(self, eblm:np.ndarray, qumap:np.ndarray, loc=None):
        """Adjoint operation to apply_beam.

            Note:
                The adjoint takes as input a map in data pixel space and produces a map in harmonic space

        """
        if loc is None:
            loc = read_map(self.loc)
        syng_adj(map=qumap,  loc=loc, alm=eblm, **self.adj_syng_params)
        almxfl(eblm[0], self.transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, inplace=True)
        if self.maskbeam:
            qu = self._beamgeom.synthesis(gclm=eblm, spin=2, lmax=self.lmax_len, mmax=self.mmax_len, nthreads=self.sht_threads)
            self._beamgeom.adjoint_synthesis(m=qu, alm=eblm, spin=2, lmax=self.lmax_sol, mmax=self.mmax_sol, nthreads=self.sht_threads, apply_weights=True)

    def apply_alm(self, eblm:np.ndarray):
        """Applies operator B^T N^{-1} B

        """
        assert self.lmax_sol == self.lmax_len, (self.lmax_sol, self.lmax_len) # not implemented wo lensing
        assert self.mmax_sol == self.mmax_len, (self.mmax_sol, self.mmax_len)
        assert  Alm.getlmax(eblm[0].size, self.mmax_sol) == self.lmax_sol, ( Alm.getlmax(eblm[0].size, self.mmax_sol), self.lmax_sol)
        tim = timer(True, prefix='opfilt_pp')
        loc = read_map(self.loc)
        self._allocate_maps(dtype=rtype[eblm.dtype])
        tim.add('applyalm init')
        self.apply_beam(eblm, qumap=self._qu, loc=loc)
        tim.add('beam')
        self.apply_map(self._qu)
        tim.add('Ni')
        self.apply_adjoint_beam(eblm, qumap=self._qu, loc=loc)
        tim.add('beam (adjoint)')
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

    def synthesize(self, cls:dict):
        """Generate a simulated data map matching the noise model

            Note:
                :math:`B X^{\text{unl}} + n`

        """
        from lenspyx.utils_hp import rng # not sure this is the right thing to do
        clseb = {spec: cls.get(spec, np.array([0.0])) for spec in ['ee', 'bb', 'eb', 'be']}
        eblm = synalms(clseb, lmax=self.lmax_len, mmax=self.mmax_len)
        qu = np.empty((self.ncomp_dat, self.npix), dtype=rtype[eblm.dtype])
        self.apply_beam(eblm, qumap=qu, loc=read_map(self.loc))
        assert len(self.n_inv) == 1, 'not implemented, fix next line'
        qu += rng.standard_normal(qu.shape, dtype=qu.dtype) * np.sqrt(cli(read_map(self.n_inv[0])))
        return qu

    def get_qlms(self, qudat: np.ndarray or list, eblm_wf: np.ndarray, lmax_qlm:int, mmax_qlm:int,
                 norm:str='k', eb_only=False):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                eblm_wf: Wiener-filtered CMB maps (alm arrays)
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs
                lmax_qlm: maximum multipole of output
                mmax_qlm: maximum m of lm output
                norm: normalization of the output (whether kappa-like ('k'), or potential-like...)
                eb_only: if True, only the EB estimator is returned

        """
        assert len(qudat) == self.ncomp_dat and len(eblm_wf) == 2
        assert norm in ['k', 'kappa', 'kappa-like', 'p', 'phi', 'phi-like'], 'implement this (easy)'
        qgeom = self._get_sky_geom((2 * self.lmax_sol + lmax_qlm // 2) + 1, weighted=False)
        resmap_c = np.empty((qgeom.npix(),), dtype=eblm_wf.dtype)
        resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        self._get_irespmap(qudat, eblm_wf, qgeom, map_out=resmap_r, eb_only=eb_only)  # inplace onto resmap_c and resmap_r
        if not eb_only:
            eblm_wf_g = eblm_wf
        else:
            eblm_wf_g = np.zeros_like(eblm_wf)
            eblm_wf_g[0, :] = eblm_wf[0]  # For gradient leg
        gcs_r = self._get_gpmap(eblm_wf_g, 3, qgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = self._get_gpmap(eblm_wf_g, 1, qgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r
        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        gc = qgeom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, self.sht_threads)
        del gc_r, gc_c
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        if norm[0] == 'k':
            fl *= cli(np.arange(lmax_qlm + 1) * np.arange(1, lmax_qlm + 2, dtype=float) * 0.5)
        almxfl(gc[0], fl, mmax_qlm, True)
        almxfl(gc[1], fl, mmax_qlm, True)
        return gc

    def _get_gpmap(self, eblm_wf:np.ndarray or list, spin:int, qgeom:utils_geom.Geom):
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
        eblm = np.copy(eblm_wf)
        almxfl(eblm[0], fl, self.mmax_sol, True)
        almxfl(eblm[1], fl, self.mmax_sol, False)
        valuesc = np.empty((qgeom.npix(),), dtype=eblm_wf.dtype)
        values = valuesc.view(rtype[valuesc.dtype]).reshape((qgeom.npix(), 2)).T
        return qgeom.synthesis(eblm, spin, lmax, self.mmax_sol, self.sht_threads, map=values)

    def _get_irespmap(self, qu_dat:np.ndarray, eblm_wf:np.ndarray or list, qgeom:utils_geom.Geom,
                      map_out=None, eb_only=False):
        """Builds inverse variance weighted map to feed into the QE

                :math:`B^t N^{-1}(X^{\rm dat} - B X^{WF})`


        """
        assert len(qu_dat) == self.ncomp_dat and len(eblm_wf) == 2, (len(eblm_wf), len(qu_dat))
        ebwf = np.copy(eblm_wf)
        loc = read_map(self.loc)
        self._allocate_maps(dtype=qu_dat.dtype)
        self.apply_beam(eblm=ebwf, qumap=self._qu, loc=loc)
        self._qu -= qu_dat
        self.apply_map(self._qu)
        self.apply_adjoint_beam(eblm=ebwf, qumap=self._qu, loc=loc)
        ebwf *= -1 # FIXME. I think I am including a pair of adjoint- and forward synthesis too much here in the beam-masked case
        if eb_only:
            ebwf[0, :] = 0
        return qgeom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads, map=map_out)

pre_op_dense = None # not implemented

def calc_prep(maps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv):
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
    eblm = np.empty((ninv_filt.ncomp_sol, Alm.getsize(ninv_filt.lmax_sol, ninv_filt.mmax_sol)), dtype=ctype[qumap.dtype])
    ninv_filt.apply_adjoint_beam(eblm=eblm, qumap=qumap, loc=read_map(ninv_filt.loc))
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
                print("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(fl)-1, lmax_sol))
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

        icls = {'ee':cli(s_cls['ee']), 'bb':cli(s_cls['bb'])}
        assert np.all(ninv_filt.transf_elm[np.where(icls['ee'][:ninv_filt.lmax_len] == 0.)] == 0.), 'inconsistent inputs'
        assert np.all(ninv_filt.transf_blm[np.where(icls['bb'][:ninv_filt.lmax_len] == 0.)] == 0.), 'inconsistent inputs'
        self.icls = icls
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
        nlm[0] += almxfl(eblm[0], self.icls['ee'], self.mmax_sol, False)
        nlm[1] += almxfl(eblm[1], self.icls['bb'], self.mmax_sol, False)
        return nlm
