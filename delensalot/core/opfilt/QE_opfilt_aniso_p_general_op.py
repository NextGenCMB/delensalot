"""This module contains the definitions for a Wiener-filter of CMB polarization.

    In this module, the data is distributed on arbitrary pixels, in contrast to the standard ones where
    the data is distributed on isolatitude rings.

"""

import numpy as np
import psutil
from scipy.interpolate import UnivariateSpline as spl

from lenspyx.remapping import utils_geom, deflection_029
from lenspyx.lensing import dlm2angles
from delensalot.utils import timer, cli, clhash, read_map
from lenspyx.utils_hp import almxfl, Alm, alm2cl, synalms
from delensalot.core.opfilt import bmodes_ninv as bni
from duccjc.sht import synthesis_general_cap as syng, adjoint_synthesis_general_cap as syng_adj # FIXME
from ducc0.misc import lensing_rotate
from copy import deepcopy

rtype = {np.dtype(np.complex128):np.dtype(np.float64), np.dtype(np.complex64):np.dtype(np.float32)}
ctype = {rtype[ctyp]:ctyp for ctyp in rtype}

class operator(object):
    def __init__(self, *args, **kwargs):
        pass
    def apply_inplace(self, vec_a, vec_b):
        assert 0, 'not implemented -- subclass this'

    def apply_adjoint_inplace(self, vec_a, vec_b):
        assert 0, 'not implemented -- subclass this'

    def apply_derivative(self, param, vec_a, vec_b):
        assert 0, 'I cant derive w.r.t. anything -- subclass this'

    def apply_adjoint_derivative(self, param, vec_a, vec_b):
        assert 0, 'I cant derive w.r.t. anything -- subclass this'

    def eval_speed(self, vec_a, vec_b):
        """Evaluate the execution speed of the operator and its adjoint


        """
        # TODO: this we can actually test here
        pass

    def test_adjointness(self):
        # TODO: this we can actually test here
        pass

    def allocate(self):
        """Intended usage is necessary preparations for e.g., cg-inversion



        """
        assert 0, 'implement this'

    def deallocate(self):
        """Clean-up all potentially allocated array to avoid unnecessary mess

            To be used e.g. at the end of a Wiener filter iterative search

        """
        assert 0, 'implement this'

class Id(operator):
    def __init__(self):
        super(Id).__init__(self)
    def allocate(self):
        pass
    def deallocate(self):
        pass
    def apply_inplace(self, vec_a, vec_b):
        pass
    def apply_adjoint_inplace(self, vec_a, vec_b):
        pass



class BtNiB_light(operator):
    def __init__(self, spin, lmmax:tuple[int, int], bls:list[np.ndarray], s2is:list[float], Ni, geom:utils_geom.Geom,
                 nthreads=0, r_dtype=np.float64):
        """Inverse noise matrix operator, combining beam and inverse noise variance maps for a number of frequency channels

            Args:
                spin: spin-weight of the input components
                lmmax: alm array layout
                bls: 1d beam functions
                s2is: one number per frequency. The inverse variance map is s2i * Ni, with the same Ni in each channel
                Ni: 'relative hit' map
                geom: isolatitude pixelization of the sphere
                nthreads: number of threads for SHTs
                r_dtype: precision of the computation

            Note:
                This performs (a channel, i pixel, lm harmonics)
                    :math:`\sum_{a, i, lm} b_l'Y^{\dagger}_{l'm'}(\hn_i) N^{-1}_i /s^2_a Y_{lm}(\hat n_i) b^a_l a_{lm}`

            #FIXME: must extend this to synthesis_general case

        """
        assert len(bls) == len(s2is)
        super(BtNiB_light).__init__(self)
        nthreads = nthreads or psutil.cpu_count(logical=False)

        self.nchannel = len(bls) # Number of frequency channel
        self.ncomp = 1 + (spin > 0)
        self.spin = spin
        self.bls = bls
        self.Ni = Ni
        self.geom = geom
        self.npix = Ni.size
        self.sis = np.sqrt(s2is)

        self.nthreads=nthreads

        self.lmmax = lmmax

        self.dtype =r_dtype
        self._alms = None
        self._maps = None

    def _synthesis(self, alms, maps):
        if isinstance(self.geom, utils_geom.Geom):
            self.geom.synthesis(alms, self.spin, self.lmmax[0], self.lmmax[1], self.nthreads, map=maps)
        elif isinstance(self.geom, np.ndarray):
            assert 0, 'insert synthesis_general'

    def _adjoint_synthesis(self, alms, maps):
        if isinstance(self.geom, utils_geom.Geom):
            self.geom.adjoint_synthesis(maps, self.spin, self.lmmax[0], self.lmmax[1], self.nthreads, alm=alms,
                                        apply_weights=False)
        elif isinstance(self.geom, np.ndarray):
            assert 0, 'insert adjoint synthesis_general'

    def apply_inplace(self, alm_in, alm_ou):
        assert alm_in.shape == (self.ncomp, Alm.getsize(*self.lmmax))
        assert alm_ou.shape == (self.ncomp, Alm.getsize(*self.lmmax))
        self.allocate()
        for i, (bl, si) in enumerate(zip(self.bls, self.sis)):
            self._alms[:] = alm_in
            for alm in self._alms:
                almxfl(alm, bl * si, self.lmmax[1], True)
            self._synthesis(self._alms, self._maps)
            self._maps *= self.Ni
            self._adjoint_synthesis(alm_ou if i == 0 else self._alms, self._maps)
            if i == 0:
                for alm in alm_ou:
                    almxfl(alm, bl * si, self.lmmax[1], True)
            else:
                for j in range(self.ncomp):
                    alm_ou[j] += almxfl(self._alms[j], bl * si, self.lmmax[1], False)


    def apply_adjoint_inplace(self, alm_in, alm_ou):
        self.apply_inplace(alm_ou, alm_in) # self-adjoint. Just reverse the inputs

    def allocate(self):
        self._alms = np.empty((self.ncomp, Alm.getsize(*self.lmmax)), dtype=self.dtype)
        self._maps = np.empty((self.ncomp, self.npix), dtype=ctype[self.dtype])

    def deallocate(self):
        self._alms = None
        self._maps = None


class MD(operator):
    def __init__(self, thtcap, spin:int, sht_mode:str, lmmax_len:tuple[int, int], lmmax_unl:tuple[int, int],
                 dgclm: np.ndarray, Lmmax: tuple[int, int], nthreads=0, epsilon=1e-7, verbose=False):
        """Lensing deflection operator

                This implements the operator D, or M D, where D is lensing, M some masking window,
                with eventual projection onto harmonic components.

                The input is a (set of) unlensed alms, and output is a (set of) lensed alms

            Args:
                thtcap:
                spin, sht_mode: spin of the transform and its DUCC type ('mode' DUCC parameter)
                    e.g. 2, 'GRAD_ONLY' for E-only reconstruction, or 0, 'STANDARD' for temperature
                lmmax_unl: lmax and mmax of the input alm layout
                lmmax_len: lmax and mmax of the output alm layout
                dgclm: (1 or 2, alm_size) array, deflection field harmonic components
                Lmmax: lmax and mmax of the delfection field components
                epsilon: desirec precision of the remapping operations
                nthreads: number of threads assigned to (adjoint_)synthesis_general
                verbose: some printouts if set



                #TODO: what to do with dgclm ? we dont need it anymore after computing the pointing
        """
        super().__init__(self)
        assert int(spin) >= 0, spin
        nthreads = nthreads or psutil.cpu_count(logical=False)

        # FIXME
        _dl_7 = 20

        self.lmmax_len = lmmax_len
        self.lmmax_unl = lmmax_unl
        self.Lmmax = Lmmax

        self.thtcap = thtcap
        self.spin = int(spin)

        self.geom = self._prepare_geom(_dl_7=_dl_7)
        self.dgclm = np.atleast_2d(dgclm)
        # FIXME:We dont need this once locs are calculated, just do it on instantiation ? or give locs as input ?


        self.nthreads = nthreads

        # These are potentially large arrays that might be instantiated later (or not)
        self._loc   = None  # tht and phis coordinates -- deflected angles
        self._gamma = None  # phase with which to rotate non-zero spin fields
        self._mapsc = None  # calculation must go through an intermediate map.
        self._mapsr = None  # For non-zero spins we use a complex array with a real view, or just a real array for spin-0

        lmax_unl, mmax_unl = lmmax_unl
        lensing_syng_params = {'epsilon':epsilon,
                       'thtcap':thtcap,
                       'eps_apo': np.sqrt(_dl_7 / lmax_unl * np.pi / thtcap),
                        'spin':spin, 'lmax':lmax_unl, 'mmax':mmax_unl,
                        'nthreads':nthreads, 'mode':sht_mode, 'verbose':verbose}
        adj_lensing_syng_params = lensing_syng_params.copy()
        adj_lensing_syng_params['lmax'] = lmax_unl
        adj_lensing_syng_params['mmax'] = mmax_unl

        adj_syn_params = {'spin':spin, 'lmax':lmmax_len[0],
                          'mmax':lmmax_len[1], 'nthreads':nthreads, 'apply_weights':False}
        syn_params = {'spin':spin, 'lmax':lmmax_len[0],
                          'mmax':lmmax_len[1], 'nthreads':nthreads}


        self.lensing_syng_params = lensing_syng_params
        self.adj_lensing_syng_params = adj_lensing_syng_params

        self.adj_syn_params = adj_syn_params
        self.syn_params = syn_params

        self.rtype = np.dtype(np.float64) if epsilon <= 1e-7 else np.dtype(np.float32)

    def _prepare_geom(self, _dl_7=20):
        """Prepares geometry that will handle the lensing and windowing operations


        """
        lmax = self.lmmax_unl[0]
        eps = np.sqrt(_dl_7 / lmax)
        dlmax = lmax * eps
        thtmax = min(self.thtcap * (1 + eps), np.pi)
        bgeom = utils_geom.Geom.get_thingauss_geometry(int(lmax + dlmax) + 2, self.spin+1, False).restrict(0, thtmax * 1.0001, False, True)

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
        """Build pointing coordinates for synthesis general and its adjoint


        """
        if self._loc is None:
            angles = dlm2angles(self.dgclm, self.geom, self.Lmmax[1], self.nthreads, self.spin>0)
            self._loc  = angles[:, 0:2]
            if self.spin:
                self._gamma = angles[:, 2]

    def _allocate(self):
        """Allocate necessary intermediate arrays, in order to avoid re-allocation at every time in cg-process etc


        """
        if self._mapsr is None:
            if self.spin:
                mapsc = np.empty((self.geom.npix(),), dtype=ctype[self.rtype])
                mapsr = mapsc.view(self.rtype).reshape((self.geom.npix(), 2)).T
            else: # Only one component -- no need for mapsc
                mapsr = np.empty((self.geom.npix(),), dtype=self.rtype)
                mapsc = None
            self._mapsc = mapsc
            self._mapsr = mapsr


    def deallocate(self):
        """Clean-up all potentially allocated array to avoid unnecessary mess

            To be used e.g. at the end of a Wiener filter iterative search

        """
        self._mapsc = None
        self._mapsr = None

    def apply_inplace(self, alms_unl, alms_len):
        assert alms_unl.ndim == 2 and alms_unl.shape[1] == Alm.getsize(*self.lmmax_unl)
        assert alms_len.ndim == 2 and alms_len.shape[1] == Alm.getsize(*self.lmmax_len)
        assert not self.adj_lensing_syng_params.get('apply_weights')

        self._prepare_ptg()
        self._allocate() # This to avoid creating new large arrays all the time in cg-searches...
        syng(alm=alms_unl, map=self._mapsr, loc=self._loc, **self.lensing_syng_params)
        if self.spin:
            lensing_rotate(self._mapsc, self._gamma, self.spin, self.nthreads)
        for of, w, npi in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
            self._mapsr[:, of:of + npi] *= w
        self.geom.adjoint_synthesis(m=self._mapsr, alm=alms_len, **self.adj_syn_params)

    def apply_adjoint_inplace(self, alms_unl, alms_len):
        assert alms_unl.ndim == 2 and alms_unl.shape[1] == Alm.getsize(*self.lmmax_unl)
        assert alms_len.ndim == 2 and alms_len.shape[1] == Alm.getsize(*self.lmmax_len)
        assert not self.adj_lensing_syng_params.get('apply_weights', False)

        self._prepare_ptg()
        self._allocate()
        self.geom.synthesis(alms_len, map=self._mapsr, **self.syn_params)
        for of, w, npi in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
            self._mapsr[:, of:of + npi] *= w
        if self.spin:
            lensing_rotate(self._mapsc, -self._gamma, self.spin, self.nthreads)
        syng_adj(alm=alms_unl, map=self._mapsr, loc=self._loc, **self.adj_lensing_syng_params)


class alm_filter_ninv(object):
    def __init__(self, loc:np.ndarray, ninv:list, nlevs_iso, transfs:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,
                 epsilon=1e-7,
                 tpl:bni.template_dense or None=None, verbose=False, maskbeam=False,
                 dgclm: np.ndarray=np.zeros(3, dtype=np.complex128), Lmmax:tuple[int, int]=(1, 1)):
        r"""CMB inverse-variance and Wiener filtering instance, to use for cg-inversion

            Args:
                loc: co-latitude and longitude of the pixels. Must be something readable to give a (npix, 2) array
                ninv: list of inverse-pixel noise variance maps (itself can be (a list of) string, or array, or ...)
                transfs: CMB transfer functions (assumed to be a 2d array with the first dimension the number of unlensed alm components)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                epsilon: accuracy parameter of ducc synthesis_general and its adjoint
                sht_threads: number of threads for lenspyx SHTs
                verbose: some printout if set, defaults to False
                nlevp_iso: some reference value of the isotropic white noise level in uK-amin (preconditioner etc)
                maskbeam: the data model includes a area-masking operation prior to the beam convolution
                dgclm: lensing deflection, if present

        """
        ncomp_unl = 2 - np.any(dgclm) # There is lensing and we will deal for now with pol
        assert transfs.ndim == 2 and transfs.shape[0] == ncomp_unl

        lmax_transf = transfs.shape[1] - 1

        self.n_inv = ninv

        self.transfs = transfs

        self.lmmax_len = (min(lmax_transf, lenalm_info[0]),  min(lenalm_info[1], lmax_transf))
        self.lmmax_sol = unlalm_info

        self.lmax_sol = unlalm_info[0]
        self.mmax_sol = unlalm_info[1]
        self.ncomp_sol = ncomp_unl
        self.ncomp_dat = 2 # here Q and U

        self.sht_threads = sht_threads
        self.verbose=verbose

        self.nlevs_iso = nlevs_iso
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed
        self.loc = loc

        spin = 2
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
                       'eps_apo': np.sqrt(dl / unlalm_info[0] * np.pi / thtcap),
                        'spin':spin, 'lmax':self.lmmax_len[0], 'mmax':self.lmmax_len[1],
                        'nthreads':sht_threads, 'mode':'STANDARD', 'verbose':verbose}
        adj_syng_params = syng_params.copy()
        adj_syng_params['lmax'] = self.lmmax_sol[0]
        adj_syng_params['mmax'] = self.lmmax_sol[1]

        assert Lmmax is not None, Lmmax
        dgclm2d = np.atleast_2d(dgclm)
        assert Alm.getsize(*Lmmax) == dgclm2d[0].size, (Alm.getsize(*Lmmax), dgclm2d[0].size)


        # FIXME: what to for QE case ?
        sht_mode = 'GRAD_ONLY' if ncomp_unl == 1 else 'STANDARD'
        self.nalms_unl = 1 if (spin == 0 or sht_mode == 'GRAD_ONLY') else 2
        self.spin = spin


        self.MD_op = MD(thtcap, spin, sht_mode, self.lmmax_len , self.lmmax_sol, dgclm, Lmmax,
                      nthreads=sht_threads, epsilon=epsilon, verbose=False)

        if verbose:
            print("general QE p filter setup with thtcap = %.2f deg"%(thtcap/np.pi*180))
            print('eps apo %.3f'%syng_params.get('eps_apo', 0))

        self.syng_params = syng_params
        self.adj_syng_params = adj_syng_params

        self.thtcap = thtcap
        self.npix = npix

        self._dl_7 = dl_7
        self._qu = None
        self._almlen = None

    def hashdict(self):
        return {}

    def _get_sky_geom(self, lmax, weighted=True):
        """Returns a suitable weighted geometry object for the masked beam operation


        """
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

    def _allocate_maps(self, r_dtype):
        """This allocate the positions-space maps for cg iterations


        """
        if self._qu is None:
            self._qu = np.empty((self.ncomp_dat, self.npix), dtype=r_dtype)
        else:
            assert self._qu.dtype == r_dtype, 'not sure how thats possible, find some way to treat the types properly'
        if self._almlen is None:
            self._almlen = np.empty((self.ncomp_dat, Alm.getsize(*self.lmmax_len)), dtype=ctype[r_dtype])

    def deallocate(self):
        """Just cleans some stuff if relevant by deallocating potentially large arrays


        """
        self._qu = None
        self._almlen = None

    def get_fal(self):
        ret = self.transfs ** 2
        for nlev, r in zip(self.nlevs_iso, ret):
            r /= (nlev / 180. / 60. * np.pi) ** 2
        return ret

    def apply_beam(self, eblm:np.ndarray, qumap:np.ndarray, loc=None):
        """Action of beam operator. Here defined from lensed harmonic space to data pixel space

            Note:
                This can modify eblm

        """
        assert eblm.shape == (self.ncomp_dat, Alm.getsize(*self.lmmax_len)), (eblm.shape, (self.ncomp_dat, Alm.getsize(*self.lmmax_len)))
        if loc is None:
            loc = read_map(self.loc)
        for alm, bl in zip(eblm, self.transfs):
            almxfl(alm, bl, self.lmmax_len[1], inplace=True)
        syng(alm=eblm, map=qumap, loc=loc, **self.syng_params)

    def apply_adjoint_beam(self, eblm:np.ndarray, qumap:np.ndarray, loc=None):
        """Adjoint operation to apply_beam.

            Note:
                The adjoint takes as input a map in data pixel space and produces a map in harmonic space

        """
        assert eblm.shape == (self.ncomp_dat, Alm.getsize(*self.lmmax_len)), (eblm.shape, (self.ncomp_dat, Alm.getsize(*self.lmmax_len)))
        if loc is None:
            loc = read_map(self.loc)
        syng_adj(map=qumap,  loc=loc, alm=eblm, **self.adj_syng_params)
        for alm, bl in zip(eblm, self.transfs):
            almxfl(alm, bl, self.lmmax_len[1], inplace=True)

    def apply_alm(self, alm_unl:np.ndarray):
        """Applies operator

            :math:`\mathcal{B}^\dagger N^{-1} \mathcal{B}`


        """
        assert alm_unl.shape == (self.nalms_unl, Alm.getsize(*self.lmmax_sol))
        tim = timer(True, prefix='opfilt_pp')
        loc = read_map(self.loc)
        self._allocate_maps(r_dtype=rtype[alm_unl.dtype])
        tim.add('applyalm init')
        self.MD_op.apply_inplace(alm_unl, self._almlen)
        tim.add('MD')
        self.apply_beam(self._almlen, qumap=self._qu, loc=loc)
        tim.add('beam')
        self.apply_map(self._qu)
        tim.add('Ni')
        self.apply_adjoint_beam(self._almlen, qumap=self._qu, loc=loc)
        tim.add('beam (adjoint)')
        self.MD_op.apply_adjoint_inplace(alm_unl, self._almlen)
        tim.add('MD (adjoint)')
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
        eblm = synalms(clseb, lmax=self.lmmax_len[0], mmax=self.lmmax_len[1])
        qu = np.empty((self.ncomp_dat, self.npix), dtype=rtype[eblm.dtype])
        self.apply_beam(eblm, qumap=qu, loc=read_map(self.loc))
        assert len(self.n_inv) == 1, 'not implemented, fix next line'
        qu += rng.standard_normal(qu.shape, dtype=qu.dtype) * np.sqrt(cli(read_map(self.n_inv[0])))
        return qu

    def get_qlms(self, qudat: np.ndarray or list, alm_wf: np.ndarray, lmax_qlm:int, mmax_qlm:int,
                 norm:str='k', eb_only=False):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                alm_wf: Wiener-filtered CMB maps (alm arrays)
                lmax_qlm: maximum multipole of output
                mmax_qlm: maximum m of lm output
                norm: normalization of the output (whether kappa-like ('k'), or potential-like...)
                eb_only: if True, only the EB estimator is returned

            Returns:
                gradient and curl alms

        """
        assert alm_wf.shape == (self.nalms_unl, Alm.getsize(*self.lmmax_sol))
        assert norm in ['k', 'kappa', 'kappa-like', 'p', 'phi', 'phi-like'], 'implement this (easy)'
        assert self.spin == 2
        qgeom = self._get_sky_geom((2 * self.lmax_sol + lmax_qlm) // 2 + 1, weighted=True)
        resmap_c = np.empty((qgeom.npix(),), dtype=alm_wf.dtype)
        resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        self._get_irespmap(qudat, alm_wf, qgeom, map_out=resmap_r, eb_only=eb_only)  # inplace onto resmap_c and resmap_r
        if not eb_only:
            alm_wf_g = alm_wf
        else:
            alm_wf_g = np.zeros_like(alm_wf)
            alm_wf_g[0, :] = alm_wf[0]  # For gradient leg
        gcs_r = self._get_gpmap(alm_wf_g, 3, qgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = self._get_gpmap(alm_wf_g, 1, qgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r
        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        gc = qgeom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, self.sht_threads, apply_weights=True)
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

                :math:`B^t N^{-1}(X^{\rm dat} - B M D X^{WF})`


        """
        assert len(qu_dat) == self.ncomp_dat and len(eblm_wf) == self.ncomp_sol
        loc = read_map(self.loc)
        self._allocate_maps(r_dtype=qu_dat.dtype)
        self.MD_op.apply_inplace(eblm_wf, self._almlen)
        self.apply_beam(eblm=self._almlen, qumap=self._qu, loc=loc)
        self._qu -= qu_dat
        self.apply_map(self._qu)
        self.apply_adjoint_beam(eblm=self._almlen, qumap=self._qu, loc=loc)
        self._almlen *= -1 # FIXME. is M^t B^t implemented correctly  here? best would be to have a BM I guess
        if eb_only:
            self._almlen[0, :] = 0
        return qgeom.synthesis(self._almlen, self.spin, self.lmmax_len[0], self.lmmax_len[1], self.sht_threads, map=map_out)

pre_op_dense = None # not implemented

def calc_prep(maps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv):
    """cg-inversion pre-operation  (D^t M^t B^t N^{-1} X^{dat})

        Args:
            maps: input polarisation maps
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert maps.ndim == 2 and maps.shape[0] == ninv_filt.ncomp_dat, (maps.shape, ninv_filt.ncomp_dat)
    ninv_filt._allocate_maps(maps.dtype)
    ninv_filt._qu[:] = maps
    ninv_filt.apply_map(ninv_filt._qu)
    ninv_filt.apply_adjoint_beam(eblm=ninv_filt._almlen, qumap=ninv_filt._qu, loc=read_map(ninv_filt.loc))
    alms_unl = np.empty((ninv_filt.nalms_unl, Alm.getsize(*ninv_filt.lmmax_sol)), dtype=ctype[maps.dtype])
    ninv_filt.MD_op.apply_adjoint_inplace(alms_unl, ninv_filt._almlen)
    return alms_unl

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass

class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
        #FIXME: 'ee', 'bb' ?
        specs = ['ee', 'bb'][:ninv_filt.nalms_unl]
        ninv_fals = ninv_filt.get_fal()  # (N_lev * transf) ** 2 basically
        lmax_sol = ninv_filt.lmax_sol
        flmat = {}
        for fl, clk in zip(ninv_fals, specs):
            assert len(s_cls[clk]) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls[clk]))
            if len(fl) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
                print("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(fl)-1, lmax_sol))
                assert np.all(fl > 0)
                spl_sq = spl(np.arange(len(fl), dtype=float), np.log(fl), k=2, ext='extrapolate')
                flmat[clk] = cli(s_cls[clk][:lmax_sol + 1]) + np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
            else:
                flmat[clk] = cli(s_cls[clk][:lmax_sol + 1]) + fl

        self.flmat = np.array([cli(flmat[k]) * (s_cls[k][:lmax_sol +1] > 0.) for k in specs])
        self.lmmax =(ninv_filt.lmax_sol, ninv_filt.mmax_sol)

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, eblm):
        assert eblm.shape == (len(self.flmat), Alm.getsize(*self.lmmax)), (eblm.shape,  Alm.getsize(*self.lmmax))
        ret = np.copy(eblm) # TODO is a copy needed here ?
        for alm, fl in zip(ret, self.flmat):
            almxfl(alm, fl, self.lmmax[1], True)
        return ret

class dot_op:
    def __init__(self, lmax:int, mmax:int or None):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


        """
        if mmax is None or mmax < 0: mmax = lmax
        self.lmmax = (lmax, min(mmax, lmax))

    def __call__(self, eblm1, eblm2):
        assert eblm1[0].size == Alm.getsize(*self.lmmax) and eblm2.size == eblm1.size
        ret = 0
        for alm, blm in zip(eblm1, eblm2):
            ret += np.sum(alm2cl(alm, blm, self.lmmax[0], self.lmmax[1], None) * (2 * np.arange(self.lmmax[0] + 1) + 1))
        return ret

class fwd_op:
    """Forward operation for polarization-only, no primordial B power cg filter


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):

        icls = np.array([cli(s_cls['ee']),cli(s_cls['bb'])])
        for bl, cl in zip(ninv_filt.transfs, icls):
            assert np.all(bl[np.where(cl[:ninv_filt.lmmax_len[0]] == 0.)] == 0.), 'inconsistent inputs'
        self.icls = icls
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'icls':clhash(self.icls)}

    def __call__(self, alms):
        return self.calc(alms)

    def calc(self, alms):
        nlms = np.copy(alms) #TODO is this copy needed ?
        self.ninv_filt.apply_alm(nlms)
        for nlm, alm, icl in zip(nlms, alms, self.icls):
            nlm += almxfl(alm, icl, self.mmax_sol, False)
        return nlms
