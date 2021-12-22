"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import numpy as np
from lenscarf.utils_hp import almxfl, Alm, alm2cl
from lenscarf.utils import clhash, cli, read_map
from lenscarf import  utils_scarf
from lenscarf import remapping
from lenscarf.opfilt import  opfilt_pp, opfilt_base, bmodes_ninv as bni
from scipy.interpolate import UnivariateSpline as spl
from lenscarf.utils import timer


apply_fini = opfilt_pp.apply_fini
pre_op_dense = None # not implemented

class alm_filter_ninv_wl(opfilt_base.scarf_alm_filter_wl):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list, ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,
                 tpl:bni.template_dense or None, verbose=False, lmin_dotop=0, wee=True):
        r"""CMB inverse-variance and Wiener filtering instance, using unlensed E and lensing deflection

            Args:
                ninv_geom: scarf geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (strings, or arrays, or ...)
                ffi: remapping.deflection instance that performs the forward and backward lensing
                transf: CMB transfer function (assumed to be the same in E and B)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for scarf SHTs
                verbose: some printout if set, defaults to False
                wee: includes the EE-like term in the generalized QE


        """
        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = len(transf) - 1

        lmax_sol = lmax_unl
        mmax_sol = min(lmax_unl, mmax_unl)
        super().__init__(lmax_sol, mmax_sol, ffi)

        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)
        self.n_inv = ninv
        self.b_transf = transf
        self.lmin_dotop = lmin_dotop
        self.wee = wee


        sc_job = utils_scarf.scarfjob()
        if not np.all(ninv_geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            print('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
            nr = ninv_geom.get_nrings()
            ninv_geom_ = utils_scarf.Geometry(nr, ninv_geom.nph.copy(), ninv_geom.ofs.copy(), 1, ninv_geom.phi0.copy(), ninv_geom.theta.copy(), np.ones(nr, dtype=float))
            # Does not seem to work without the 'copy'
        else:
            ninv_geom_ = ninv_geom
        assert np.all(ninv_geom_.weight == 1.)
        sc_job.set_geometry(ninv_geom_)
        sc_job.set_nthreads(sht_threads)
        sc_job.set_triangular_alm_info(lmax_len, mmax_len)
        self.ninv_geom = ninv_geom
        self.sc_job = sc_job

        self.verbose=verbose

        self._nlevp = None
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf),
                'geom':utils_scarf.Geom.hashdict(self.sc_job.geom),
                'deflection':self.ffi.hashdict(),
                'unalm':(self.lmax_sol, self.mmax_sol), 'lenalm':(self.lmax_len, self.mmax_len) }

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
                nlev_febl = 10800. / np.sqrt(np.sum(read_map(self.n_inv[0])) / (4.0 * np.pi)) / np.pi
            elif len(self.n_inv) == 3:
                nlev_febl = 10800. / np.sqrt(
                    (0.5 * np.sum(read_map(self.n_inv[0])) + np.sum(read_map(self.n_inv[2]))) / (4.0 * np.pi)) / np.pi
            else:
                assert 0
            self._nlevp = nlev_febl
            print('Using nlevp %.2f amin'%self._nlevp)
        n_inv_cl_p = self.b_transf ** 2  / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return n_inv_cl_p, n_inv_cl_p.copy()

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol, lmin=self.lmin_dotop)

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

    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  D^t B^T N^{-1} B D, where D is lensing, B the transfer function)

        """
        # Forward lensing here
        tim = self.tim
        tim.reset_t0()
        lmax_unl =Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        eblm = self.ffi.lensgclm([elm, np.zeros_like(elm)], self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        tim.add('lensgclm fwd')

        almxfl(eblm[0], self.b_transf, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf, self.mmax_len, inplace=True)
        tim.add('transf')

        qumap = self.sc_job.alm2map_spin(eblm, 2)
        tim.add('alm2map_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        self.apply_map(qumap)  # applies N^{-1}
        tim.add('apply ninv')

        eblm = self.sc_job.map2alm_spin(qumap, 2)
        tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        # The map2alm is here a sum rather than integral, so geom.weights are assumed to be unity
        almxfl(eblm[0], self.b_transf, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf, self.mmax_len, inplace=True)
        tim.add('transf')

        # backward lensing with magn. mult. here
        eblm = self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        elm[:] = eblm[0]
        tim.add('lensgclm bwd')
        if self.verbose:
            print(tim)

    def get_qlms(self, qudat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: utils_scarf.pbdGeometry, alm_wf_leg2 :None or np.ndarray=None):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: gradient leg Winer filtered CMB, if different from ivf leg
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            Note: all implementation signs are super-weird but end result correct...
        """
        assert len(qudat) == 2
        assert (qudat[0].size == utils_scarf.Geom.npix(self.ninv_geom)) and (qudat[0].size == qudat[1].size)
        ebwf = np.array([elm_wf, np.zeros_like(elm_wf)])
        repmap, impmap = self._get_irespmap(qudat, ebwf, q_pbgeom)
        if alm_wf_leg2 is not None:
            ebwf[0, :] = alm_wf_leg2
        Gs, Cs = self._get_gpmap(ebwf, 3, q_pbgeom)  # 2 pos.space maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(ebwf, 1,  q_pbgeom)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        lmax_qlm = self.ffi.lmax_dlm
        mmax_qlm = self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C


    def _get_gpmap(self, eblm_wf:np.ndarray or list, spin:int, q_pbgeom:utils_scarf.pbdGeometry):
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
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap(eblm, self.mmax_sol, spin, False)

    def _get_irespmap(self, qudat:np.ndarray, ebwf:np.ndarray or list, q_pbgeom:utils_scarf.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """

        assert len(qudat) == 2 and len(ebwf) == 2
        assert np.all(self.sc_job.geom.weight == 1.) # sum rather than integrals

        ebwf = self.ffi.lensgclm(ebwf, self.mmax_sol, 2, self.lmax_len, self.mmax_len, False)
        almxfl(ebwf[0], self.b_transf, self.mmax_len, True)
        almxfl(ebwf[1], self.b_transf, self.mmax_len, True)
        qu = qudat - self.sc_job.alm2map_spin(ebwf, 2)
        self.apply_map(qu)
        ebwf = self.sc_job.map2alm_spin(qu, 2)
        almxfl(ebwf[0], self.b_transf * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.b_transf * 0.5, self.mmax_len, True)
        return q_pbgeom.geom.alm2map_spin(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))


class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel, ninv_fbl = ninv_filt.get_febl() # (N_lev * transf) ** 2 basically
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            print("PRE_OP_DIAG: extending E transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        flmat = cli(s_cls['ee'][:lmax_sol + 1]) + ninv_fel[:lmax_sol + 1]
        self.flmat = cli(flmat) * (s_cls['ee'][:lmax_sol +1] > 0.)
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, elm):
        return self.calc(elm)

    def calc(self, elm):
        assert Alm.getsize(self.lmax, self.mmax) == elm.size, (self.lmax, self.mmax, Alm.getlmax(elm.size, self.mmax))
        return almxfl(elm, self.flmat, self.mmax, False)

def calc_prep(qumaps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            qumaps: input polarisation maps array of shape (2, npix)
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(qumaps, np.ndarray)
    assert np.all(ninv_filt.sc_job.geom.weight==1.) # Sum rather than integral, hence requires unit weights
    qumap= np.copy(qumaps)
    ninv_filt.apply_map(qumap)

    eblm = ninv_filt.sc_job.map2alm_spin(qumap, 2)
    almxfl(eblm[0], ninv_filt.b_transf, ninv_filt.mmax_len, True)
    almxfl(eblm[1], ninv_filt.b_transf, ninv_filt.mmax_len, True)
    elm, blm = ninv_filt.ffi.lensgclm(eblm, ninv_filt.mmax_len, 2, ninv_filt.lmax_sol,ninv_filt.mmax_sol, backwards=True)
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
    return elm


class dot_op:
    def __init__(self, lmax: int, mmax: int or None, lmin=0):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


        """
        if mmax is None or mmax < 0: mmax = lmax
        self.lmax = lmax
        self.mmax = min(mmax, lmax)
        self.lmin = int(lmin)

    def __call__(self, elm1, elm2):
        assert elm1.size == Alm.getsize(self.lmax, self.mmax), (elm1.size, Alm.getsize(self.lmax, self.mmax))
        assert elm2.size == Alm.getsize(self.lmax, self.mmax), (elm2.size, Alm.getsize(self.lmax, self.mmax))
        return np.sum(alm2cl(elm1, elm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))


class fwd_op:
    """Forward operation for polarization-only, no primordial B power cg filter


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
        self.iclee = cli(s_cls['ee'])
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'iclee': clhash(self.iclee),
                'n_inv_filt': self.ninv_filt.hashdict()}

    def __call__(self, elm):
        return self.calc(elm)

    def calc(self, elm):
        nlm = np.copy(elm)
        self.ninv_filt.apply_alm(nlm)
        nlm += almxfl(elm, self.iclee, self.mmax_sol, False)
        almxfl(nlm, self.iclee > 0., self.mmax_sol, True)
        return nlm