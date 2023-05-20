"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from lenspyx.remapping import utils_geom
from lenspyx import remapping

from delensalot.utils import timer, clhash, cli, read_map
from delensalot.utility.utils_hp import almxfl, Alm, alm2cl, synalm, default_rng
from delensalot.core.opfilt import  opfilt_pp, opfilt_base, bmodes_ninv as bni




apply_fini = opfilt_pp.apply_fini
pre_op_dense = None # not implemented


class alm_filter_ninv_wl(opfilt_base.scarf_alm_filter_wl):
    def __init__(self, ninv_geom:utils_geom.Geom, ninv:list, ffi_ee:remapping.deflection, ffi_eb:remapping.deflection,
                 transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,tpl:bni.template_dense or None,
                 transf_blm:np.ndarray or None=None, verbose=False, lmin_dotop=0):
        r"""CMB inverse-variance and Wiener filtering instance, using unlensed E and lensing deflection

            This instance distinguishes between deflection fields mapping E to E and E to B

            Args:
                ninv_geom: scarf geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (either 1 (QQ=UU) or 3  (QQ UU and QU noise) arrays of the right size)
                ffi_ee: remapping.deflection instance for the E to E lensing operations
                ffi_ee: remapping.deflection instance for the E to B lensing operations
                transf: E-CMB transfer function
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for scarf SHTs
                verbose: some printout if set, defaults to False
                transf_blm: B-CMB transfer function (if different from E)

        """
        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = len(transf) - 1

        lmax_sol = lmax_unl
        mmax_sol = min(lmax_unl, mmax_unl)
        super().__init__(lmax_sol, mmax_sol, ffi_ee)

        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)
        self.n_inv = ninv
        self.b_transf_elm = transf
        self.b_transf_blm = transf if transf_blm is None else transf_blm
        self.lmin_dotop = lmin_dotop

        self.ffi_eb = ffi_eb
        self.ffi_ee = ffi_ee

        sc_job = utils_geom.scarfjob()
        if not np.all(ninv_geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            log.info('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
            nr = ninv_geom.get_nrings()
            ninv_geom_ = utils_geom.Geom(nr, ninv_geom.nph.copy(), ninv_geom.ofs.copy(), 1, ninv_geom.phi0.copy(), ninv_geom.theta.copy(), np.ones(nr, dtype=float))
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

    def set_ffi(self, ffi):
        """Update of lensing deflection instance"""
        assert 0, 'fix this'

    def lensforward(self, elm): # this can only take a e and no b
        eblm = self.ffi_ee.lensgclm([elm, np.zeros_like(elm)], self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        if self.ffi_eb is not self.ffi_ee: # filling B-mode with second deflection
            eblm_2 = self.ffi_eb.lensgclm([elm, np.zeros_like(elm)], self.mmax_sol, 2, self.lmax_len, self.mmax_len)
            eblm[1][:] = eblm_2[1]
        return eblm

    def lensbackward(self, eblm):
        eblm_ee = self.ffi_ee.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        if self.ffi_eb is not self.ffi_ee: # filling B-mode with second deflection
            eblm_eb = self.ffi_eb.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        else:
            eblm_eb = eblm_ee
        return 0.5 * (eblm_ee[0] + eblm_eb[0])

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transfe':clhash(self.b_transf_elm),'transfb':clhash(self.b_transf_blm),
                'geom':utils_geom.Geom.hashdict(self.sc_job.geom),
                'deflection_ee':self.ffi_ee.hashdict(),'deflection_eb':self.ffi_eb.hashdict(),
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
            log.info('Using nlevp %.2f amin'%self._nlevp)
        n_inv_cl_e = self.b_transf_elm ** 2  / (self._nlevp/ 180. / 60. * np.pi) ** 2
        n_inv_cl_b = self.b_transf_blm ** 2  / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return n_inv_cl_e, n_inv_cl_b.copy()

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
        eblm = self.lensforward(elm)
        tim.add('lensgclm fwd')

        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')

        qumap = self.sc_job.alm2map_spin(eblm, 2)
        tim.add('alm2map_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        self.apply_map(qumap)  # applies N^{-1}
        tim.add('apply ninv')

        eblm = self.sc_job.map2alm_spin(qumap, 2)
        tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        # The map2alm is here a sum rather than integral, so geom.weights are assumed to be unity
        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, inplace=True)
        tim.add('transf')

        # backward lensing with magn. mult. here
        elm[:] = self.lensbackward(eblm)
        tim.add('lensgclm bwd')
        if self.verbose:
            print(tim)

    def synalm(self, unlcmb_cls:dict, cmb_phas=None, get_unlelm=False):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        eblm = self.lensforward(elm)
        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, True)
        # cant use here sc_job since it is using the unit weight transforms
        QU = self.ninv_geom.alm2map_spin(eblm, 2, self.lmax_len, self.mmax_len, self.ffi_ee.sht_tr, (-1., 1.))
        del eblm # Adding noise
        if len(self.n_inv) == 1: # QQ = UU
            pixnoise = np.sqrt(cli(self.n_inv[0]))
            QU[0] += default_rng().standard_normal(utils_geom.Geom.npix(self.ninv_geom)) * pixnoise
            QU[1] += default_rng().standard_normal(utils_geom.Geom.npix(self.ninv_geom)) * pixnoise
        elif len(self.n_inv) == 3: #QQ UU QU
            assert 0, 'this is not implemented at the moment, but this is easy'
        else:
            assert 0, 'you should never land here'
        return elm, QU if get_unlelm else QU

    def get_qlms(self, qudat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, alm_wf_leg2 :None or np.ndarray=None):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: gradient leg Winer filtered CMB, if different from ivf leg
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            Note: all implementation signs are super-weird but end result correct...
        """
        assert len(qudat) == 2
        assert (qudat[0].size == utils_geom.Geom.npix(self.ninv_geom)) and (qudat[0].size == qudat[1].size)
        (repmap_e, impmap_e), (repmap_b, impmap_b) = self._get_irespmap(qudat, elm_wf, q_pbgeom)
        if alm_wf_leg2 is not None:
            elm_wf[:] = alm_wf_leg2
        Gs, Cs = self._get_gpmap(elm_wf, 3, q_pbgeom, self.ffi_ee)  # 2 pos.space maps
        GC_ee = (repmap_e - 1j * impmap_e) * (Gs + 1j * Cs)  # (-2 , +3)
        if self.ffi_eb is not self.ffi_ee:
            Gs, Cs = self._get_gpmap(elm_wf, 3, q_pbgeom, self.ffi_eb)  # 2 pos.space maps
        GC_eb = (repmap_b - 1j * impmap_b) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = self._get_gpmap(elm_wf, 1,  q_pbgeom, self.ffi_ee)
        GC_ee -= (repmap_e + 1j * impmap_e) * (Gs - 1j * Cs)  # (+2 , -1)
        if self.ffi_eb is not self.ffi_ee:
            Gs, Cs = self._get_gpmap(elm_wf, 1, q_pbgeom, self.ffi_eb)
        GC_eb -= (repmap_b + 1j * impmap_b) * (Gs - 1j * Cs)  # (+2 , -1)

        del repmap_e, impmap_e, repmap_b, impmap_b, Gs, Cs
        lmax_qlm = self.ffi_ee.lmax_dlm
        mmax_qlm = self.ffi_ee.mmax_dlm
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        Gee, Cee = q_pbgeom.geom.map2alm_spin([GC_ee.real, GC_ee.imag], 1, lmax_qlm, mmax_qlm, self.ffi_ee.sht_tr, (-1., 1.))
        del GC_ee
        for G in [Gee, Cee]:
            almxfl(G, fl, mmax_qlm, True)
        lmax_qlm = self.ffi_eb.lmax_dlm
        mmax_qlm = self.ffi_eb.mmax_dlm
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        Geb, Ceb = q_pbgeom.geom.map2alm_spin([GC_eb.real, GC_eb.imag], 1, lmax_qlm, mmax_qlm, self.ffi_eb.sht_tr, (-1., 1.))
        del GC_eb
        for G in [Geb, Ceb]:
            almxfl(G, fl, mmax_qlm, True)
        return (Gee, Cee), (Geb, Ceb)

    def get_qlms_mf(self, mfkey, q_pbgeom:utils_geom.pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        if mfkey in [1]: # This should be B^t x, D dC D^t B^t Covi x, x random phases in pixel space here
            if phas is None:
                # unit variance phases in Q U space
                phas = np.array([default_rng().standard_normal(utils_geom.Geom.npix(self.ninv_geom)),
                                 default_rng().standard_normal(utils_geom.Geom.npix(self.ninv_geom))])
            assert phas[0].size == utils_geom.Geom.npix(self.ninv_geom)
            assert phas[1].size == utils_geom.Geom.npix(self.ninv_geom)

            soltn = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(soltn, phas, dot_op=self.dot_op())

            phas = self.ninv_geom.map2alm_spin(phas, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
            almxfl(phas[0], 0.5 * self.b_transf_elm, self.mmax_len, True)
            almxfl(phas[1], 0.5 * self.b_transf_blm, self.mmax_len, True)
            repmap, impmap = q_pbgeom.geom.alm2map_spin(phas, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
            assert 0, 'not implemented'
            Gs, Cs = self._get_gpmap(soltn, 3, q_pbgeom)  # 2 pos.space maps
            GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
            Gs, Cs = self._get_gpmap(soltn, 1, q_pbgeom)
            GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
            del repmap, impmap, Gs, Cs

        elif mfkey in [0]: # standard gQE, quite inefficient but simple
            assert phas is None, 'discarding this phase anyways'
            QUdat = np.array(self.synalm(cls_filt))
            elm_wf = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(elm_wf, QUdat, dot_op=self.dot_op())
            return self.get_qlms(QUdat, elm_wf, q_pbgeom)
        else:
            assert 0, mfkey + ' not implemented'
        assert 0, 'fix these lines'
        lmax_qlm = self.ffi.lmax_dlm
        mmax_qlm = self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_gpmap(self, elm_wf:np.ndarray or list, spin:int, q_pbgeom:utils_geom.pbdGeometry, ffi:remapping.deflection):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert  Alm.getlmax(elm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(elm_wf.size, self.mmax_sol), self.lmax_sol)
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        elm = almxfl(elm_wf, fl, self.mmax_sol, False)
        ffi = ffi.change_geom(q_pbgeom) if q_pbgeom is not ffi.pbgeom else ffi
        return ffi.gclm2lenmap([elm, np.zeros_like(elm)], self.mmax_sol, spin, False)


    def _get_irespmap(self, qudat:np.ndarray, elm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """

        assert len(qudat) == 2
        assert np.all(self.sc_job.geom.weight == 1.) # sum rather than integrals
        assert  Alm.getlmax(elm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(elm_wf.size, self.mmax_sol), self.lmax_sol)

        ebwf = self.lensforward(elm_wf)
        almxfl(ebwf[0], self.b_transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], self.b_transf_blm, self.mmax_len, True)
        qu = qudat - self.sc_job.alm2map_spin(ebwf, 2)
        self.apply_map(qu)
        ebwf = self.sc_job.map2alm_spin(qu, 2)
        almxfl(ebwf[0], self.b_transf_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.b_transf_blm * 0.5, self.mmax_len, True)
        res_e = q_pbgeom.geom.alm2map_spin([ebwf[0], np.zeros_like(ebwf[0])], 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
        res_b = q_pbgeom.geom.alm2map_spin([np.zeros_like(ebwf[1]), ebwf[1]], 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
        return res_e, res_b


class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel, ninv_fbl = ninv_filt.get_febl() # (N_lev * transf) ** 2 basically
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending E transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
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
    qumap = np.copy(qumaps)
    ninv_filt.apply_map(qumap)
    eblm = ninv_filt.sc_job.map2alm_spin(qumap, 2)
    almxfl(eblm[0], ninv_filt.b_transf_elm, ninv_filt.mmax_len, True)
    almxfl(eblm[1], ninv_filt.b_transf_blm, ninv_filt.mmax_len, True)
    elm = ninv_filt.lensbackward(eblm)
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