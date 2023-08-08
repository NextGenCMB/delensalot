"""Lenspyx-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
from scipy.interpolate import UnivariateSpline as spl
import numpy as np

from lenspyx import remapping
from lenspyx.remapping import utils_geom
from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utils import clhash, cli, read_map, timer
from delensalot.utility.utils_hp import almxfl, Alm, alm2cl, synalm, default_rng
from delensalot.core.opfilt import opfilt_base, QE_opfilt_aniso_p, bmodes_ninv as bni
from delensalot.core.opfilt import MAP_opfilt_iso_tp
apply_fini = QE_opfilt_aniso_p.apply_fini

pre_op_dense = None # not implemented
fwd_op = MAP_opfilt_iso_tp.fwd_op
dot_op = MAP_opfilt_iso_tp.dot_op
pre_op_diag = MAP_opfilt_iso_tp.pre_op_diag


class alm_filter_ninv_wl(opfilt_base.alm_filter_wl):
    def __init__(self, ninv_geom:utils_geom.Geom, ninv:list, ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,p_tpl:bni.template_dense or None,
                 transf_elm:np.ndarray or None=None, transf_blm:np.ndarray or None=None, verbose=False, lmin_dotop=0):
        r"""CMB inverse-variance and Wiener filtering instance, using unlensed E and lensing deflection

            Args:
                ninv_geom: lenspyx geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (either 1 (QQ=UU) or 3  (QQ UU and QU noise) arrays of the right size)
                ffi: remapping.deflection instance that performs the forward and backward lensing
                transf: T-CMB transfer function
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for lenspyx SHTs
                verbose: some printout if set, defaults to False
                transf_elm: E-CMB transfer function (if different from T)
                transf_blm: B-CMB transfer function (if different from E)

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
        #FIXME: transfer fcts
        self.b_transf_tlm = transf
        self.b_transf_elm = transf if transf_elm is None else transf_elm
        self.b_transf_blm = transf_elm if transf_blm is None else transf_blm
        self.lmin_dotop = lmin_dotop

        self.sht_threads = sht_threads
        self.ninv_geom = ninv_geom

        self.verbose=verbose

        self._nlevp, self._nlevt = None, None
        self.tim = timer(True, prefix='opfilt')

        self.p_template = p_tpl # here just one template allowed
        self.t_template = None

    def hashdict(self):
        return {'ninv':self._ninv_hash(),
                'transft': clhash(self.b_transf_tlm),
                'transfe':clhash(self.b_transf_elm),
                'transfb':clhash(self.b_transf_blm),
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

    def get_ftebl(self):
        assert len(self.n_inv) in [2, 4], (len(self.n_inv))
        if self._nlevp is None:
            if len(self.n_inv) == 1 + 1: # + 1 because of T
                nlev_febl = 10800. / np.sqrt(np.sum(read_map(self.n_inv[1 + 0])) / (4.0 * np.pi)) / np.pi
            elif len(self.n_inv) == 3 + 1:
                nlev_febl = 10800. / np.sqrt(
                    (0.5 * np.sum(read_map(self.n_inv[1 + 0])) + np.sum(read_map(self.n_inv[1 + 2]))) / (4.0 * np.pi)) / np.pi
            else:
                assert 0
            self._nlevp = nlev_febl
            log.info('Using nlevp %.2f amin'%self._nlevp)
        if self._nlevt is None:
            nlev_ftl = 10800. / np.sqrt(np.sum(read_map(self.n_inv[0])) / (4.0 * np.pi)) / np.pi
            self._nlevt = nlev_ftl
            log.info('Using nlevt %.2f amin'%self._nlevt)

        n_inv_cl_t = self.b_transf_tlm ** 2 /  (self._nlevt / 180. / 60. * np.pi) ** 2
        n_inv_cl_e = self.b_transf_elm ** 2  / (self._nlevp / 180. / 60. * np.pi) ** 2
        n_inv_cl_b = self.b_transf_blm ** 2  / (self._nlevp / 180. / 60. * np.pi) ** 2
        return n_inv_cl_t, n_inv_cl_e, n_inv_cl_b

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol, lmin=self.lmin_dotop)

    def apply_map(self, tqumap, polonly=False):
        """Applies pixel inverse-noise variance maps

            Now only accepts TT, QQ=UU, or TT, QQ, QU, UU type of inverse noise variance map


        """
        assert len(tqumap) == 2 if polonly else 3, (len(tqumap), polonly)
        qumap = tqumap[(not polonly):]
        assert len(qumap) == 2, (len(qumap))
        if not polonly:
            assert self.t_template is None
            tqumap[0] *= self.n_inv[0]
        if len(self.n_inv) == 1 + 1:  #  QQ = UU (+1 because of T)
            qumap *= self.n_inv[1 + 0]
            if self.p_template is not None:
                ts = [self.p_template] # Hack, this is only meant for one template
                coeffs = np.concatenate(([t.dot(qumap) for t in ts]))
                coeffs = np.dot(ts[0].tniti(), coeffs)
                pmodes = np.zeros_like(qumap)
                im = 0
                for t in ts:
                    t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                    im += t.nmodes
                pmodes *= self.n_inv[1 + 0]
                qumap -= pmodes

        elif len(self.n_inv) == 1 + 3:  # QQ, QU, UU
            assert self.p_template is None
            qmap, umap = qumap
            qmap_copy = qmap.copy()
            qmap *= self.n_inv[1 + 0]
            qmap += self.n_inv[1 + 1] * umap
            umap *= self.n_inv[1 + 2]
            umap += self.n_inv[1 + 1] * qmap_copy
            del qmap_copy
        else:
            assert 0

    def apply_alm(self, telm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  D^t B^T N^{-1} B D, where D is lensing, B the transfer function)

        """
        # Forward lensing here
        tim = self.tim
        tim.reset_t0()
        lmax_unl = Alm.getlmax(telm[0].size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)

        eblm = self.ffi.lensgclm(telm[1], self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        tlm  = self.ffi.lensgclm(telm[0], self.mmax_sol, 0, self.lmax_len, self.mmax_len)
        tim.add('lensgclm fwd')
        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, inplace=True)
        almxfl(tlm    , self.b_transf_tlm, self.mmax_len, inplace=True)
        tlm2d = tlm.reshape(1, tlm.size)
        tim.add('transf')
        tqumap = np.empty((3, self.ninv_geom.npix()), dtype=float)
        self.ninv_geom.synthesis(eblm, 2, self.lmax_len, self.mmax_len, self.sht_threads, map=tqumap[1:])
        self.ninv_geom.synthesis(tlm, 0, self.lmax_len, self.mmax_len, self.sht_threads, map=tqumap[0:1])
        tim.add('alm2map_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, len(self.ninv_geom.ofs)))

        self.apply_map(tqumap)  # applies N^{-1}
        tim.add('apply ninv')

        self.ninv_geom.adjoint_synthesis(tqumap[1:], 2, self.lmax_len, self.mmax_len, self.sht_threads,
                                                apply_weights=False, alm=eblm)
        self.ninv_geom.adjoint_synthesis(tqumap[0], 0, self.lmax_len, self.mmax_len, self.sht_threads,
                                                apply_weights=False, alm=tlm2d)
        # The map2alm is here a sum rather than integral, so geom.weights are assumed to be unity
        tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, len(self.ninv_geom.ofs)))

        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, inplace=True)
        almxfl(tlm    , self.b_transf_tlm, self.mmax_len, inplace=True)

        tim.add('transf')

        # Writing onto elm2d
        self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
                                backwards=True, out_sht_mode='GRAD_ONLY', gclm_out=telm[1:])
        self.ffi.lensgclm(tlm, self.mmax_len, 0, self.lmax_sol, self.mmax_sol,
                               backwards=True,  gclm_out=telm[0:1])

        tim.add('lensgclm bwd')
        if self.verbose:
            print(tim)

    def synalm(self, unlcmb_cls:dict, cmb_phas=None, get_unlelm=False):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        assert 0, 'fix this'
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        eblm = self.ffi.lensgclm(elm, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(eblm[0], self.b_transf_elm, self.mmax_len, True)
        almxfl(eblm[1], self.b_transf_blm, self.mmax_len, True)
        # cant use here geom_ since it is using the unit weight transforms
        QU = self.ninv_geom.alm2map_spin(eblm, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
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

    def get_qlms_old(self, qudat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, alm_wf_leg2 :None or np.ndarray=None):
        """

            Args:
                qudat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: gradient leg Winer filtered CMB, if different from ivf leg
                q_pbgeom: lenspyx pbounded-geometry of for the position-space mutliplication of the legs

            Note: all implementation signs are super-weird but end result correct...
        """
        assert len(qudat) == 2
        assert (qudat[0].size == utils_geom.Geom.npix(self.ninv_geom)) and (qudat[0].size == qudat[1].size)
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

    def get_qlms(self, tqu_dat: np.ndarray or list, telm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, alm_wf_leg2:None or np.ndarray =None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert alm_wf_leg2 is None
        assert tqu_dat[0].size == self.ninv_geom.npix(), (Alm.getlmax(tqu_dat[0].size, self.mmax_len), self.ninv_geom.npix())
        assert tqu_dat[1].size == self.ninv_geom.npix(), (Alm.getlmax(tqu_dat[1].size, self.mmax_len), self.ninv_geom.npix())
        assert tqu_dat[2].size == self.ninv_geom.npix(), (Alm.getlmax(tqu_dat[2].size, self.mmax_len), self.ninv_geom.npix())

        assert Alm.getlmax(telm_wf[0].size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(telm_wf[0].size, self.mmax_sol), self.lmax_sol)
        assert Alm.getlmax(telm_wf[1].size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(telm_wf[1].size, self.mmax_sol), self.lmax_sol)

        t_dat = tqu_dat[0]
        qu_dat = tqu_dat[1:]
        tlm_wf, elm_wf = telm_wf

        # Spin-2 part
        resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
        resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        self._get_irespmap(qu_dat, elm_wf, q_pbgeom, map_out=resmap_r)  # inplace onto resmap_c and resmap_r

        gcs_r = self._get_gpmap(elm_wf, 3, q_pbgeom)  # 2 pos.space maps, uses then complex view onto real array
        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        gcs_r = self._get_gpmap(elm_wf, 1, q_pbgeom)
        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        del resmap_c, resmap_r, gcs_r

        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array

        # Spin-0 part
        gc_r += self._get_gtmap(tlm_wf, q_pbgeom) * self._get_irestmap(t_dat, tlm_wf, q_pbgeom)
        # Projection onto gradient and curl
        lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
        gc = q_pbgeom.geom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
        del gc_r, gc_c
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(gc[0], fl, mmax_qlm, True)
        almxfl(gc[1], fl, mmax_qlm, True)
        return gc

    def get_qlms_mf(self, mfkey, q_pbgeom:utils_geom.pbdGeometry, mchain, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        assert 0, 'fix this, this was pol-only'
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

            Gs, Cs = self._get_gpmap([soltn, np.zeros_like(soltn)], 3, q_pbgeom)  # 2 pos.space maps
            GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
            Gs, Cs = self._get_gpmap([soltn, np.zeros_like(soltn)], 1, q_pbgeom)
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
        lmax_qlm = self.ffi.lmax_dlm
        mmax_qlm = self.ffi.mmax_dlm
        G, C = q_pbgeom.geom.map2alm_spin([GC.real, GC.imag], 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr, (-1., 1.))
        del GC
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C

    def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n)
                                           sqrt(l-2 (l+3)) _3 Ylm(n)`

            Output is list with real and imaginary part of the spin 1 or 3 transforms.


        """
        assert elm_wf.ndim == 1
        assert Alm.getlmax(elm_wf.size, self.mmax_sol) == self.lmax_sol
        assert spin in [1, 3], spin
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        elm = almxfl(elm_wf, fl, self.mmax_sol, False).reshape((1, elm_wf.size))
        return ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)

    def _get_irespmap(self, qudat:np.ndarray, ewf:np.ndarray or list, q_pbgeom:utils_geom.pbdGeometry, map_out=None):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """

        assert len(qudat) == 2, (len(qudat), len(ewf))
        assert len(self.n_inv) in [2, 4], (len(self.n_inv)) # must adapt this if T and Pol noise couplings
        ebwf = self.ffi.lensgclm(ewf, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(ebwf[0], self.b_transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], self.b_transf_blm, self.mmax_len, True)
        qu = qudat - self.ninv_geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.sht_threads)
        self.apply_map(qu, polonly=True)
        self.ninv_geom.adjoint_synthesis(qu, 2, self.lmax_len, self.mmax_len, self.sht_threads,
                                                apply_weights=False, alm=ebwf)
        almxfl(ebwf[0], self.b_transf_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.b_transf_blm * 0.5, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)

    def _get_irestmap(self, tlm_dat:np.ndarray, tlm_wf:np.ndarray, q_pbgeom: utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        twf = almxfl(self.ffi.lensgclm(tlm_wf, self.mmax_sol, 0, self.lmax_len, self.mmax_len).squeeze(), self.b_transf_tlm, self.mmax_len, False)
        twf = tlm_dat - self.ninv_geom.synthesis(twf, 0, self.lmax_len, self.mmax_len, self.sht_threads).squeeze()
        assert len(self.n_inv) in [2, 4] and self.t_template is None, 'fix the following line'
        twf *= self.n_inv[0]
        tlm = self.ninv_geom.adjoint_synthesis(twf, 0, self.lmax_len, self.mmax_len, self.sht_threads, apply_weights=False).squeeze()
        almxfl(tlm, self.b_transf_tlm, self.mmax_len, True)
        return q_pbgeom.geom.synthesis(tlm, 0, self.lmax_len, self.mmax_len, self.ffi.sht_tr).squeeze()

    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom: utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert Alm.getlmax(tlm_wf.size, self.mmax_sol) == self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        ffi = self.ffi.change_geom(q_pbgeom.geom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        alm = almxfl(tlm_wf, fl, self.mmax_sol, False)
        return ffi.gclm2lenmap(alm, self.mmax_sol, 1, False)



def calc_prep(tqumaps:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            tqumaps: input intensity and polarisation maps array of shape (2, npix)
            s_cls: CMB spectra dictionary (here only 'tt' and 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(tqumaps, np.ndarray)
    tqumap = np.copy(tqumaps)
    ninv_filt.apply_map(tqumap)
    eblm = ninv_filt.ninv_geom.adjoint_synthesis(tqumap[1:], 2, ninv_filt.lmax_len, ninv_filt.mmax_len, ninv_filt.sht_threads,
                                                 apply_weights=False)
    tlm = ninv_filt.ninv_geom.adjoint_synthesis(tqumap[0:1], 0, ninv_filt.lmax_len, ninv_filt.mmax_len, ninv_filt.sht_threads,
                                                 apply_weights=False).squeeze()
    almxfl(eblm[0], ninv_filt.b_transf_elm, ninv_filt.mmax_len, True)
    almxfl(eblm[1], ninv_filt.b_transf_blm, ninv_filt.mmax_len, True)
    almxfl(tlm,     ninv_filt.b_transf_tlm, ninv_filt.mmax_len, True)

    telm = np.empty((2, Alm.getsize(ninv_filt.lmax_sol, ninv_filt.mmax_sol)), complex)
    ninv_filt.ffi.lensgclm(eblm, ninv_filt.mmax_len, 2, ninv_filt.lmax_sol, ninv_filt.mmax_sol,
                                      backwards=True, out_sht_mode='GRAD_ONLY', gclm_out=telm[1:])
    ninv_filt.ffi.lensgclm(tlm, ninv_filt.mmax_len, 0, ninv_filt.lmax_sol, ninv_filt.mmax_sol,
                                      backwards=True, gclm_out=telm[0:1])
    almxfl(telm[0], s_cls['tt'] > 0., ninv_filt.mmax_sol, True)
    almxfl(telm[1], s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
    return telm