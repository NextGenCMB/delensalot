"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from delensalot.utils_hp import almxfl, Alm, alm2cl, synalm, default_rng
from delensalot.utils import clhash, cli, read_map
from lenspyx.remapping import utils_geom
from lenspyx import remapping
from delensalot.opfilt import opfilt_base, tmodes_ninv as tni

from scipy.interpolate import UnivariateSpline as spl
from delensalot.utils import timer


pre_op_dense = None # not implemented
def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass

class alm_filter_ninv_wl(opfilt_base.scarf_alm_filter_wl):
    def __init__(self, ninv_geom:utils_geom.Geometry, ninv: np.ndarray, ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int,verbose=False, lmin_dotop=0, tpl:tni.template_tfilt or None =None):
        r"""CMB inverse-variance and Wiener filtering instance, using unlensed E and lensing deflection

            Args:
                ninv_geom: scarf geometry for the inverse-pixel-noise variance SHTs
                ninv: inverse-pixel noise variance maps
                ffi: remapping.deflection instance that performs the forward and backward lensing
                transf: E-CMB transfer function
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for scarf SHTs
                verbose: some printout if set, defaults to False

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
        self.b_transf_tlm = transf
        self.lmin_dotop = lmin_dotop


        sc_job = utils_geom.scarfjob()
        if not np.all(ninv_geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            log.info('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
            nr = ninv_geom.get_nrings()
            ninv_geom_ = utils_geom.Geometry(nr, ninv_geom.nph.copy(), ninv_geom.ofs.copy(), 1, ninv_geom.phi0.copy(), ninv_geom.theta.copy(), np.ones(nr, dtype=float))
            # Does not seem to work without the 'copy'
        else:
            ninv_geom_ = ninv_geom
        assert np.all(ninv_geom_.weight == 1.)
        assert utils_geom.Geom.npix(ninv_geom_) == ninv.size, (utils_geom.Geom.npix(ninv_geom_), ninv.size)
        sc_job.set_geometry(ninv_geom_)
        sc_job.set_nthreads(sht_threads)
        sc_job.set_triangular_alm_info(lmax_len, mmax_len)
        self.ninv_geom = ninv_geom
        self.sc_job = sc_job

        self.verbose=verbose

        self._nlevt = None
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf_tlm),
                'geom':utils_geom.Geom.hashdict(self.sc_job.geom),
                'deflection':self.ffi.hashdict(),
                'unalm':(self.lmax_sol, self.mmax_sol), 'lenalm':(self.lmax_len, self.mmax_len) }

    def _ninv_hash(self):
        assert isinstance(self.n_inv, np.ndarray)
        return clhash(self.n_inv)

    def get_ftl(self):
        if self._nlevt is None:
            nlev = 10800. / np.sqrt(np.sum(read_map(self.n_inv)) / (4.0 * np.pi)) / np.pi
            self._nlevt = nlev
            log.info('Using nlevt %.2f amin'%self._nlevt)
        n_inv_cl_t = self.b_transf_tlm ** 2  / (self._nlevt / 180. / 60. * np.pi) ** 2
        return n_inv_cl_t

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol, lmin=self.lmin_dotop)

    def apply_map(self, tmap):
        """Applies pixel inverse-noise variance maps


        """
        tmap *= self.n_inv
        if self.template is not None:
            ts = [self.template] # Hack, this is only meant for one template
            coeffs = np.concatenate(([t.dot(tmap) for t in ts]))
            coeffs = np.dot(ts[0].tniti(), coeffs)
            pmodes = np.zeros_like(tmap)
            im = 0
            for t in ts:
                t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv
            tmap -= pmodes

    def apply_alm(self, tlm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  D^t B^T N^{-1} B D, where D is lensing, B the transfer function)

        """
        # Forward lensing here
        tim = self.tim
        tim.reset_t0()
        lmax_unl = Alm.getlmax(tlm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        # glm is -tlm for spin 0 but two signs cancel
        tlm_len = self.ffi.lensgclm(tlm, self.mmax_sol, 0, self.lmax_len, self.mmax_len)
        tim.add('lensgclm fwd')

        almxfl(tlm_len, self.b_transf_tlm, self.mmax_len, inplace=True)
        tim.add('transf')

        tmap = self.sc_job.alm2map(tlm_len)
        tim.add('alm2map lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        self.apply_map(tmap)  # applies N^{-1}
        tim.add('apply ninv')

        tlm_len = self.sc_job.map2alm(tmap)
        tim.add('map2alm lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        # The map2alm is here a sum rather than integral, so geom.weights are assumed to be unity
        almxfl(tlm_len, self.b_transf_tlm, self.mmax_len, inplace=True)
        tim.add('transf')

        # backward lensing with magn. mult. here
        tlm[:]= self.ffi.lensgclm(tlm_len, self.mmax_len, 0, self.lmax_sol, self.mmax_sol, backwards=True)
        tim.add('lensgclm bwd')
        if self.verbose:
            print(tim)

    def synalm(self, unlcmb_cls:dict, cmb_phas=None):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array


        """
        tlm = synalm(unlcmb_cls['tt'], self.lmax_sol, self.mmax_sol) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(tlm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(tlm.size, self.mmax_sol), self.lmax_sol)
        tlm_len = self.ffi.lensgclm(tlm, self.mmax_sol, 0, self.lmax_len, self.mmax_len, False)
        almxfl(tlm_len, self.b_transf_tlm, self.mmax_len, True)
        # cant use here sc_job since it is using the unit weight transforms
        T = self.ninv_geom.alm2map(tlm_len, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))
        pixnoise = np.sqrt(cli(self.n_inv))
        T += default_rng().standard_normal(utils_geom.Geom.npix(self.ninv_geom)) * pixnoise
        return T

    def get_qlms(self, tlm_dat: np.ndarray, tlm_wf: np.ndarray, q_pbgeom: utils_geom.pbdGeometry, alm_wf_leg2=None):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                tlm_dat: input temperature data maps (geom must match that of the filter)
                tlm_wf: Wiener-filtered T CMB map (alm arrays)
                alm_wf_leg2: Gradient leg Wiener-filtered T CMB map (alm arrays), if different from ivf leg
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert Alm.getlmax(tlm_wf.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        if alm_wf_leg2 is None:
            d1 = self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(tlm_wf, q_pbgeom)
        else:
            assert Alm.getlmax(alm_wf_leg2.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(alm_wf_leg2.size, self.mmax_sol), self.lmax_sol)
            d1 = self._get_irestmap(tlm_dat, tlm_wf, q_pbgeom) * self._get_gtmap(alm_wf_leg2, q_pbgeom)
        G, C = q_pbgeom.geom.map2alm_spin(d1, 1, self.ffi.lmax_dlm, self.ffi.mmax_dlm, self.ffi.sht_tr, (-1., 1.))
        del d1
        fl = - np.sqrt(np.arange(self.ffi.lmax_dlm + 1, dtype=float) * np.arange(1, self.ffi.lmax_dlm + 2))
        almxfl(G, fl, self.ffi.mmax_dlm, True)
        almxfl(C, fl, self.ffi.mmax_dlm, True)
        return G, C


    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert  Alm.getlmax(tlm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        return ffi.gclm2lenmap([almxfl(tlm_wf, fl, self.mmax_sol, False), np.zeros_like(tlm_wf)], self.mmax_sol, 1, False)


    def _get_irestmap(self, tdat:np.ndarray, twf:np.ndarray, q_pbgeom:utils_geom.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """

        assert np.all(self.sc_job.geom.weight == 1.) # sum rather than integrals
        twf_len = self.ffi.lensgclm(twf, self.mmax_sol, 0, self.lmax_len, self.mmax_len, False)
        almxfl(twf_len, self.b_transf_tlm, self.mmax_len, True)
        t = tdat - self.sc_job.alm2map(twf_len)
        self.apply_map(t)
        twf_len = self.sc_job.map2alm(t)
        almxfl(twf_len, self.b_transf_tlm, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        return q_pbgeom.geom.alm2map(twf_len, self.lmax_len, self.mmax_len, self.ffi.sht_tr, (-1., 1.))


class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
        assert len(s_cls['tt']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['tt']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_ftl = ninv_filt.get_ftl() # (N_lev * transf) ** 2 basically
        if len(ninv_ftl) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.info("PRE_OP_DIAG: extending T transfer fct from lmax %s to lmax %s"%(len(ninv_ftl)-1, lmax_sol))
            assert np.all(ninv_ftl >= 0)
            nz = np.where(ninv_ftl > 0)
            spl_sq = spl(np.arange(len(ninv_ftl), dtype=float)[nz], np.log(ninv_ftl[nz]), k=2, ext='extrapolate')
            ninv_ftl = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        flmat = cli(s_cls['tt'][:lmax_sol + 1]) + ninv_ftl[:lmax_sol + 1]
        self.flmat = cli(flmat) * (s_cls['tt'][:lmax_sol +1] > 0.)
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, tlm):
        return self.calc(tlm)

    def calc(self, tlm):
        assert Alm.getsize(self.lmax, self.mmax) == tlm.size, (self.lmax, self.mmax, Alm.getlmax(tlm.size, self.mmax))
        return almxfl(tlm, self.flmat, self.mmax, False)

def calc_prep(tmap:np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            tmap: input temperature maps array of size (npix,)
            s_cls: CMB spectra dictionary (here only 'tt' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(tmap, np.ndarray)
    assert np.all(ninv_filt.sc_job.geom.weight==1.) # Sum rather than integral, hence requires unit weights
    tmapc= np.copy(tmap)
    ninv_filt.apply_map(tmapc)

    tlm = ninv_filt.sc_job.map2alm(tmapc)
    almxfl(tlm, ninv_filt.b_transf_tlm, ninv_filt.mmax_len, True)
    tlm = ninv_filt.ffi.lensgclm(tlm, ninv_filt.mmax_len, 0, ninv_filt.lmax_sol, ninv_filt.mmax_sol, backwards=True)
    almxfl(tlm, s_cls['tt'] > 0., ninv_filt.mmax_sol, True)
    return tlm


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

    def __call__(self, tlm1, tlm2):
        assert tlm1.size == Alm.getsize(self.lmax, self.mmax), (tlm1.size, Alm.getsize(self.lmax, self.mmax))
        assert tlm2.size == Alm.getsize(self.lmax, self.mmax), (tlm2.size, Alm.getsize(self.lmax, self.mmax))
        return np.sum(alm2cl(tlm1, tlm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))


class fwd_op:
    """Forward operation for temperature-only


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
        self.icltt = cli(s_cls['tt'])
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'icltt': clhash(self.icltt),
                'n_inv_filt': self.ninv_filt.hashdict()}

    def __call__(self, tlm):
        return self.calc(tlm)

    def calc(self, tlm):
        nlm = np.copy(tlm)
        self.ninv_filt.apply_alm(nlm)
        nlm += almxfl(tlm, self.icltt, self.mmax_sol, False)
        almxfl(nlm, self.icltt > 0., self.mmax_sol, True)
        return nlm