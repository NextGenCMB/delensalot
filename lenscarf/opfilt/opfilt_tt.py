"""Scarf-geometry based inverse-variance filter, without any lensing remapping


"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from lenscarf.utils_hp import almxfl, Alm, alm2cl
from lenscarf.utils import timer, cli, clhash, read_map
from lenscarf import  utils_scarf
from lenscarf.opfilt import tmodes_ninv as tni
from scipy.interpolate import UnivariateSpline as spl


class alm_filter_ninv(object):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, tpl:tni.template_tfilt or None=None, verbose=False):
        r"""CMB inverse-variance and Wiener filtering instance, to use for cg-inversion

            Args:
                ninv_geom: scarf geometry for the inverse-pixel-noise variance SHTs
                ninv: list of inverse-pixel noise variance maps (itself can be (a list of) string, or array, or ...)
                transf: CMB transfer function (assumed to be the same in E and B)
                unlalm_info: tuple of int, lmax and mmax of unlensed CMB
                lenalm_info: tuple of int, lmax and mmax of lensed CMB
                sht_threads: number of threads for scarf SHTs
                verbose: some printout if set, defaults to False


        """
        self.n_inv = read_map(ninv)
        self.b_transf = transf

        lmax_unl, mmax_unl = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = len(transf) - 1
        self.lmax_len = min(lmax_transf, lmax_len)
        self.mmax_len = min(mmax_len, lmax_transf)
        self.lmax_sol = lmax_unl
        self.mmax_sol = min(lmax_unl, mmax_unl)

        sc_job = utils_scarf.scarfjob()
        if not np.all(ninv_geom.weight == 1.): # All map2alm's here will be sums rather than integrals...
            log.info('*** alm_filter_ninv: switching to same ninv_geometry but with unit weights')
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

        self._nlevt = None
        self.tim = timer(True, prefix='opfilt')

        self.template = tpl # here just one template allowed



    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf),
                'geom':utils_scarf.Geom.hashdict(self.sc_job.geom),
                'unalm':(self.lmax_sol, self.mmax_sol),
                'lenalm':(self.lmax_len, self.mmax_len) }

    def _ninv_hash(self):
        return clhash(self.n_inv)

    def get_ftl(self):
        if self._nlevt is None:
            nlev_ftl =  10800. / np.sqrt(np.sum(read_map(self.n_inv)) / (4.0 * np.pi)) / np.pi
            self._nlevt = nlev_ftl
        return self.b_transf ** 2 / (self._nlevt/ 180. / 60. * np.pi) ** 2

    def apply_alm(self, tlm:np.ndarray):
        """Applies operator B^T N^{-1} B

        """
        assert self.lmax_sol == self.lmax_len, (self.lmax_sol, self.lmax_len) # not implemented wo lensing
        assert self.mmax_sol == self.mmax_len, (self.mmax_sol, self.mmax_len)

        tim = timer(True, prefix='opfilt_pp')
        lmax_unl = Alm.getlmax(tlm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        almxfl(tlm, self.b_transf, self.mmax_len, inplace=True)
        tim.add('transf')

        tmap = self.sc_job.alm2map(tlm)
        tim.add('alm2map_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        self.apply_map(tmap)  # applies N^{-1}
        tim.add('apply ninv')

        tlm[:] = self.sc_job.map2alm(tmap)
        tim.add('map2alm_spin lmax %s mmax %s nrings %s'%(self.lmax_len, self.mmax_len, self.sc_job.geom.get_nrings()))

        # The map2alm is here a sum rather than integral, so geom.weights are assumed to be unity
        almxfl(tlm, self.b_transf, self.mmax_len, inplace=True)
        tim.add('transf')
        if self.verbose:
            print(tim)

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


    def _get_gtmap(self, tlm_wf:np.ndarray, q_pbgeom:utils_scarf.pbdGeometry):
        """Wiener-filtered gradient leg to feed into the QE


            :math:`\sum_{lm} (-Tlm) sqrt(l (l+1)) _1 Ylm(n)


        """
        assert  Alm.getlmax(tlm_wf.size, self.mmax_sol)== self.lmax_sol, ( Alm.getlmax(tlm_wf.size, self.mmax_sol), self.lmax_sol)
        fl = -np.sqrt(np.arange(self.lmax_sol + 1) * np.arange(1, self.lmax_sol + 2))
        gclm = (almxfl(tlm_wf, fl, self.mmax_sol, False), np.zeros_like(tlm_wf))
        return q_pbgeom.geom.alm2map_spin(gclm, 1 , self.lmax_sol, self.mmax_sol, self.sc_job.nthreads, [-1., 1.])

    def _get_irestmap(self, tdat:np.ndarray, twf:np.ndarray, q_pbgeom:utils_scarf.pbdGeometry):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """

        assert np.all(self.sc_job.geom.weight == 1.) # sum rather than integrals
        twf_len = np.copy(twf)
        almxfl(twf_len, self.b_transf, self.mmax_len, True)
        t = tdat - self.sc_job.alm2map(twf_len)
        self.apply_map(t)
        twf_len = self.sc_job.map2alm(t)
        almxfl(twf_len, self.b_transf, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        return q_pbgeom.geom.alm2map(twf_len, self.lmax_len, self.mmax_len, self.sc_job.nthreads, (-1., 1.))

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
    assert np.all(ninv_filt.sc_job.geom.weight==1.) # Sum rather than integral, hence requires unit weights
    tmap= np.copy(maps)
    ninv_filt.apply_map(tmap)
    tlm = ninv_filt.sc_job.map2alm(tmap)
    lmax_tr = len(ninv_filt.b_transf) - 1
    almxfl(tlm, ninv_filt.b_transf * (s_cls['tt'][:lmax_tr+1] > 0.), ninv_filt.mmax_len, inplace=True)
    return tlm

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass

class pre_op_diag:
    """Cg-inversion diagonal preconditioner

    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
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
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
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