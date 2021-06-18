"""Scarf-geometry based inverse-variance filter, without any lensing remapping


"""
import numpy as np
from lenscarf.utils_hp import almxfl, Alm, alm2cl
from lenscarf.utils import timer, cli, clhash, read_map
from lenscarf import  utils_scarf


class alm_filter_ninv(object):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, verbose=False):
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

        npix = utils_scarf.Geom.npix(ninv_geom)
        assert np.all([ni.size == npix for ni in ninv])

        self.n_inv = ninv
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
        self.sc_job = sc_job

        self.verbose=verbose

        self._nlevp = None

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf),
                'geom':utils_scarf.Geom.hashdict(self.sc_job.geom),
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
        fel = self.b_transf ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        fbl = self.b_transf ** 2 / (self._nlevp/ 180. / 60. * np.pi) ** 2
        return fel, fbl

    def apply_alm(self, elm:np.ndarray):
        """Applies operator B^T N^{-1} B

        """
        assert self.lmax_sol == self.lmax_len, (self.lmax_sol, self.lmax_len) # not implemented wo lensing
        assert self.mmax_sol == self.mmax_len, (self.mmax_sol, self.mmax_len)

        tim = timer(True, prefix='opfilt_pp')
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        eblm = np.array([elm, np.zeros_like(elm)])
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
        elm[:] = eblm[0]
        if self.verbose:
            print(tim)

    def apply_map(self, qumap):
        """Applies pixel inverse-noise variance maps


        """
        if len(self.n_inv) == 1:  #  QQ = UU
            qumap *= self.n_inv[0]
        elif len(self.n_inv) == 3:  # QQ, QU, UU
            qmap, umap = qumap
            qmap_copy = qmap.copy()
            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap
            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy
            del qmap_copy
        else:
            assert 0

pre_op_dense = None #FIXME: not implemented. need something like dense TT but with monopole and dipole taken out.

def calc_prep(maps:list or np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            maps: input polarisation maps
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert ninv_filt.lmax_sol == ninv_filt.lmax_len, (ninv_filt.lmax_sol, ninv_filt.lmax_len)  # not implemented wo lensing
    assert ninv_filt.mmax_sol == ninv_filt.mmax_len, (ninv_filt.mmax_sol, ninv_filt.mmax_len)
    assert np.all(ninv_filt.sc_job.geom.weight==1.) # Sum rather than integral, hence requires unit weights
    qumap= [np.copy(maps[0]), np.copy(maps[1])]
    ninv_filt.apply_map(qumap)
    elm, blm = ninv_filt.sc_job.map2alm_spin(qumap, 2)
    almxfl(elm, ninv_filt.b_transf, ninv_filt.mmax_len, inplace=True)
    almxfl(blm, ninv_filt.b_transf, ninv_filt.mmax_len, inplace=True)
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, inplace=True)
    return elm

def apply_fini(*args, **kwargs):
    """cg-inversion post-operation

        If nothing output is Wiener-filtered CMB


    """
    pass


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

    def __call__(self, elm1, elm2):
        assert elm1.size == Alm.getsize(self.lmax, self.mmax), (elm1.size, Alm.getsize(self.lmax, self.mmax))
        assert elm2.size == Alm.getsize(self.lmax, self.mmax), (elm1.size, Alm.getsize(self.lmax, self.mmax))
        return np.sum(alm2cl(elm1, elm2, self.lmax, self.mmax, None) * (2 * np.arange(self.lmax + 1) + 1))

class fwd_op:
    """Forward operation for polarization-only, no primordial B power cg filter


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_ninv):
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
        return almxfl(nlm + almxfl(elm, self.iclee, self.mmax_sol, False), self.iclee > 0., self.mmax_sol, False)