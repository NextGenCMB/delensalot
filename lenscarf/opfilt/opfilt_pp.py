"""Scarf-geometry based inverse-variance filters


"""
import numpy as np
from lenscarf.utils_hp import almxfl, Alm, alm2cl
from lenscarf.utils import timer, cli, clhash
from lenscarf import  utils_scarf
from lenscarf import remapping


class alm_filter_ninv(object):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, verbose=False):

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
            print('*** alm_filter_ninv: swtiching to same ninv_geometry but with unit weights')
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

    def apply_alm(self, elm:np.ndarray):
        # applies Y^T N^{-1} Y (now  D^t B^T N^{-1} B D)
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



class alm_filter_ninv_wl(alm_filter_ninv):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list,ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, verbose=False):
        """Scarf-based inverse-variance CMB polarization filter


        """
        super().__init__(ninv_geom, ninv, transf, unlalm_info, lenalm_info, sht_threads, verbose=verbose)
        self.ffi = ffi

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf),
                'geom':utils_scarf.Geom.hashdict(self.sc_job.geom),
                'deflection':self.ffi.hashdict(),
                'unalm':(self.lmax_sol, self.mmax_sol),
                'lenalm':(self.lmax_len, self.mmax_len) }

    def apply_alm(self, elm:np.ndarray):
        # applies Y^T N^{-1} Y (now  D^t B^T N^{-1} B D)

        # Forward lensing here
        tim = timer(True, prefix='opfilt_pp')
        lmax_unl =Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        eblm = self.ffi.lensgclm(elm, 2, lmax_out=self.lmax_len, mmax=self.mmax_sol, mmax_out=self.mmax_len)
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
        eblm = self.ffi.lensgclm(eblm[0], 2, clm=eblm[1],
                                 mmax=self.mmax_len, lmax_out=self.lmax_sol, mmax_out=self.mmax_sol, backwards=True)
        elm[:] = eblm[0]
        tim.add('lensgclm bwd')
        if self.verbose:
            print(tim)

pre_op_dense = None #FIXME: not implemented. need something like dense TT but with monopole and dipole taken out.

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
        tcl = alm2cl(elm1, elm2, self.lmax, self.mmax, None)
        return np.sum(tcl * (2 * np.arange(tcl.size) + 1))

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