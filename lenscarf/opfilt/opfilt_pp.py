"""Scarf-geometry based inverse-variance filters


"""
import numpy as np
from lenscarf.utils_hp import almxfl, Alm
from lenscarf.utils import timer
from lenscarf import  utils_scarf
from lenscarf import remapping

class alm_filter_ninv_wl:
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list,ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int):
        """Scarf-based inverse-variance CMB polarization filter


        """
        npix = utils_scarf.Geom.npix(ninv_geom)
        assert np.all([ni.size == npix for ni in ninv])

        self.ffi= ffi
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
        sc_job.set_geometry(ninv_geom)
        sc_job.set_nthreads(sht_threads)
        sc_job.set_triangular_alm_info(lmax_len, mmax_len)
        self.sc_job = sc_job

    def hashdict(self):
        #FIXME:
        return {}

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
        tim.add('alm2map_spin lmax %s nrings %s'%(self.lmax_len, self.sc_job.geom.get_nrings()))

        self.apply_map(qumap)  # applies N^{-1}
        tim.add('apply ninv')

        eblm = self.sc_job.map2alm_spin(qumap, 2)
        tim.add('map2alm_spin lmax %s nrings %s'%(self.lmax_len, self.sc_job.geom.get_nrings()))

        # FIXME: npix / 4 pi is wrong here
        almxfl(eblm[0], self.b_transf * (qumap[0].size / (4 * np.pi)), self.mmax_len, inplace=True) # factor npix / 4pi
        almxfl(eblm[1], self.b_transf * (qumap[0].size / (4 * np.pi)), self.mmax_len, inplace=True)
        tim.add('transf')

        # backward lensing with magn. mult. here
        eblm = self.ffi.lensgclm(eblm[0], 2, clm=eblm[1],
                                 mmax=self.mmax_len, lmax_out=self.lmax_sol, mmax_out=self.mmax_sol, backwards=True)
        elm[:] = eblm[0]
        tim.add('lensgclm bwd')
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