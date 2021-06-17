"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping


"""
import numpy as np
from lenscarf.utils_hp import almxfl, Alm
from lenscarf.utils import timer, clhash
from lenscarf import  utils_scarf
from lenscarf import remapping
from lenscarf.opfilt import opfilt_pp

dot_op = opfilt_pp.dot_op
fwd_op = opfilt_pp.fwd_op
apply_fini = opfilt_pp.apply_fini
pre_op_dense = None # not implemented

class alm_filter_ninv_wl(opfilt_pp.alm_filter_ninv):
    def __init__(self, ninv_geom:utils_scarf.Geometry, ninv:list, ffi:remapping.deflection, transf:np.ndarray,
                 unlalm_info:tuple, lenalm_info:tuple, sht_threads:int, verbose=False):
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


        """
        super().__init__(ninv_geom, ninv, transf, unlalm_info, lenalm_info, sht_threads, verbose=verbose)
        self.ffi = ffi

    def hashdict(self):
        return {'ninv':self._ninv_hash(), 'transf':clhash(self.b_transf),
                'geom':utils_scarf.Geom.hashdict(self.sc_job.geom),
                'deflection':self.ffi.hashdict(),
                'unalm':(self.lmax_sol, self.mmax_sol), 'lenalm':(self.lmax_len, self.mmax_len) }

    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  D^t B^T N^{-1} B D, where D is lensing, B the transfer function)

        """
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


def calc_prep(maps:list or np.ndarray, s_cls:dict, ninv_filt:alm_filter_ninv_wl):
    """cg-inversion pre-operation  (D^t B^t N^{-1} X^{dat})

        Args:
            maps: input polarisation maps
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert np.all(ninv_filt.sc_job.geom.weight==1.) # Sum rather than integral, hence requires unit weights
    qumap= [np.copy(maps[0]), np.copy(maps[1])]
    ninv_filt.apply_map(qumap)

    elm, blm = ninv_filt.sc_job.map2alm_spin(qumap, 2)
    almxfl(elm, ninv_filt.b_transf, ninv_filt.mmax_len, inplace=True)
    almxfl(blm, ninv_filt.b_transf, ninv_filt.mmax_len, inplace=True)
    elm, blm = ninv_filt.ffi.lensgclm(elm, 2, ninv_filt.lmax_sol,
                                      clm=blm, backwards=True, mmax=ninv_filt.mmax_len,  mmax_out=ninv_filt.mmax_sol)
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, inplace=True)
    return elm
