import os
from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.core.cg_simple import cd_solve, cd_monitors, multigrid
from delensalot.core.opfilt import MAP_opfilt_iso_p

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy

from . import operator

import healpy as hp
from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

class base:
    def __init__(self, filter_desc):
        self.ID = filter_desc['ID']
        self.ivf_field = filter_desc['ivf_field']
        self.WF_field = filter_desc['WF_field']
        self.ivf_operator = filter_desc['ivf_operator']
        self.WF_operator = filter_desc['WF_operator']
        self.Ninv = filter_desc['Ninv_desc']
        self.beam = filter_desc['beam']

        self.datmaps = None
        self.data = None

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']


        # FIXME the following is to be replaced with new routines
        def build_opfilt_iso_p():
            lenjob_geomlib =  get_geom(('thingauss', {'lmax': 3500, 'smax': 3}))
            ffi = deflection(lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(3000, 3000)), 3000, numthreads=8, verbosity=0, epsilon=1e-8)  
            def extract():
                return {
                    'nlev_p': 0.6,
                    'ffi': ffi,
                    'transf': filter_desc["ttebl"]['e'],
                    'unlalm_info': (3500, 3500),
                    'lenalm_info': (3500, 3500),
                    'wee': True,
                    'transf_b': filter_desc["ttebl"]['b'],
                    'nlev_b': 0.6,
                }
            return MAP_opfilt_iso_p.alm_filter_nlev_wl(**extract())
        self.ninv = build_opfilt_iso_p()
        self.opfilt = MAP_opfilt_iso_p
        self.dotop = self.ninv.dot_op()


    def update_operator(self, simidx, it):
        self.ivf_operator.set_field(simidx, it)
        self.WF_operator.set_field(simidx, it)


    def get_WF(self, data, simidx, it):
        if not self.WF_field.is_cached(simidx, it):
            cg_sol_curr = self.WF_field.get_field(simidx, it-1)
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.ninv)
            mchain.solve(cg_sol_curr, data, dot_op=self.dotop)
            self.WF_field.cache_field(cg_sol_curr, simidx, it)
        return self.WF_field.get_field(simidx, it)


    def get_ivf(self, data, eblm_wf, simidx, it):
        if not self.ivf_field.is_cached(simidx, it):
            print(self.beam.act(self.Ninv*data, is_adjoint=True))
            print(self.beam.act(self.ivf_operator.act(eblm_wf)))
            # NOTE self.Ninv*data is an almxfl with inverse noise level scaling.
            # NOTE do i want to operatorize this?
            ivflm = self.beam.act(self.Ninv*data, is_adjoint=True) - self.beam.act(self.ivf_operator.act(eblm_wf))
            self.ivf_field.cache_field(ivflm, simidx, it)
        return self.ivf_field.get_field(self.ivflm.format(simidx, it))
    

        # resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
        # resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        # self._get_irespmap(eblm_dat, elm_wf, q_pbgeom, map_out=resmap_r) # inplace onto resmap_c and resmap_r
        # # This is the actual ivf. Need to replace operations by operator.
        # def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray, q_pbgeom:pbdGeometry, map_out=None):
        #     """Builds inverse variance weighted map to feed into the QE
        #         :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`
        #     """
        #     assert len(eblm_dat) == 2
        #     ebwf = self.ffi.lensgclm(np.atleast_2d(eblm_wf), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        #     almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
        #     almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
        #     ebwf += eblm_dat
        #     almxfl(ebwf[0], self.inoise_1_elm * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        #     almxfl(ebwf[1], self.inoise_1_blm * 0.5,            self.mmax_len, True)
        #     return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)


# class dot_op:
#     def __init__(self, lmax: int, mmax: int or None, lmin=0):
#         """scalar product operation for cg inversion

#             Args:
#                 lmax: maximum multipole defining the alm layout
#                 mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


#         """
#         if mmax is None or mmax < 0: mmax = lmax
#         self.lmax = lmax
#         self.mmax = min(mmax, lmax)
#         self.lmin = int(lmin)

#     def __call__(self, elm1, elm2):
#         assert elm1.size == Alm.getsize(self.lmax, self.mmax), (elm1.size, Alm.getsize(self.lmax, self.mmax))
#         assert elm2.size == Alm.getsize(self.lmax, self.mmax), (elm2.size, Alm.getsize(self.lmax, self.mmax))
#         return np.sum(alm2cl(elm1, elm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))

    