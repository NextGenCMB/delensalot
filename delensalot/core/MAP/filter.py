import os
from os.path import join as opj
import numpy as np

from delensalot.core import cachers
from delensalot.utils import read_map
from delensalot.core.cg_simple import cd_solve, cd_monitors, multigrid
from delensalot.core.opfilt import MAP_opfilt_iso_p

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

from . import operator

import healpy as hp
from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class base:
    def __init__(self, filter_desc):
        self.ID = filter_desc['ID']
        self.ivf_field = filter_desc['ivf_field']
        self.WF_field = filter_desc['WF_field']
        self.ivf_operator = filter_desc['ivf_operator']
        self.WF_operator = filter_desc['WF_operator']
        self.Ninv = read_map(filter_desc['Ninv_desc'])
        self.beam = filter_desc['beam']

        self.lm_max_ivf = filter_desc['lm_max_ivf']
        self.transfer = filter_desc["ttebl"]
        # TODO make sure ninv_desc[0][x] are actually the right inverse noise levels
        self.n1elm = _extend_cl(np.array(self.transfer['e'])**1, self.lm_max_ivf[0]) * _extend_cl(np.array(filter_desc['Ninv_desc'][1][0][0])**2, self.lm_max_ivf[0]) * (180 * 60 / np.pi) ** 2
        self.n1blm = _extend_cl(np.array(self.transfer['b'])**1, self.lm_max_ivf[0]) * _extend_cl(np.array(filter_desc['Ninv_desc'][1][0][0])**2, self.lm_max_ivf[0]) * (180 * 60 / np.pi) ** 2

        self.datmaps = None
        self.data = None

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']


        # FIXME the following is to be replaced with new routines, and is only needed for the current cg multigrid_chain implementation
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
        # NOTE this is eq. 21 of the paper, in essence it should do the following:
        #     this is ivf_operator
        # ---------------------------
        #     ebwf = self.ffi.lensgclm(np.atleast_2d(eblm_wf), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        #     almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
        #     almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
        # ---------------------------
        #     ebwf += eblm_dat
        
        #     this is self.Ninv*ivflm
        # ---------------------------
        #     almxfl(ebwf[0], self.inoise_1_elm * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        #     almxfl(ebwf[1], self.inoise_1_blm * 0.5,            self.mmax_len, True)
        # ---------------------------
        #     return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)
        if not self.ivf_field.is_cached(simidx, it):
            ivflm = self.beam.act(self.ivf_operator.act(eblm_wf, spin=2))
            ivflm -= data
            almxfl(ivflm[0], self.n1elm * 0.5, self.lm_max_ivf[1], True)  # Factor of 1/2 because of \dagger rather than ^{-1}
            almxfl(ivflm[1], self.n1blm * 0.5, self.lm_max_ivf[1], True)
            ivflm = self.beam.act(ivflm, adjoint=True)

            self.ivf_field.cache_field(ivflm, simidx, it)
        return self.ivf_field.get_field(simidx, it)

        # resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
        # resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
        # self._get_irespmap(eblm_dat, elm_wf, q_pbgeom, map_out=resmap_r) # inplace onto resmap_c and resmap_r