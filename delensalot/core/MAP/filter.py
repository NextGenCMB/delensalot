import os
from os.path import join as opj
import numpy as np

from delensalot.utils import read_map
from delensalot.core.cg import cd_solve, cd_monitors, multigrid
# from delensalot.core.cg_simple import cd_solve, cd_monitors, multigrid
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
    def __init__(self, filter_desc, secondary):
        self.ID = filter_desc['ID']
        self.ivf_field = filter_desc['ivf_field']
        self.wf_field = filter_desc['wf_field']
        self.ivf_operator = filter_desc['ivf_operator']
        self.wf_operator = filter_desc['wf_operator']
        self.Ninv = read_map(filter_desc['Ninv_desc'])
        self.beam = filter_desc['beam']
        self.nlevp, self.nlevt = filter_desc['nlev']['P'], filter_desc['nlev']['T']
        self.filter_desc = filter_desc

        self.lm_max_ivf = filter_desc['lm_max_ivf']
        self.transfer = filter_desc["ttebl"]
        self.transfere = _extend_cl(filter_desc["ttebl"]['e'], self.lm_max_ivf[0])
        self.transferb = _extend_cl(filter_desc["ttebl"]['b'], self.lm_max_ivf[0])
        # TODO make sure ninv_desc[0][x] are actually the right inverse noise levels
        self.n1elm = _extend_cl(np.array(self.transfer['e'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        self.n1blm = _extend_cl(np.array(self.transfer['b'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        self.datmaps = None
        self.data = None

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.secondary = secondary


    # FIXME the following is to be replaced with new routines, and is only needed for the current cg multigrid_chain implementation
    def build_opfilt_iso_p(self, it):
        lenjob_geomlib =  get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
        ffi = deflection(lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(4500, 4500)), 4500, numthreads=8, verbosity=0, epsilon=1e-8)
        dfield = self.secondary.get_est(0, it-1, scale='d')
        if dfield.shape[0] == 1:
            dfield = [dfield[0],None]
        ffi = ffi.change_dlm(dfield, 3000)
        def extract():
            return {
                'nlev_p': 1.0,
                'ffi': ffi,
                'transf': self.transfer['e'],
                'unlalm_info': (4000, 4000), # unl
                'lenalm_info': self.lm_max_ivf, # ivf
                'wee': True,
                'transf_b': self.transfer['b'],
                'nlev_b': 1.0,
            }
        return MAP_opfilt_iso_p.alm_filter_nlev_wl(**extract())


    def update_operator(self, simidx, it):
        self.ivf_operator.set_field(simidx, it)
        self.wf_operator.set_field(simidx, it)


    def get_wflm(self, simidx, it, data=None):
        self.ninv = self.build_opfilt_iso_p(it)
        self.opfilt = MAP_opfilt_iso_p
        self.dotop = self.ninv.dot_op()
        if not self.wf_field.is_cached(simidx, it):
            cg_sol_curr = self.wf_field.get_field(simidx, it-1)
            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.ninv)
            mchain.solve(cg_sol_curr, data, dot_op=self.dotop)
            self.wf_field.cache_field(cg_sol_curr, simidx, it)
        return self.wf_field.get_field(simidx, it)
    

    def get_wflm_new(self, simidx, it, data=None):
        def calc_prep(eblm, s_cls, ninv_filt):
            """cg-inversion pre-operation
                This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
                Args:
                    eblm: input data polarisation elm and blm
                    s_cls: CMB spectra dictionary (here only 'ee' key required)
                    ninv_filt: inverse-variance filtering instance
            """
            assert isinstance(eblm, np.ndarray) and eblm.ndim == 2
            assert Alm.getlmax(eblm[0].size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(eblm[0].size, ninv_filt.mmax_len), ninv_filt.lmax_len)
            eblmc = np.empty_like(eblm)
            eblmc[0] = almxfl(eblm[0], ninv_filt.inoise_1_elm, ninv_filt.mmax_len, False)
            eblmc[1] = almxfl(eblm[1], ninv_filt.inoise_1_blm, ninv_filt.mmax_len, False)
            elm = ninv_filt.ffi.lensgclm(eblmc, ninv_filt.mmax_len, 2, ninv_filt.lmax_sol,ninv_filt.mmax_sol,
                                            backwards=True, out_sht_mode='GRAD_ONLY').squeeze()
            almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, True)
            return elm
        
        if not self.wf_field.is_cached(simidx, it):
            cg_sol_curr = self.wf_field.get_field(simidx, it-1)
            mchain = multigrid.multigrid_chain(self.chain_descr, self.cls_filt, self.ninv)
            mchain.solve(cg_sol_curr, data, dot_op=self.dotop)
            self.wf_field.cache_field(cg_sol_curr, simidx, it)
        return self.wf_field.get_field(simidx, it)


    def get_ivfreslm(self, simidx, it, data, eblm_wf):
        # NOTE this is eq. 21 of the paper, in essence it should do the following:
        if not self.ivf_field.is_cached(simidx, it):
            ivflm = -1*self.beam.act(self.ivf_operator.act(eblm_wf, spin=2))
            # data[0] *= 0.+0.j
            ivflm += data
            almxfl(ivflm[0], self.n1elm * 0.5, self.lm_max_ivf[1], True)  # Factor of 1/2 because of \dagger rather than ^{-1}
            almxfl(ivflm[1], self.n1blm * 0.5, self.lm_max_ivf[1], True)
            # ivflm = self.beam.act(ivflm, adjoint=True)
            self.ivf_field.cache_field(ivflm, simidx, it)
        return self.ivf_field.get_field(simidx, it)