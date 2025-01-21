import numpy as np

from delensalot.core import cachers

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

from lenspyx.remapping.deflection_028 import rtype, ctype

from . import filter

import healpy as hp


"""
fields:
    gradient (which is klm)
    gradient increment from curvature
    total gradient increment from previous
    2-term gradient increment from previous (no prior change)
    we can build all gradients from the increments: dont store gradient, but can cache.

    either build total or individual gradients from increments, or get new from iteration
        for building, i need the initial gradient (klm_it0), which is stored, and all increments
        for new, i need the field increments
"""

class base:
    def __init__(self, gradient_desc, filter, simidx):
        self.ID = gradient_desc['ID']
        self.field = gradient_desc['field']
        self.gfield = gradient_desc['gfield']
        self.gradient_operator = gradient_desc['gradient_operator']
        self.filter = filter
        self.simidx = simidx
        self.noisemodel_coverage = gradient_desc['noisemodel_coverage']
        self.estimator_key = gradient_desc['estimator_key']
        self.simulationdata = gradient_desc['simulationdata']
        self.lm_max_ivf = gradient_desc['lm_max_ivf']
        self.lm_max_qlm = gradient_desc['lm_max_qlm']
        self.ffi = gradient_desc['ffi']


    def get_gradient_total(self, it, component=None):
        # if already cached, load it, otherwise calculate the new one
        if self.gfield.cacher.is_cached(self.gfield.total_fns.format(idx=self.simidx, it=it)):
            print('total is cached at iter, ', it)
            return self.gfield.get_total(self.simidx, it, component)
        else:
            print("building total {} gradient for iter {} ".format(self.ID, it+1))
            g = 0
            g += self.get_gradient_prior(it)
            g += self.get_gradient_meanfield(it)
            g += self.get_gradient_quad(it)
            return g


    def get_gradient_quad(self, it, component=None):
        assert 0, "subclass this"


    def get_gradient_meanfield(self, it, component=None):
        return self.gfield.get_meanfield(self.simidx, it, component)


    def get_gradient_prior(self, it, component=None):
        return self.gfield.get_prior(self.simidx, it, component)


    def get_WF(self):
        curr_iter = self.maxiterdone()
        return self.filter.get_WF(curr_iter)
    

    def get_ivf(self):
        curr_iter = self.maxiterdone()
        XWF = self.filter.get_WF(curr_iter)
        return self.filter.get_ivf(curr_iter, XWF)


    def update_operator(self, simidx, it):
        print('updating fields for operator in gradient ', self.ID)
        self.filter.update_operator(simidx, it)
        self.gradient_operator.set_field(simidx, it)


    def update_gradient(self):
        pass


    def get_data(self, lm_max):
        if self.noisemodel_coverage == 'isotropic':
            # dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if self.estimator_key in ['p_p', 'p_eb', 'peb', 'p_be']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
            if self.k in ['pee']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)[0]
            elif self.k in ['ptt']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)
            elif self.k in ['p']:
                EBobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
                Tobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)         
                ret = np.array([Tobs, *EBobs])
                return ret
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.simidx), dtype=float)
            else:
                assert 0, 'implement if needed'



class lensing(base):

    def __init__(self, gradient_desc, filter, simidx):
        super().__init__(gradient_desc, filter, simidx)
    

    def get_gradient_quad_new(self, it, component=None):
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        # y(res,1)
        # almxfl(res)
        if not self.gfield.quad_is_cached(self.simidx, it):
            data = self.get_data(self.lm_max_ivf)
            XWF = self.filter.get_WF(data, self.simidx, it)
            ivf = self.filter.get_ivf(data, XWF, self.simidx, it)
            
            # TODO cast ivf to map space with lm_max_len, but how does conj() work then?
            ivfmap = self.ffi.geom.synthesis(ivf, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
            ivfmapconj = self.ffi.geom.synthesis(ivf.conj(), 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
            
            # TODO cast gradientoperator * XWF with lm_max_unl, but how does conj() work then?
            xwfglm = self.gradient_operator.act(XWF, spin=2)
            xwfmap = self.ffi.geom.synthesis(xwfglm, 3, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
            xwfmapconj = self.ffi.geom.synthesis(xwfglm.conj(), 1, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
            
            # NOTE This is an actual calculation inside this class
            qlms = ivfmapconj * xwfmap
            qlms -= ivfmap * xwfmapconj
            
            # TODO at last, cast qlms to alm space with lm_max_qlm
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 1, self.lm_max_qlm[0], self.lm_max_qlm[1], self.ffi.sht_tr)
            fl = -np.sqrt(np.arange(self.lm_max_qlm[0] + 1, dtype=float) * np.arange(1, self.lm_max_qlm[0] + 2))
            
            almxfl(qlms[0], fl, self.lm_max_qlm[1], True)
            almxfl(qlms[1], fl, self.lm_max_qlm[1], True)
            self.gfield.cache_quad(qlms, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)
    

    def get_gradient_quad(self, it, component=None):
        def _extend_cl(cl, lmax):
            if np.isscalar(cl):
                return np.ones(lmax + 1, dtype=float) * cl
            ret = np.zeros(lmax + 1, dtype=float)
            ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
            return ret
        self.mmax_sol = 3500
        self.lmax_len = self.lm_max_ivf[0]
        self.mmax_len = self.lm_max_ivf[1]
        nlev_elm = _extend_cl(self.filter.nlevp, self.lmax_len)
        nlev_blm = _extend_cl(self.filter.nlevp, self.lmax_len)
        self.inoise_2_elm  = _extend_cl(self.filter.transfer['e'] ** 2, self.lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_elm  = _extend_cl(self.filter.transfer['b'] ** 1, self.lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2

        self.inoise_2_blm = _extend_cl(self.filter.transfer['e'] ** 2, self.lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_blm = _extend_cl(self.filter.transfer['b'] ** 1, self.lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2

        self.transf_elm  = self.filter.n1elm # _extend_cl(transf_elm, self.lmax_len)
        self.transf_blm  = self.filter.n1blm # self.transf_blm  = _extend_cl(transf_blm, self.lmax_len)

        dfield = self.field.get_klm(self.simidx, it)
        self.ffi = self.ffi.change_dlm(dfield, self.lm_max_qlm[1])

        def _get_irespmap(eblm_dat:np.ndarray, eblm_wf:np.ndarray, map_out=None):
            ebwf = self.ffi.lensgclm(np.atleast_2d(eblm_wf), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
            almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
            almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
            ebwf += eblm_dat
            almxfl(ebwf[0], self.inoise_1_elm * 0.5, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
            almxfl(ebwf[1], self.inoise_1_blm * 0.5, self.mmax_len, True)
            return self.ffi.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)

        def _get_gpmap(elm_wf:np.ndarray, spin:int):
            lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)
            i1, i2 = (2, -1) if spin == 1 else (-2, 3)
            fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
            fl[:spin] *= 0.
            fl = np.sqrt(fl)
            elm = np.atleast_2d(almxfl(elm_wf, fl, self.mmax_sol, False))
            return self.ffi.gclm2lenmap(elm, self.mmax_sol, spin, False)
        
        if not self.gfield.quad_is_cached(self.simidx, it):
            data = self.get_data(self.lm_max_ivf)
            XWF = self.filter.get_WF(data, self.simidx, it)
            print(XWF.shape, 'XWF shape')
            # ivf = self.filter.get_ivf(data, XWF, self.simidx, it)

            elm_wf = XWF 
            resmap_c = np.empty((16544332,), dtype=XWF.dtype)
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            _get_irespmap(data, elm_wf, map_out=resmap_r) # inplace onto resmap_c and resmap_r

            gcs_r = _get_gpmap(elm_wf, 3)  # 2 pos.space maps, uses then complex view onto real array
            gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
            gcs_r = _get_gpmap(elm_wf, 1)
            gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
            del resmap_c, resmap_r, gcs_r
            lmax_qlm, mmax_qlm = self.ffi.lmax_dlm, self.ffi.mmax_dlm
            gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
            gc = self.ffi.geom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, self.ffi.sht_tr)
            del gc_r, gc_c
            fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
            almxfl(gc[0], fl, mmax_qlm, True)
            almxfl(gc[1], fl, mmax_qlm, True)
            self.gfield.cache_quad(gc, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)
    

class birefringence(base):

    def __init__(self, gradient_desc, filter, simidx):
        super().__init__(gradient_desc, filter, simidx)
    

    def get_gradient_quad(self, it, component=None):
        if not self.gfield.quad_is_cached(self.simidx, it):
            data = self.get_data(self.lm_max_ivf)
            XWF = self.filter.get_WF(data, self.simidx, it)
            ivf = self.filter.get_ivf(data, XWF, self.simidx, it)
            
            ivfmap = self.ffi.geom.synthesis(ivf, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)

            xwfglm = self.gradient_operator.act(XWF, spin=2)
            xwfmap = self.ffi.geom.synthesis(xwfglm, 2, self.lm_max_ivf[0], self.lm_max_ivf[1], self.ffi.sht_tr)
 
            qlms = -4 * ( ivfmap[0] * xwfmap[1] - ivfmap[1] * xwfmap[0] )
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.lm_max_qlm[0], self.lm_max_qlm[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms, self.simidx, it=it)
        return self.gfield.get_quad(self.simidx, it, component)