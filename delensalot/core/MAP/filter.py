import os
import numpy as np

from scipy.interpolate import UnivariateSpline as spl
import healpy as hp

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.core.cg import multigrid
from delensalot.core.MAP import CG
from delensalot.core.opfilt import MAP_opfilt_iso_p as MAP_opfilt_iso_p # MAP_opfilt_iso_p_operator

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli


def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1
    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class ivf:
    def __init__(self, filter_desc):
        self.ID = filter_desc['ID']
        self.ivf_field = filter_desc['ivf_field']
        self.ivf_operator = filter_desc['ivf_operator']
        self.beam = filter_desc['beam']
        self.transfer = filter_desc["ttebl"]
        self.lm_max_ivf = filter_desc['lm_max_ivf']
        self.nlevp, self.nlevt = filter_desc['nlev']['P'], filter_desc['nlev']['T']

        self.n1elm = _extend_cl(np.array(self.transfer['e'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        self.n1blm = _extend_cl(np.array(self.transfer['b'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        

    def get_ivfreslm(self, simidx, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper, in essence it should do the following:
        if not self.ivf_field.is_cached(simidx, it):
            assert eblm_wf is not None and data is not None
            ivfreslm = -1*self.beam.act(self.ivf_operator.act(eblm_wf, spin=2, lmax_in=self.lm_max_ivf[1], lm_max=self.lm_max_ivf))
            # data[0] *= 0.+0.j
            ivfreslm += data
            almxfl(ivfreslm[0], self.n1elm * 0.5, self.lm_max_ivf[1], True)  # Factor of 1/2 because of \dagger rather than ^{-1}
            almxfl(ivfreslm[1], self.n1blm * 0.5, self.lm_max_ivf[1], True)
            self.ivf_field.cache_field(ivfreslm, simidx, it)
        return self.ivf_field.get_field(simidx, it)
    

    def update_operator(self, simidx, it):
        self.ivf_operator.set_field(simidx, it)


class wf:
    def __init__(self, filter_desc, secondaries):
        self.ID = filter_desc['ID']
        self.wf_field = filter_desc['wf_field']
        self.wf_operator = filter_desc['wf_operator']

        self.beam = filter_desc['beam']
        self.nlevp, self.nlevt = filter_desc['nlev']['P'], filter_desc['nlev']['T']

        self.lm_max_ivf = filter_desc['lm_max_ivf']
        self.lm_max_unl = filter_desc['lm_max_unl']
        self.transfer = filter_desc["ttebl"]

        self.in1el = _extend_cl(np.array(self.transfer['e'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        self.in1bl = _extend_cl(np.array(self.transfer['b'])**1, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2

        self.in2el = _extend_cl(np.array(self.transfer['e'])**2, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2
        self.in2bl = _extend_cl(np.array(self.transfer['b'])**2, self.lm_max_ivf[0]) * cli(_extend_cl(self.nlevp**2, self.lm_max_ivf[0])) * (180 * 60 / np.pi) ** 2

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.ffi = filter_desc['ffi']
        self.secondaries = secondaries


    def get_wflm(self, simidx, it, data=None):
        if not self.wf_field.is_cached(simidx, it):
            cg_sol_curr = self.wf_field.get_field(simidx, it-1)
            tpn_alm = self.calc_prep(data) # this changes lmmax to lmmax_unl via lensgclm
            mchain = CG.conjugate_gradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache_field(cg_sol_curr, simidx, it)
        return self.wf_field.get_field(simidx, it)


    def calc_prep(self, eblm):
        """cg-inversion pre-operation
            This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        lmax_len, mmax_len = self.lm_max_ivf
        lmax_sol, mmax_sol = self.lm_max_unl
        assert isinstance(eblm, np.ndarray) and eblm.ndim == 2
        assert Alm.getlmax(eblm[0].size, mmax_len) == lmax_len, (Alm.getlmax(eblm[0].size, mmax_len), lmax_len)
        eblmc = np.empty_like(eblm)
        eblmc[0] = almxfl(eblm[0], self.in1el, mmax_len, False)
        eblmc[1] = almxfl(eblm[1], self.in1bl, mmax_len, False)

        elm = self.wf_operator.act(eblmc, spin=2, lmax_in=mmax_len, lm_max=self.lm_max_unl, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY').squeeze()

        almxfl(elm, self.cls_filt['ee'] > 0., mmax_sol, True)
        return elm
    

    def fwd_op(self, elm):
        lmax_len, mmax_len = self.lm_max_ivf
        lmax_sol, mmax_sol = self.lm_max_unl

        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(elm)

        lmax_unl = Alm.getlmax(nlm.size, mmax_sol)
        assert lmax_unl == lmax_sol, (lmax_unl, lmax_sol)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = nlm.reshape((1, nlm.size))
        eblm = self.wf_operator.act(elm_2d, spin=2, lmax_in=mmax_sol, lm_max=self.lm_max_ivf, adjoint=False, backwards=False)
        almxfl(eblm[0], self.in2el, mmax_len, inplace=True)
        almxfl(eblm[1], self.in2bl, mmax_len, inplace=True)
        elm_2d = self.wf_operator.act(eblm, spin=2, lmax_in=mmax_len, lm_max=self.lm_max_unl, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')

        nlm = elm_2d.squeeze()
        nlm += almxfl(elm, iclee, mmax_sol, False)
        almxfl(nlm, iclee > 0.0, mmax_sol, True)

        return nlm


    def precon_op(self, eblm):
        """Applies the preconditioner operation for diagonal preconditioning.
        Returns:
            np.ndarray: Preconditioned alm.
        """
        assert len(self.cls_filt['ee']) > self.lm_max_unl[0], (self.lm_max_unl[0], len(self.cls_filt['ee']))
        
        ninv_fel = np.copy(self.in2el)
        # Extend transfer function to avoid preconditioning with zero (~ Gaussian beam)
        if len(ninv_fel) - 1 < self.lm_max_unl[0]:
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(self.lm_max_unl[0] + 1, dtype=float)))

        flmat = cli(self.cls_filt['ee'][:self.lm_max_unl[0] + 1]) + ninv_fel[:self.lm_max_unl[0] + 1]
        flmat = cli(flmat) * (self.cls_filt['ee'][:self.lm_max_unl[0] + 1] > 0.)

        assert Alm.getsize(self.lm_max_unl[0], self.lm_max_unl[1]) == eblm.size, (self.lm_max_unl[0], self.lm_max_unl[1], Alm.getlmax(eblm.size, self.lm_max_unl[1]))
        return almxfl(eblm, flmat, self.lm_max_unl[1], False)


    def update_operator(self, simidx, it, secondary=None, component=None):
        dfield = self.secondaries['lensing'].get_est(simidx, it, scale='d', component=component)
        if dfield.shape[0] == 1:
            dfield = [dfield[0],None]
        self.ffi = self.ffi.change_dlm(dfield, self.wf_operator.operators[0].LM_max[1])
        self.wf_operator.set_field(simidx, it, component)


    def get_template(self, simidx, it, secondary, component):
        self.wf_operator.set_field(simidx, it, secondary, component)
        estCMB = self.get_wflm(simidx, it)
        return self.wf_operator.act(estCMB, secondary=secondary)
    

    # NOTE this access the old opfilt operator
    def _get_wflm(self, simidx, it, data=None):
        def build_opfilt_iso_p(it):
            lenjob_geomlib =  get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
            ffi = deflection(lenjob_geomlib, np.zeros(shape=hp.Alm.getsize(4500, 4500)), 4500, numthreads=8, verbosity=0, epsilon=1e-8)
            dfield = self.secondary.get_est(0, it-1, scale='d')
            if dfield.shape[0] == 1:
                dfield = [dfield[0],None]
            ffi = ffi.change_dlm(dfield,self.wf_operator.LM_max[1])
            def extract():
                return {
                    'nlev_p': 1.0,
                    'ffi': ffi,
                    'transf': self.transfer['e'],
                    'unlalm_info': self.lm_max_unl, # unl
                    'lenalm_info': (4000, 4000), # ivf
                    'wee': True,
                    'transf_b': self.transfer['b'],
                    'nlev_b': 1.0,
                }
            return MAP_opfilt_iso_p.alm_filter_nlev_wl(**extract())
        self.ninv = build_opfilt_iso_p(it)
        self.opfilt = MAP_opfilt_iso_p
        self.dotop = self.ninv.dot_op()
        # if not self.wf_field.is_cached(simidx, it):
        cg_sol_curr = self.wf_field.get_field(simidx, it-1)
        mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.ninv)
        mchain.solve(cg_sol_curr, data, dot_op=self.dotop)
            # self.wf_field.cache_field(cg_sol_curr, simidx, it)
        print('the wiener-filter is: ',cg_sol_curr)
        return cg_sol_curr
        # return self.wf_field.get_field(simidx, it)