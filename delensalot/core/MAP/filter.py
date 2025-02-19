from os.path import join as opj
import numpy as np

from scipy.interpolate import UnivariateSpline as spl
import healpy as hp

from delensalot.core.MAP import CG
from delensalot.core.opfilt import MAP_opfilt_iso_p as MAP_opfilt_iso_p

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli

from delensalot.core.MAP import field

filterfield_desc = lambda ID, libdir: {
    "ID": ID,
    "libdir": opj(libdir),
    "fns": f"{ID}_simidx{{idx}}_it{{it}}",}

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
        self.ivf_operator = filter_desc['ivf_operator']
        self.libdir = filter_desc['libdir']
        self.beam_operator = filter_desc['beam_operator']
        self.inoise_operator = filter_desc['inoise_operator']
        self.ivf_field = field.filter(filterfield_desc('ivf', self.libdir))


    def get_ivfreslm(self, simidx, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper, in essence it should do the following:
        if not self.ivf_field.is_cached(simidx, it):
            assert eblm_wf is not None and data is not None
            ivfreslm = self.ivf_operator.act(eblm_wf, spin=2)
            ivfreslm = -1*self.beam_operator.act(ivfreslm)
            ivfreslm += data
            ivfreslm = self.inoise_operator.act(0.5*ivfreslm, adjoint=False)
            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            self.ivf_field.cache_field(ivfreslm, simidx, it)
        return self.ivf_field.get_field(simidx, it)
    

    def update_operator(self, simidx, it):
        self.ivf_operator.set_field(simidx, it)


class wf:
    def __init__(self, filter_desc):
        self.wf_operator = filter_desc['wf_operator']
        self.libdir = filter_desc['libdir']
        self.beam_operator = filter_desc['beam_operator']
        self.inoise_operator = filter_desc['inoise_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        filterfield_desc = lambda ID: {
            "ID": ID,
            "libdir": opj(self.libdir),
            "fns": f"{ID}_simidx{{idx}}_it{{it}}",
        }
        self.wf_field = field.filter(filterfield_desc('wf'))


    def get_wflm(self, simidx, it, data=None):
        if not self.wf_field.is_cached(simidx, it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(simidx, it-1)
            tpn_alm = self.calc_prep(data) # this changes lmmax to lmmax_unl via lensgclm
            mchain = CG.conjugate_gradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache_field(cg_sol_curr, simidx, it)
        return self.wf_field.get_field(simidx, it)


    def calc_prep(self, eblm):
        """cg-inversion pre-operation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        eblmc = np.empty_like(eblm)
        eblmc = self.inoise_operator.act(eblm, adjoint=False)
        eblmc = self.beam_operator.act(eblmc, adjoint=False)
        elm = self.wf_operator.act(eblmc, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY').squeeze()
        almxfl(elm, self.cls_filt['ee'] > 0., Alm.getlmax(elm, None), True)
        return elm
    

    def fwd_op(self, elm):
        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(elm)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = nlm.reshape((1, nlm.size))
        eblm = self.wf_operator.act(elm_2d, spin=2, adjoint=False, backwards=False)
        eblm = self.beam_operator.act(eblm, adjoint=False)
        eblm = self.inoise_operator.act(eblm, adjoint=False)
        eblm = self.beam_operator.act(eblm, adjoint=False)
        elm_2d = self.wf_operator.act(eblm, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')

        nlm = elm_2d.squeeze()
        nlm += almxfl(elm, iclee, Alm.getlmax(elm, None)[1], False)
        almxfl(nlm, iclee > 0.0, Alm.getlmax(elm, None)[1], True)
        return nlm


    def precon_op(self, eblm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(eblm[0], None)[0], (Alm.getlmax(eblm[0], None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel = _extend_cl(self.beam_operator.transferfunction['e']*2, len(self.inoise_operator.n1eblm[0])-1) * self.inoise_operator.n1eblm[0]
        # Extend transfer function to avoid preconditioning with zero (~ Gaussian beam)
        if len(ninv_fel) - 1 < Alm.getlmax(eblm[0], None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(eblm[0], None) + 1, dtype=float)))

        flmat = cli(self.cls_filt['ee'][:Alm.getlmax(eblm[0], None) + 1]) + ninv_fel[:Alm.getlmax(eblm[0], None) + 1]
        flmat = cli(flmat) * (self.cls_filt['ee'][:Alm.getlmax(eblm[0], None) + 1] > 0.)
        return almxfl(eblm, flmat, Alm.getlmax(eblm[0], None), False)


    def update_operator(self, simidx, it, secondary=None, component=None):
        self.wf_operator.set_field(simidx, it, component)


    def get_template(self, simidx, it, secondary, component, lm_max_in=None, lm_max_out=None):
        self.wf_operator.set_field(simidx, it, secondary, component)
        estCMB = self.get_wflm(simidx, it)

        # NOTE making sure that QE is perturbative, and resetting MAP to non-perturbative.
        # Must be done for each call, as the operators used are the same instance.
        for operator in self.wf_operator.operators:
            if operator.ID == 'lensing' and 'lensing' in secondary:
                operator.perturbative = (it == 0)

        if lm_max_in is not None and lm_max_out is not None:
            self.wf_operator.update_lm_max(lm_max_in=lm_max_in, lm_max_out=lm_max_out)
                
        return self.wf_operator.act(estCMB, secondary=secondary)