from os.path import join as opj
import numpy as np

from scipy.interpolate import UnivariateSpline as spl
import healpy as hp

from delensalot.core.MAP import CG
from delensalot.core.opfilt import MAP_opfilt_iso_p as MAP_opfilt_iso_p

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd
from delensalot.utils import cli

from delensalot.core.MAP import field
from delensalot.core.MAP import operator


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
        self.simidx = filter_desc['simidx']


    def get_ivfreslm(self, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper, in essence it should do the following:
        if not self.ivf_field.is_cached(self.simidx, it):
            assert eblm_wf is not None and data is not None
            data = alm_copy_nd(data, None, self.beam_operator.lm_max) 
            ivfreslm = self.ivf_operator.act(eblm_wf, spin=2)
            ivfreslm = -1*self.beam_operator.act(ivfreslm)
            ivfreslm += data
            ivfreslm = self.inoise_operator.act(0.5*ivfreslm, adjoint=False)
            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            self.ivf_field.cache(ivfreslm,  self.simidx, it)
        return self.ivf_field.get_field(self.simidx, it)
    

    def update_operator(self, it):
        self.ivf_operator.set_field(self.simidx, it)


class wf:
    def __init__(self, filter_desc):
        self.wf_operator: operator.secondary_operator = filter_desc['wf_operator']
        self.libdir = filter_desc['libdir']
        self.beam_operator: operator.beam = filter_desc['beam_operator']
        self.inoise_operator: operator.inoise_operator = filter_desc['inoise_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.simidx = filter_desc['simidx']

        self.wf_field: field.filter = field.filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(self.simidx, it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(self.simidx, it-1) # FIXME I could use a check here to make sure cg_sol_curr is of the right shape..
            tpn_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = CG.conjugate_gradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, self.simidx, it)
        return self.wf_field.get_field(self.simidx, it)


    def calc_prep(self, eblm):
        """cg-inversion pre-operation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        eblm_ = alm_copy_nd(eblm, None, self.beam_operator.lm_max)
        eblmc = self.inoise_operator.act(eblm_, adjoint=False)
        eblmc = self.beam_operator.act(eblmc, adjoint=False)
        elm = self.wf_operator.act(eblmc, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')[0].squeeze() # NOTE lm_sky -> lm_pri
        elm = almxfl_nd(elm, self.cls_filt['ee'] > 0., None, False)
        return elm
    

    def fwd_op(self, ewflm):
        """ acts on elm, which is a lm_max_pri map
        """
        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(ewflm)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = nlm.reshape((1, nlm.size))
        eblm = self.wf_operator.act(elm_2d, spin=2, adjoint=False, backwards=False) # # NOTE lm_max_pri -> lm_max_sky
        eblm = self.beam_operator.act(eblm, adjoint=False)
        eblm = self.inoise_operator.act(eblm, adjoint=False)
        eblm = self.beam_operator.act(eblm, adjoint=False)
        elm_2d = np.atleast_2d(self.wf_operator.act(eblm, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')[0]) # lm_sky -> lm_pri

        nlm = elm_2d.squeeze()
        nlm += almxfl(ewflm, iclee * (iclee > 0.0), Alm.getlmax(ewflm.size, None), False)
        return nlm


    def precon_op(self, elm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(elm.size, None), (Alm.getlmax(elm[0].size, None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel = _extend_cl(self.beam_operator.transferfunction['e']*2, len(self.inoise_operator.n1eblm[0])-1) * self.inoise_operator.n1eblm[0]
        # Extend transfer function to avoid preconditioning with zero (~ Gaussian beam)
        if len(ninv_fel) - 1 < Alm.getlmax(elm.size, None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(elm.size, None) + 1, dtype=float)))

        lmax = Alm.getlmax(elm.size, None)
        flmat = cli(self.cls_filt['ee'][:lmax+1]) + ninv_fel[:]
        flmat = cli(flmat) * (self.cls_filt['ee'][:lmax+1] > 0.)
        return almxfl(elm, flmat, lmax, False)


    def update_operator(self, it, secondary=None, component=None):
        self.wf_operator.set_field(self.simidx, it, component)


    def get_template(self, it, secondary, component, lm_max_in=None, lm_max_out=None):
        self.wf_operator.set_field(self.simidx, it, secondary, component)
        estCMB = self.get_wflm(self.simidx, it)

        # NOTE making sure that QE is perturbative, and resetting MAP to non-perturbative.
        # Must be done for each call, as the operators used are the same instance.
        for operator in self.wf_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        if lm_max_in is not None and lm_max_out is not None:
            self.wf_operator.update_lm_max(lm_max_in=lm_max_in, lm_max_out=lm_max_out)
                
        return self.wf_operator.act(estCMB, secondary=secondary)