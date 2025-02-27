import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np

from scipy.interpolate import UnivariateSpline as spl
import healpy as hp

from delensalot.core.MAP import cg
from delensalot.core.opfilt import MAP_opfilt_iso_p as MAP_opfilt_iso_p

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd
from delensalot.utils import cli

from delensalot.core.MAP.context import ComputationContext
from delensalot.core.MAP import field
from delensalot.core.MAP import operator_v2 as operator

import matplotlib.pyplot as plt
import healpy as hp

filterfield_desc = lambda ID, libdir: {
    "ID": ID,
    "libdir": opj(libdir),
    "fns": f"{ID}_idx{{idx}}_{{idx2}}_it{{it}}",}


class IVF_v2:
    def __init__(self, filter_desc): 
        self.libdir = filter_desc['libdir']
        self.ivf_operator = filter_desc['ivf_operator']
        self.beam_operator = filter_desc['beam_operator']
        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']
        self.add_operator: operator.Add = filter_desc['add_operator']


    # def get_ivfreslm(self, it, data=None, eblm_wf=None):
    #     # NOTE this is eq. 21 of the paper
    #     if not self.ivf_field.is_cached(it=it):
    #         assert eblm_wf is not None and data is not None
    #         map = self.ivf_operator.apply_inplace(eblm_wf)    # pri -> pri                    alm -> map      alm -> map
    #         self.beam_operator.apply_inplace(-1*ivfreslm)     # pri -> sky                    map -> alm      map -> alm

    #         a = self.add_operator.apply(ivfreslm, data)
    #         ivfreslm = self.inv_operator.apply(a)

    #         ivfreslm = self.beam_operator.act(ivfreslm)  #                                    alm -> alm      alm -> alm
    #         self.ivf_field.cache(ivfreslm, it=it)
    #     return self.ivf_field.get_field(it=it)
    def get_ivfreslm(self, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert eblm_wf is not None and data is not None
            ivfreslm = self.ivf_operator.apply_inplace(eblm_wf)
            ivfreslm = -1*self.beam_operator.apply_inplace(ivfreslm)

            if data.dtype in [np.complex64, np.complex128]:
                ivfreslm += data
                ivfreslm = self.inv_operator.apply(0.5*ivfreslm)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = self.inv_operator.data_geomlib.synthesis(ivfreslm, 2, *lm_max, 6)
                ivfresmap += data
                ivfreslm = self.inv_operator.apply(ivfresmap)

            ivfreslm = self.beam_operator.apply_inplace(ivfreslm)
            self.ivf_field.cache(ivfreslm, it=it)
        return self.ivf_field.get_field(it=it)


    def update_operator(self, field):
        self.ivf_operator.set_field(field)


class WF_v2:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.wf_operator: operator.Secondary = filter_desc['wf_operator']
        self.beam_operator: operator.Beam = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(it=it-1)# * (0+1j*0)
            tpn_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = cg.ConjugateGradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    @log_on_start(logging.DEBUG, " ---- calc_prep: {data_sky.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- calc_prep", logger=log)  
    def calc_prep(self, data_sky): # sky -> pri
        """cg-inversion pre-operation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        # NOTE if data_sky is complex, I can continue with data_sky, but not doing this at this time
        eblm_sky = self.inv_operator.apply(data_sky)
        self.beam_operator.apply_inplace(eblm_sky) #  eblm_sky or data_sky                             
        elm_pri = self.wf_operator.apply_adjoint_inplace(eblm_sky) # eblm_sky or data_sky NOTE lm_sky -> lm_pri
        almxfl(elm_pri, self.cls_filt['ee'] > 0., None, True)
        return elm_pri
    

    @log_on_start(logging.DEBUG, " ---- fwd_op: {ewflm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- fwd_op", logger=log)  
    def fwd_op(self, ewflm): # pri -> pri
        """ acts on elm, which is a lm_max_pri map
        """
        iclee = cli(self.cls_filt['ee'])
        nlm = np.copy(ewflm).reshape((1, ewflm.size))
        nlm = self.wf_operator.apply_inplace(nlm) # NOTE                lm_pri -> lm_pri    alm -> alm  alm -> alm
        elm_sky = self.beam_operator.apply_inplace(nlm) #               lm_pri -> lm_sky    alm -> alm  alm -> alm
        self.inv_operator.apply(elm_sky) #                              lm_sky -> lm_sky    alm -> alm  alm -> alm
        self.beam_operator.apply_inplace(elm_sky) #                     lm_sky -> lm_sky    alm -> alm  alm -> alm

        elm_pri = self.wf_operator.apply_adjoint_inplace(elm_sky) #     lm_sky -> lm_pri
        ret = elm_pri + almxfl(ewflm, iclee * (iclee > 0.0), Alm.getlmax(ewflm.size, None), False)
        # hp.mollview(hp.alm2map(elm_pri, nside=256))
        # plt.show()
        return ret


    @log_on_start(logging.DEBUG, " ---- precon_op: {elm.shape}", logger=log)
    @log_on_end(logging.DEBUG, " done ---- precon_op", logger=log)  
    def precon_op(self, elm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(elm.size, None), (Alm.getlmax(elm[0].size, None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel, ninv_fbl = self.inv_operator.get_febl(self.beam_operator.transferfunction)
        if len(ninv_fel) - 1 < Alm.getlmax(elm.size, None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(elm.size, None) + 1, dtype=float)))

        lmax = Alm.getlmax(elm.size, None)
        flmat = cli(self.cls_filt['ee'][:lmax+1]) + ninv_fel[:]
        flmat = cli(flmat) * (self.cls_filt['ee'][:lmax+1] > 0.)
        return almxfl(elm, flmat, lmax, False)


    def get_template(self, it, secondary, component, lm_max_in=None, lm_max_out=None):
        ctx, _ = ComputationContext()  # Get the singleton instance
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.wf_operator.set_field(idx=idx, it=it, secondary=secondary, component=component, idx2=idx2)
        estCMB = self.get_wflm(it=it)

        for operator in self.wf_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.wf_operator.act(estCMB, secondary=secondary)
    


class IVF:
    def __init__(self, filter_desc): 
        self.libdir = filter_desc['libdir']
        self.ivf_operator = filter_desc['ivf_operator']
        self.beam_operator = filter_desc['beam_operator']
        self.ivf_field = field.Filter(filterfield_desc('ivf', self.libdir))
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']


    def get_ivfreslm(self, it, data=None, eblm_wf=None):
        # NOTE this is eq. 21 of the paper
        if not self.ivf_field.is_cached(it=it):
            assert eblm_wf is not None and data is not None
            ivfreslm = self.ivf_operator.act(eblm_wf, spin=2)
            ivfreslm = -1*self.beam_operator.act(ivfreslm)

            if data.dtype in [np.complex64, np.complex128]:
                ivfreslm += data
                ivfreslm = self.inv_operator.act(0.5*ivfreslm, adjoint=False)
            else:
                lm_max = self.inv_operator.lm_max
                ivfresmap = self.inv_operator.geom_lib.synthesis(ivfreslm, 2, *lm_max, 6)
                ivfresmap += data
                ivfreslm = self.inv_operator.apply_map(ivfresmap)

            ivfreslm = self.beam_operator.act(ivfreslm, adjoint=False)
            self.ivf_field.cache(ivfreslm, it=it)
        return self.ivf_field.get_field(it=it)
    

    def update_operator(self, field):
        self.ivf_operator.set_field(field)
    # def update_operator(self, it):
    #     ctx, _ = ComputationContext()
    #     idx, idx2 = ctx.idx, ctx.idx2
    #     self.ivf_operator.set_field(idx=idx, it=it, idx2=idx2)


class WF:
    def __init__(self, filter_desc):
        self.libdir = filter_desc['libdir']
        self.wf_operator: operator.SecondaryOperator = filter_desc['wf_operator']
        self.beam_operator: operator.BeamOperator = filter_desc['beam_operator']
        self.inv_operator: operator.InverseNoiseVariance = filter_desc['inv_operator']

        self.chain_descr = filter_desc['chain_descr']
        self.cls_filt = filter_desc['cls_filt']

        self.wf_field: field.Filter = field.Filter(filterfield_desc('wf', self.libdir))


    def get_wflm(self, it, data=None):
        if not self.wf_field.is_cached(it=it):
            assert data is not None, 'data is required for the calculation'
            cg_sol_curr = self.wf_field.get_field(it=it-1) * (0+1j*0)
            tpn_alm = self.calc_prep(data) # NOTE lm_sky -> lm_pri
            mchain = cg.ConjugateGradient(self.precon_op, self.chain_descr, self.cls_filt)
            mchain.solve(cg_sol_curr, tpn_alm, self.fwd_op)
            self.wf_field.cache(cg_sol_curr, it=it)
        return self.wf_field.get_field(it=it)


    def calc_prep(self, qumap):
        """cg-inversion pre-operation. This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`
        """
        if qumap.dtype in [np.complex64, np.complex128]:
            eblmc = self.inv_operator.act(qumap, adjoint=False)
        else:
            eblmc = self.inv_operator.apply_map(qumap)
        eblmc = self.beam_operator.act(eblmc, adjoint=False)
        elm = self.wf_operator.act(eblmc, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY') # NOTE lm_sky -> lm_pri
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
        
        if eblm.dtype in [np.complex64, np.complex128]:
            eblm = self.inv_operator.act(eblm, adjoint=False)
        else:
            lm_max = self.inv_operator.lm_max
            qumap = self.inv_operator.geom_lib.synthesis(eblm, 2, *lm_max, 6)
            eblm = self.inv_operator.apply_map(qumap)
        
        eblm = self.beam_operator.act(eblm, adjoint=False)
        elm_2d = np.atleast_2d(self.wf_operator.act(eblm, spin=2, adjoint=True, backwards=True, out_sht_mode='GRAD_ONLY')) # lm_sky -> lm_pri
        nlm = elm_2d.squeeze()
        nlm += almxfl(ewflm, iclee * (iclee > 0.0), Alm.getlmax(ewflm.size, None), False)
        return nlm


    def precon_op(self, elm):
        """Applies the preconditioner operation for diagonal preconditioning.
        """
        assert len(self.cls_filt['ee']) > Alm.getlmax(elm.size, None), (Alm.getlmax(elm[0].size, None), len(self.cls_filt['ee']))
        # FIXME indexing must change for MV and T-only
        ninv_fel, ninv_fbl = self.inv_operator.get_febl(self.beam_operator.transferfunction)
        if len(ninv_fel) - 1 < Alm.getlmax(elm.size, None):
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(Alm.getlmax(elm.size, None) + 1, dtype=float)))

        lmax = Alm.getlmax(elm.size, None)
        flmat = cli(self.cls_filt['ee'][:lmax+1]) + ninv_fel[:]
        flmat = cli(flmat) * (self.cls_filt['ee'][:lmax+1] > 0.)
        return almxfl(elm, flmat, lmax, False)


    def update_operator(self, it):
        ctx, _ = ComputationContext()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.wf_operator.set_field(idx=idx, it=it, idx2=idx2)


    def get_template(self, it, secondary, component, lm_max_in=None, lm_max_out=None):
        ctx, _ = ComputationContext()  # Get the singleton instance
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.wf_operator.set_field(idx=idx, it=it, secondary=secondary, component=component, idx2=idx2)
        estCMB = self.get_wflm(it=it)

        for operator in self.wf_operator.operators:
            if operator.ID == 'lensing':
                operator.perturbative = (it == 0)

        return self.wf_operator.act(estCMB, secondary=secondary)