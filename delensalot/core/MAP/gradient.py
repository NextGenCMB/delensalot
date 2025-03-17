from __future__ import annotations
from typing import List, Type, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from delensalot.delensalot.core.job_handler import DataContainer

import numpy as np
from os.path import join as opj

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
# from delensalot.config.etc.logger import set_logging_level

from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.core.MAP import field, operator
from delensalot.core.MAP.context import get_computation_context

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd

class SharedFilters:
    def __init__(self, sub):
        self.wfivf_filter = sub.wfivf_filter


class Gradient(SharedFilters):
    subs: List[Union[LensingGradientSub, BirefringenceGradientSub]]
    def __init__(self, subs, ipriormatrix, verbose=False):
        super().__init__(subs[0]) # NOTE I am assuming the ivf and wf class are the same in all gradients
        self.subs: List[Union[LensingGradientSub, BirefringenceGradientSub]] = subs
        self.ipriormatrix = ipriormatrix
        self.component = [comp for sub in self.subs for comp in sub.component]
        self.comp2idx = {comp: i for i, comp in enumerate(self.component)}
        # self.sub1: LensingGradientSub = object.__new__(LensingGradientSub)()
        # self.sub2: BirefringenceGradientSub = object.__new__(BirefringenceGradientSub)()
    

    @log_on_start(logging.DEBUG, 'Gradient.get_gradient_total, it={it}', logger=log)
    def get_gradient_total(self, it, data=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_total(it_, data) for it_ in it]
        totgrad = []
        for sub in self.subs:
            if sub.gfield.is_cached(it, type='total'):
                totgrad.append(sub.gfield.get_total(it))
            else:
                log.info(f'calculating total gradient for {sub.ID}')
                buff = -sub.get_gradient_meanfield(it) * (0. +0.j) + sub.get_gradient_quad(it=it, data=data)
                for compi, comp in enumerate(sub.component):
                    buff[compi] = almxfl(buff[compi], sub.chh[comp] > 0, None, False)
                totgrad.append(buff)
        prior = self.get_gradient_prior(it-1)
        totgrad = [a + b for a, b in zip(totgrad, prior)]
        return totgrad


    @log_on_start(logging.DEBUG, 'Gradient.get_gradient_quad: it={it}', logger=log)
    def get_gradient_quad(self, it, data=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data) for it_ in it]
        quadgrad = []
        for sub in self.subs:
            if sub.gfield.is_cached(it, type='quad'):
                quadgrad.append(sub.gfield.get_quad(it))
            else:
                log.info(f'calculating quad gradient for {sub.ID}')
                quadgrad.append(sub.get_gradient_quad(it=it, data=data))
        return quadgrad


    @log_on_start(logging.DEBUG, 'Gradient.get_gradient_prior: it={it}', logger=log)
    def get_gradient_prior(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_prior(it_) for it_ in it]
        orig = self._get_est_for_prior(it)
        original_shapes = [arr.shape for arr in orig]
        est = np.vstack(orig)

        result = []
        for xi, x in enumerate(self.ipriormatrix):
            prod_ = sum(almxfl(est[yi], self.ipriormatrix[xi, yi], None, False) for yi, y in enumerate(x))
            result.append(prod_)

        final_result = []
        start = 0
        for shape in original_shapes:
            num_rows = shape[0]
            final_result.append(result[start:start + num_rows]) 
            start += num_rows
        return final_result


    @log_on_start(logging.DEBUG, 'Gradient._get_est_for_prior: it={it}', logger=log)
    def _get_est_for_prior(self, it):
        if it>0:
            print('Setting it=0 as only this is supported for now')
        # it = 0
        if isinstance(it, (list, np.ndarray)):
            return [self._get_est_for_prior(it_) for it_ in it]
        return [sub.gfield._get_est(it=it) for sub in self.subs]
    
    
    @log_on_start(logging.DEBUG, 'Gradient.get_gradient_meanfield: it={it}', logger=log)
    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        return [sub.get_gradient_meanfield(it) for sub in self.subs]


    def update_operator(self, field):
        self.wfivf_filter.update_operator(field)


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if hasattr(self.wfivf_filter, name):
                return getattr(self.wfivf_filter, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder
    

    def _copyQEtoDirectory(self, QE_searchs, idxs, libdir, lm_max_wf):
        ctx, _ = get_computation_context()
        # NOTE this turns them into convergence fields
        sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(QE_searchs)}
        QE_search_secondariesIDs = [QE_search.secondary.ID for QE_search in QE_searchs]
        # secondaries = {QE_search.secondary.ID: QE_search.secondary for QE_search in QE_searchs}
        secondaries = [self.wfivf_filter.sec_operator.operators[sec2idx[sec]] for sec in QE_search_secondariesIDs]
        secondaries_field = {
            sub.ID: field.Secondary({
                "ID":  sub.ID,
                "component": sub.component,
                "libdir": libdir,
        }) for sub in secondaries}

        for idx in idxs:
            # idx2 = idx 
            for idx2 in idxs: # FIXME must implement idx2 for QE_searchs. For now, I fake and generate cross from idx
                ctx.set(idx=idx, idx2=idx2)
                for operator in self.wfivf_filter.sec_operator.operators:
                    QE_searchs[sec2idx[operator.ID]].init_filterqest()
                    if not all(secondaries_field[operator.ID].is_cached(it=0)):
                        klm_QE = QE_searchs[sec2idx[operator.ID]].get_est(idx)
                        secondaries_field[operator.ID].cache_klm(klm_QE,it=0)
                    
                    #TODO cache QE wflm into the filter directory
                    if not self.wfivf_filter.wf_field.is_cached(it=0):
                        wflm_QE = QE_searchs[sec2idx[operator.ID]].get_wflm(idx, lm_max_wf)
                        self.wfivf_filter.wf_field.cache(np.array(wflm_QE), it=0)

                for sub in self.subs:
                    if not sub.gfield.is_cached(it=0):
                        kmflm_QE = QE_searchs[sec2idx[sub.gfield.ID]].get_kmflm(idx)
                        sub.gfield.cache(kmflm_QE, it=0, type='meanfield')

        # FIXME need to reset the context to the original idx and idx2


class GradSub:
    def __init__(self, gradient_desc):
        
        self.component = gradient_desc['component']
        self.ID = gradient_desc['ID']
        self.chh = gradient_desc['chh']
        libdir = gradient_desc['libdir']
        self.sky_coverage = gradient_desc['sky_coverage']

        self.geom_lib = gradient_desc['sec_operator'].operators[-1].lenjob_geomlib
        self.sht_tr = gradient_desc['sht_tr']

        self.wfivf_filter = gradient_desc.get('wfivf_filter', None)

        self.LM_max = gradient_desc['LM_max']

        self.data_container: DataContainer = gradient_desc['data_container']

        gfield_desc = {
            "ID": self.ID,
            "libdir": opj(libdir, 'gradients'),
            "libdir_prior": opj(libdir, 'estimate'),  
            "chh": self.chh,
            "component": self.component,
        }
        self.gfield = field.Gradient(gfield_desc)


    def get_gradient_total(self, it, data=None, data_leg2=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_total(it_, data, data_leg2) for it_ in it]
        data_leg2 = data_leg2 or data
        ctx, is_new = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if component is None:
            component = self.gfield.component

        if self.gfield.cacher.is_cached(idx=idx, idx2=idx2, it=it):
            return self.gfield.get_total(it, self.LM_max, component)
        else:
            total = 0
            total += self.get_gradient_prior(it - 1)
            total -= self.get_gradient_meanfield(it) * (0. +0.j)
            total += self.get_gradient_quad(it)
            return total
            # self.gfield.cache_total(total, idx, idx2, it)
            # return total


    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        # NOTE filtering it here as it could be that not all it in list are calculated, so need to calculate them
        return self.gfield.get_meanfield(it)


class LensingGradientSub(GradSub):
    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.Compound = self._get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[-1].lm_max_in
        self.data_key = desc['data_key']
        

    def get_gradient_quad_(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        # NOTE This is the "1d"
        spin0part, spin2part = False, True
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data, data_leg2, wflm, ivfreslm) for it_ in it]
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        if self.data_container is None:
            assert wflm is not None and ivfreslm is not None, "wflm and ivfreslm must be provided as data container is missing"
        elif data is not None:
            data_leg2 = data_leg2 or data # NOTE these are the data to calculate ivfreslm and wf
        if not self.gfield.is_cached(it=it, type='quad'):
            if wflm is None:
                assert self.wfivf_filter is not None, "wfivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wfivf_filter.get_wflm(it, self.data_container.get_data(idx))
                ivfreslm = np.ascontiguousarray(self.wfivf_filter.get_ivfreslm(it, self.data_container.get_data(idx2), wflm))

            # TODO depending on shape of wflm and ivfreslm, I run different routines
            resmap_c = np.ascontiguousarray(np.empty((self.geom_lib.npix(),), dtype=wflm.dtype))
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.geom_lib.synthesis(ivfreslm, 2, *self.lm_max_in, self.sht_tr, map=resmap_r) # ivfmap
            gcs_r = self.gradient_operator.act(np.copy(wflm), spin=3) # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
            gcs_r = self.gradient_operator.act(np.copy(wflm), spin=1) # xwfglm
            gc_c -= resmap_c * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
            gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
            gc = self.geom_lib.adjoint_synthesis(gc_r, 1, self.LM_max[0], self.LM_max[0], self.sht_tr)
                
            # NOTE at last, cast qlms to alm space with LM_max and also cast it to convergence
            fl1 = np.sqrt(np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl1, self.LM_max[1], True)
            almxfl(gc[1], fl1, self.LM_max[1], True)
            fl2 = cli(0.5 * np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl2, self.LM_max[1], True)
            almxfl(gc[1], fl2, self.LM_max[1], True)
                
            self.cache(gc, it=it, type='quad')
        return self.gfield.get_quad(it)
    

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        # NOTE this is the 3d version
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data, data_leg2, wflm, ivfreslm) for it_ in it]
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        if self.data_container is None:
            assert wflm is not None and ivfreslm is not None, "wflm and ivfreslm must be provided as data container is missing"
        elif data is not None:
            data_leg2 = data_leg2 or data # NOTE these are the data to calculate ivfreslm and wf
        if not self.gfield.is_cached(it=it, type='quad'):
            if wflm is None:
                assert self.wfivf_filter is not None, "wfivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wfivf_filter.get_wflm(it, self.data_container.get_data(idx))
                ivfreslm = np.ascontiguousarray(self.wfivf_filter.get_ivfreslm(it, self.data_container.get_data(idx2), wflm))

            resmap_c = np.ascontiguousarray(np.empty((self.geom_lib.npix(),), dtype=wflm.dtype))
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            if self.data_key in ['p', 'tp', 'ee', 'eb', 'bb']:
                self.geom_lib.synthesis(ivfreslm[1:], 2, *self.lm_max_in, self.sht_tr, map=resmap_r) # ivfmap
                ponly = np.copy(wflm)
                ponly[0] *= 0
                gcs_r = self.gradient_operator.act(ponly, spin=3) # xwfglm
                gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
                ponly = np.copy(wflm)
                ponly[0] *= 0
                gcs_r = self.gradient_operator.act(ponly, spin=1) # xwfglm
                gc_c -= resmap_c * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
                gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array

            if self.data_key in ['tp', 'tt']:
                irestmap = self.geom_lib.synthesis(ivfreslm[0], 0, *self.lm_max_in, self.sht_tr)[0]
                tonly = np.copy(wflm)
                tonly[1:] *= 0
                buff_gtmap = self.gradient_operator.act(tonly, spin=1)
                gc_r_ = buff_gtmap * irestmap
            gcr = 0
            gcr += gc_r if 'gc_r' in locals() else gcr
            gcr += gc_r_ if 'gc_r_' in locals() else gcr
            gc = self.geom_lib.adjoint_synthesis(gcr, 1, self.LM_max[0], self.LM_max[0], self.sht_tr)
                
            # NOTE at last, cast qlms to alm space with LM_max and also cast it to convergence
            fl1 = np.sqrt(np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl1, self.LM_max[1], True)
            almxfl(gc[1], fl1, self.LM_max[1], True)
            fl2 = cli(0.5 * np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl2, self.LM_max[1], True)
            almxfl(gc[1], fl2, self.LM_max[1], True)
                
            self.cache(gc, it=it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        lm_max_out = filter_operator.operators[-1].lm_max_out
        return operator.Compound([operator.SpinRaise(lm_max=lm_max_out), filter_operator], out='map', sht_tr=self.sht_tr)
    

    def cache(self, gfieldlm, it, type='quad'):
        self.gfield.cache(gfieldlm, it=it, type=type)


    def is_cached(self, it, type):
        return self.gfield.is_cached(type=type, it=it)


class BirefringenceGradientSub(GradSub):

    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self._get_operator(desc['sec_operator'])
        self.lm_max = self.gradient_operator.operators[-1].operators[0].lm_max
    

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data, data_leg2, wflm, ivfreslm) for it_ in it]
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        data_leg2 = data_leg2 or data
        if not self.gfield.is_cached(it, type='quad'):
            wflm = self.wfivf_filter.get_wflm(it, self.data_container.get_data(idx))
            ivfreslm = np.ascontiguousarray(self.wfivf_filter.get_ivfreslm(it, self.data_container.get_data(idx2), wflm))

            # xwfmap = self.gradient_operator.act(wflm, spin=2)
            lmax = Alm.getlmax(wflm[0].size, None)
            xwfmap = self.geom_lib.synthesis(wflm[1:], 2, lmax, lmax, self.sht_tr)
            lmax = Alm.getlmax(ivfreslm[0].size, None)
            ivfmap = self.geom_lib.synthesis(ivfreslm[1:], 2, lmax, lmax, self.sht_tr)
 
            # qlms = -4*(ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = +2*(+ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.geom_lib.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.sht_tr)
            
            self.gfield.cache(qlms, it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        return operator.Compound([filter_operator], out='map', sht_tr=self.sht_tr)