from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from delensalot.delensalot.core.job_handler import DataContainer

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from os.path import join as opj
from typing import List, Type, Union
from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd, alm_copy_nd
from delensalot.utils import cli

from delensalot.core.MAP import field
from delensalot.core.MAP import operator as operator

from delensalot.core.MAP.filter import IVF, WF
from delensalot.core.MAP.context import get_computation_context


class SharedFilters:
    def __init__(self, sub):
        self.ivf_filter = sub.ivf_filter
        self.wf_filter = sub.wf_filter


class GradSub:
    def __init__(self, gradient_desc):
        
        self.component = gradient_desc['component']
        self.ID = gradient_desc['ID']
        self.chh = gradient_desc['chh']
        libdir = gradient_desc['libdir']
        self.data_key = gradient_desc['data_key']
        self.sky_coverage = gradient_desc['sky_coverage']

        self.ivf_filter: IVF = gradient_desc.get('ivf_filter', None)
        self.wf_filter: WF = gradient_desc.get('wf_filter', None)

        self.ffi = gradient_desc['sec_operator'].operators[0].ffi # each sec operator has the same ffi, so can pick any
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
            total += self.get_gradient_meanfield(it)
            total -= self.get_gradient_quad(it)
            return total
            # self.gfield.cache_total(total, idx, idx2, it)
            # return total


    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        # NOTE filtering it here as it could be that not all it in list are calculated, so need to calculate them
        return self.gfield.get_meanfield(it)
    

    def update_operator(self, it):
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.ivf_filter.update_operator(idx=idx, it=it, idx2=idx2)
    

    def get_data(self, idx, space, sky_coverage='full'):
        lm_max_sky = self.ivf_filter.ivf_operator.operators[-1].lm_max_in
        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            spin = 2 if space == 'map' else 0
            self.data_container.purgecache()
            if self.data_key in ['p', 'eb', 'be']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)
            elif self.data_key in ['ee']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)[0]
            elif self.data_key in ['tt']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='temperature', sky_coverage=sky_coverage, lm_max=lm_max_sky)
            elif self.data_key in ['p']:
                EBobs = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)
                Tobs = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='temperature', sky_coverage=sky_coverage, lm_max=lm_max_sky)       
                ret = np.array([Tobs, *EBobs])
            else:
                assert 0, 'implement if needed'
        return ret


    def _copyQEtoDirectory(self, QE_searchs, idxs):
        # NOTE this turns them into convergence fields
        sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(QE_searchs)}
        QE_search_secondariesIDs = [QE_search.secondary.ID for QE_search in QE_searchs]
        # secondaries = {QE_search.secondary.ID: QE_search.secondary for QE_search in QE_searchs}
        secondaries = [self.ivf_filter.ivf_operator.operators[sec2idx[sec]] for sec in QE_search_secondariesIDs]
        secondaries_field = {
            sub.ID: field.secondary({
                "ID":  sub.ID,
                "component": sub.component,
                "libdir": self.gfield.libdir_prior,
        }) for sub in secondaries}

        for idx in idxs:
            # idx2 = idx 
            for idx2 in idxs: # FIXME must implement idx2 for QE_searchs. For now, I fake and generate cross from idx
                for operator in self.ivf_filter.ivf_operator.operators:
                    QE_searchs[sec2idx[operator.ID]].init_filterqest()
                    if not all(secondaries_field[operator.ID].is_cached(idx, idx2=idx2, it=0)):
                        klm_QE = QE_searchs[sec2idx[operator.ID]].get_est(idx)
                        secondaries_field[operator.ID].cache_klm(klm_QE, idx, idx2=idx2, it=0)
                    
                    #TODO cache QE wflm into the filter directory
                    if not self.wf_filter.wf_field.is_cached(idx, it=0):
                        lm_max_out = self.gradient_operator.operators[-1].operators[0].lm_max_out
                        wflm_QE = QE_searchs[sec2idx[operator.ID]].get_wflm(idx, lm_max_out)
                        self.wf_filter.wf_field.cache(np.array(wflm_QE), idx, idx2=idx2, it=0)

                    if not self.gfield.is_cached(idx, idx2=idx2, it=0):
                        kmflm_QE = QE_searchs[sec2idx[self.gfield.ID]].get_kmflm(idx)
                        self.gfield.cache_meanfield(kmflm_QE, idx, idx2=idx2, it=0)


class LensingGradientSub(GradSub):
    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self._get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[-1].lm_max_in
        

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
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
                assert self.ivf_filter is not None, "ivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                assert self.wf_filter is not None, "wf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wf_filter.get_wflm(it, self.get_data(idx, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage))
                ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage), wflm))

            resmap_c = np.ascontiguousarray(np.empty((self.ffi.geom.npix(),), dtype=wflm.dtype))
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.ffi.geom.synthesis(ivfreslm, 2, *self.lm_max_in, self.ffi.sht_tr, map=resmap_r) # ivfmap
            
            gcs_r = self.gradient_operator.act(wflm, spin=3) # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)

            gcs_r = self.gradient_operator.act(wflm, spin=1) # xwfglm
            gc_c -= resmap_c * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
            gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
            gc = self.ffi.geom.adjoint_synthesis(gc_r, 1, self.LM_max[0], self.LM_max[0], self.ffi.sht_tr)
            
            # NOTE at last, cast qlms to alm space with LM_max and also cast it to convergence
            fl1 = -np.sqrt(np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl1, self.LM_max[1], True)
            almxfl(gc[1], fl1, self.LM_max[1], True)
            fl2 = cli(0.5 * np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl2, self.LM_max[1], True)
            almxfl(gc[1], fl2, self.LM_max[1], True)
            self.cache(gc, it=it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        lm_max_out = filter_operator.operators[0].lm_max_out
        return operator.JointOperator([operator.SpinRaiseOperator(lm_max=lm_max_out), filter_operator], out='map')
    

    def cache(self, gfieldlm, it, type='quad'):
        self.gfield.cache(gfieldlm, it=it, type=type)


    def is_cached(self, it, type):
        return self.gfield.is_cached(type=type, it=it)


class BirefringenceGradientSub(GradSub):

    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self._get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[0].lm_max_in
    

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data, data_leg2, wflm, ivfreslm) for it_ in it]
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        data_leg2 = data_leg2 or data
        if not self.gfield.is_cached(it, type='quad'):
            wflm = self.wf_filter.get_wflm(it, self.get_data(idx, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage))
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage), wflm))
            
            ivfmap = self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_in[0], self.lm_max_in[1], self.ffi.sht_tr)
            xwfmap = self.gradient_operator.act(wflm, spin=2)
 
            qlms = -4 * (ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache(qlms, it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        return operator.JointOperator([operator.Multiply({"factor": -1j}), filter_operator], out='map')


class Gradient(SharedFilters):
    subs: List[Union[LensingGradientSub, BirefringenceGradientSub]]
    def __init__(self, subs, ipriormatrix):
        super().__init__(subs[0]) # NOTE I am assuming the ivf and wf class are the same in all gradients
        self.subs: List[Union[LensingGradientSub, BirefringenceGradientSub]] = subs
        self.ipriormatrix = ipriormatrix
        self.component = [comp for sub in self.subs for comp in sub.component]
        self.comp2idx = {comp: i for i, comp in enumerate(self.component)}
        # self.sub1: LensingGradientSub = object.__new__(LensingGradientSub)()
        # self.sub2: BirefringenceGradientSub = object.__new__(BirefringenceGradientSub)()
    
    
    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_total')
    def get_gradient_total(self, it, data=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_total(it_, data) for it_ in it]
        totgrad = []
        for sub in self.subs:
            if sub.gfield.is_cached(it, type='total'):
                totgrad.append(sub.gfield.get_total(it))
            else:
                log.info(f'calculating total gradient for {sub.ID}')
                totgrad.append(sub.get_gradient_meanfield(it) - sub.get_gradient_quad(it=it, data=data))
        prior = self.get_gradient_prior(it-1)
        totgrad = [a + b for a, b in zip(totgrad, prior)]
        return totgrad

    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_quad: idx={idx}, it={it}, component={component}, data={data}, idx2={idx2}')
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


    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_prior: idx={idx}, it={it}, component={component}, idx2={idx2}')
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

    
    def _get_est_for_prior(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self._get_est_for_prior(it_) for it_ in it]
        return [sub.gfield._get_est(it=it) for sub in self.subs]
    
    
    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        return [sub.get_gradient_meanfield(it) for sub in self.subs]


    def update_operator(self, field):
        self.ivf_filter.update_operator(field)


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if hasattr(self.ivf_filter, name):
                return getattr(self.ivf_filter, name)(*args, **kwargs)
            elif hasattr(self.wf_filter, name):
                return getattr(self.wf_filter, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder


class GradSub:
    def __init__(self, gradient_desc):
        
        self.component = gradient_desc['component']
        self.ID = gradient_desc['ID']
        self.chh = gradient_desc['chh']
        libdir = gradient_desc['libdir']
        self.data_key = gradient_desc['data_key']
        self.sky_coverage = gradient_desc['sky_coverage']

        self.ivf_filter: IVF = gradient_desc.get('ivf_filter', None)
        self.wf_filter: WF = gradient_desc.get('wf_filter', None)

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
            total += self.get_gradient_meanfield(it)
            total -= self.get_gradient_quad(it)
            return total
            # self.gfield.cache_total(total, idx, idx2, it)
            # return total


    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        # NOTE filtering it here as it could be that not all it in list are calculated, so need to calculate them
        return self.gfield.get_meanfield(it)
    

    def update_operator(self, it):
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.ivf_filter.update_operator(idx=idx, it=it, idx2=idx2)
    

    def get_data(self, idx, space, sky_coverage='full'):
        lm_max_sky = self.ivf_filter.ivf_operator.operators[-1].lm_max_in
        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            spin = 2 if space == 'map' else 0
            self.data_container.purgecache()
            if self.data_key in ['p', 'eb', 'be']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)
            elif self.data_key in ['ee']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)[0]
            elif self.data_key in ['tt']:
                ret = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='temperature', sky_coverage=sky_coverage, lm_max=lm_max_sky)
            elif self.data_key in ['p']:
                EBobs = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='polarization', sky_coverage=sky_coverage, lm_max=lm_max_sky)
                Tobs = self.data_container.get_sim_obs(idx, space=space, spin=spin, field='temperature', sky_coverage=sky_coverage, lm_max=lm_max_sky)       
                ret = np.array([Tobs, *EBobs])
            else:
                assert 0, 'implement if needed'
        return ret


    def _copyQEtoDirectory(self, QE_searchs, idxs):
        # NOTE this turns them into convergence fields
        sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(QE_searchs)}
        QE_search_secondariesIDs = [QE_search.secondary.ID for QE_search in QE_searchs]
        # secondaries = {QE_search.secondary.ID: QE_search.secondary for QE_search in QE_searchs}
        secondaries = [self.ivf_filter.ivf_operator.operators[sec2idx[sec]] for sec in QE_search_secondariesIDs]
        secondaries_field = {
            sub.ID: field.secondary({
                "ID":  sub.ID,
                "component": sub.component,
                "libdir": self.gfield.libdir_prior,
        }) for sub in secondaries}

        for idx in idxs:
            # idx2 = idx 
            for idx2 in idxs: # FIXME must implement idx2 for QE_searchs. For now, I fake and generate cross from idx
                for operator in self.ivf_filter.ivf_operator.operators:
                    QE_searchs[sec2idx[operator.ID]].init_filterqest()
                    if not all(secondaries_field[operator.ID].is_cached(idx, idx2=idx2, it=0)):
                        klm_QE = QE_searchs[sec2idx[operator.ID]].get_est(idx)
                        secondaries_field[operator.ID].cache_klm(klm_QE, idx, idx2=idx2, it=0)
                    
                    #TODO cache QE wflm into the filter directory
                    if not self.wf_filter.wf_field.is_cached(idx, it=0):
                        lm_max_out = self.gradient_operator.operators[-1].operators[0].lm_max_out
                        wflm_QE = QE_searchs[sec2idx[operator.ID]].get_wflm(idx, lm_max_out)
                        self.wf_filter.wf_field.cache(np.array(wflm_QE), idx, idx2=idx2, it=0)

                    if not self.gfield.is_cached(idx, idx2=idx2, it=0):
                        kmflm_QE = QE_searchs[sec2idx[self.gfield.ID]].get_kmflm(idx)
                        self.gfield.cache_meanfield(kmflm_QE, idx, idx2=idx2, it=0)


class LensingGradientSub(GradSub):
    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self._get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[-1].lm_max_in
        

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
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
                assert self.ivf_filter is not None, "ivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                assert self.wf_filter is not None, "wf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wf_filter.get_wflm(it, self.get_data(idx, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage))
                ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage), wflm))

            resmap_c = np.ascontiguousarray(np.empty((self.wf_filter.wf_operator.operators[-1].ffi.geom.npix(),), dtype=wflm.dtype))
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            
            self.wf_filter.wf_operator.operators[-1].ffi.geom.synthesis(ivfreslm, 2, *self.lm_max_in, self.wf_filter.wf_operator.operators[-1].ffi.sht_tr, map=resmap_r) # ivfmap
            
            gcs_r = self.gradient_operator.apply_inplace(wflm, spin=3) # xwfglm
            gc_c = resmap_c.conj() * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)

            gcs_r = self.gradient_operator.apply_inplace(wflm, spin=1) # xwfglm
            gc_c -= resmap_c * gcs_r.T.copy().view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
            gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
            gc = self.wf_filter.wf_operator.operators[-1].ffi.geom.adjoint_synthesis(gc_r, 1, self.LM_max[0], self.LM_max[0], self.wf_filter.wf_operator.operators[-1].ffi.sht_tr)
            
            # NOTE at last, cast qlms to alm space with LM_max and also cast it to convergence
            fl1 = -np.sqrt(np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl1, self.LM_max[1], True)
            almxfl(gc[1], fl1, self.LM_max[1], True)
            fl2 = cli(0.5 * np.arange(self.LM_max[0]+1) * np.arange(1, self.LM_max[0]+2))
            almxfl(gc[0], fl2, self.LM_max[1], True)
            almxfl(gc[1], fl2, self.LM_max[1], True)
            self.cache(gc, it=it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        lm_max_out = filter_operator.operators[-1].lm_max_out
        return operator.Compound([operator.SpinRaise(lm_max=lm_max_out), filter_operator], spin=2) # FIXME this needs changing for supporting all estimators
    

    def cache(self, gfieldlm, it, type='quad'):
        self.gfield.cache(gfieldlm, it=it, type=type)


    def is_cached(self, it, type):
        return self.gfield.is_cached(type=type, it=it)


class BirefringenceGradientSub(GradSub):

    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self._get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[1].lm_max_in
    

    def get_gradient_quad(self, it, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_quad(it_, data, data_leg2, wflm, ivfreslm) for it_ in it]
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        data_leg2 = data_leg2 or data
        if not self.gfield.is_cached(it, type='quad'):
            wflm = self.wf_filter.get_wflm(it, self.get_data(idx, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage))
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2, space='map' if self.sky_coverage=='masked' else 'alm', sky_coverage=self.sky_coverage), wflm))
            
            ivfmap = self.wf_filter.wf_operator.operators[-1].ffi.geom.synthesis(ivfreslm, 2, self.lm_max_in[0], self.lm_max_in[1], self.wf_filter.wf_operator.operators[-1].ffi.sht_tr)
            xwfmap = self.gradient_operator.apply_inplace(wflm)
 
            qlms = -4 * (ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.wf_filter.wf_operator.operators[-1].ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.wf_filter.wf_operator.operators[-1].ffi.sht_tr)
            
            self.gfield.cache(qlms, it, type='quad')
        return self.gfield.get_quad(it)
    

    def _get_operator(self, filter_operator):
        return operator.Compound([operator.Multiply({"factor": -1j}), filter_operator], spin=2)


class Gradient(SharedFilters):
    subs: List[Union[LensingGradientSub, BirefringenceGradientSub]]
    def __init__(self, subs, ipriormatrix):
        super().__init__(subs[0]) # NOTE I am assuming the ivf and wf class are the same in all gradients
        self.subs: List[Union[LensingGradientSub, BirefringenceGradientSub]] = subs
        self.ipriormatrix = ipriormatrix
        self.component = [comp for sub in self.subs for comp in sub.component]
        self.comp2idx = {comp: i for i, comp in enumerate(self.component)}
        # self.sub1: LensingGradientSub = object.__new__(LensingGradientSub)()
        # self.sub2: BirefringenceGradientSub = object.__new__(BirefringenceGradientSub)()
    
    
    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_total')
    def get_gradient_total(self, it, data=None):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_total(it_, data) for it_ in it]
        totgrad = []
        for sub in self.subs:
            if sub.gfield.is_cached(it, type='total'):
                totgrad.append(sub.gfield.get_total(it))
            else:
                log.info(f'calculating total gradient for {sub.ID}')
                totgrad.append(sub.get_gradient_meanfield(it) - sub.get_gradient_quad(it=it, data=data))
        prior = self.get_gradient_prior(it-1)
        totgrad = [a + b for a, b in zip(totgrad, prior)]
        return totgrad

    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_quad: idx={idx}, it={it}, component={component}, data={data}, idx2={idx2}')
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


    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_prior: idx={idx}, it={it}, component={component}, idx2={idx2}')
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

    
    def _get_est_for_prior(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self._get_est_for_prior(it_) for it_ in it]
        return [sub.gfield._get_est(it=it) for sub in self.subs]
    
    
    def get_gradient_meanfield(self, it):
        if isinstance(it, (list, np.ndarray)):
            return [self.get_gradient_meanfield(it_) for it_ in it]
        return [sub.get_gradient_meanfield(it) for sub in self.subs]


    def update_operator(self, field):
        self.ivf_filter.update_operator(field)


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if hasattr(self.ivf_filter, name):
                return getattr(self.ivf_filter, name)(*args, **kwargs)
            elif hasattr(self.wf_filter, name):
                return getattr(self.wf_filter, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder