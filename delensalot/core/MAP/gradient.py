from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from delensalot.core.handler import DataContainer

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

import numpy as np
from os.path import join as opj
from typing import List, Type, Union
from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd
from delensalot.utils import cli

from delensalot.core.MAP import field
from delensalot.core.MAP import operator

from delensalot.core.MAP.filter import ivf, wf
from delensalot.core.MAP.context import get_computation_context


class SharedFilters:
    def __init__(self, quad):
        self.ivf_filter = quad.ivf_filter
        self.wf_filter = quad.wf_filter


class GradQuad:
    def __init__(self, gradient_desc):
        
        self.component = gradient_desc['component']
        self.ID = gradient_desc['ID']
        self.chh = gradient_desc['chh']
        libdir = gradient_desc['libdir']
        self.data_key = gradient_desc['data_key']

        self.ivf_filter: ivf = gradient_desc.get('ivf_filter', None)
        self.wf_filter: wf = gradient_desc.get('wf_filter', None)

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
        self.gfield = field.gradient(gfield_desc)


    def get_gradient_total(self, data=None, data_leg2=None):
        data_leg2 = data_leg2 or data
        ctx, is_new = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if component is None:
            component = self.gfield.component
        def __compute(it_):
            total = 0
            total += self.get_gradient_prior(it_ - 1)
            total += self.get_gradient_meanfield()
            total -= self.get_gradient_quad()
            return total
        
        if isinstance(it, (list, np.ndarray)):
            for it_ in it:
                if self.gfield.cacher.is_cached(idx=idx, idx2=idx2, it=it_):
                    return self.gfield.get_total(it_, self.LM_max, component)
                else:
                    return __compute(it_)
        else:
            if self.gfield.cacher.is_cached(idx=idx, idx2=idx2, it=it):
                return self.gfield.get_total(it, self.LM_max, component)
            else:
                total = __compute(it)
                self.gfield.cache_total(total, idx, idx2, it)
                return total


    def get_gradient_meanfield(self):
        # NOTE filtering it here as it could be that not all it in list are calculated, so need to calculate them
        ret = []
        return self.gfield.get_meanfield()

    
    def update_operator(self, idx, it):
        self.ivf_filter.update_operator(idx, it)


    def get_gradients_prior(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(it_) for it_ in it])
        return self.gfield.get_gradient_prior()
    

    def get_data(self, idx):
        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            # NOTE dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if self.data_key in ['p', 'eb', 'be']:
                return self.data_container.get_sim_obs(idx, space='alm', spin=0, field='polarization') 
            if self.data_key in ['ee']:
                return self.data_container.get_sim_obs(idx, space='alm', spin=0, field='polarization')
            elif self.data_key in ['tt']:
                return self.data_container.get_sim_obs(idx, space='alm', spin=0, field='temperature')
            elif self.data_key in ['tp']:
                EBobs = self.data_container.get_sim_obs(idx, space='alm', spin=0, field='polarization')
                Tobs = self.data_container.get_sim_obs(idx, space='alm', spin=0, field='temperature')
                ret = np.array([Tobs, *EBobs])
                return ret
            else:
                assert 0, 'implement if needed'
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.simidx), dtype=float)
            else:
                assert 0, 'implement if needed'


    def _copyQEtoDirectory(self, QE_searchs, simidxs):
        # NOTE this turns them into convergence fields
        sec2idx = {QE_search.secondary.ID: i for i, QE_search in enumerate(QE_searchs)}
        QE_search_secondariesIDs = [QE_search.secondary.ID for QE_search in QE_searchs]
        # secondaries = {QE_search.secondary.ID: QE_search.secondary for QE_search in QE_searchs}
        secondaries = [self.ivf_filter.ivf_operator.operators[sec2idx[sec]] for sec in QE_search_secondariesIDs]
        secondaries_field = {
            quad.ID: field.secondary({
                "ID":  quad.ID,
                "component": quad.component,
                "libdir": self.gfield.libdir_prior,
        }) for quad in secondaries}

        for idx in simidxs:
            # idx2 = idx 
            for idx2 in simidxs: # FIXME must implement idx2 for QE_searchs. For now, I fake and generate cross from idx
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


class LensingGradientQuad(GradQuad):
    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self.get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[0].lm_max_in
        

    def get_gradient_quad(self, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        if self.data_container is None:
            assert wflm is not None and ivfreslm is not None, "wflm and ivfreslm must be provided as data container is missing"
        elif idx is not None:
            idx2 = idx2 or idx # NOTE these are the identifier to get the data from the container
        elif data is not None:
            data_leg2 = data_leg2 or data # NOTE these are the data to calculate ivfreslm and wf
        # log.log(logging.DEBUG, 'idx', idx, 'idx2', idx2, 'it', it, ' for gradient_quad lensing')
        if not self.gfield.is_cached(type='quad'):
            if wflm is None:
                assert self.ivf_filter is not None, "ivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                assert self.wf_filter is not None, "wf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wf_filter.get_wflm(self.get_data(idx))
                ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(self.get_data(idx2), wflm))

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
            self.gfield.cache_quad(gc)
        return self.gfield.get_quad()
    

    def get_operator(self, filter_operator):
        lm_max_out = filter_operator.operators[0].lm_max_out
        return operator.joint([operator.spin_raise(lm_max=lm_max_out), filter_operator], out='map')
    

    def cache(self, gfieldlm, idx, it, idx2=None):
        self.gfield.cache(gfieldlm, idx=idx, idx2=idx2, it=it)


    def is_cached(self, idx, it, idx2=None):
        return self.gfield.quad_is_cached(idx=idx, idx2=idx2, it=it)


class BirefringenceGradientQuad(GradQuad):

    def __init__(self, desc):
        super().__init__(desc)
        self.gradient_operator: operator.joint = self.get_operator(desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[0].lm_max_in
    

    def get_gradient_quad(self, data=None, data_leg2=None, wflm=None, ivfreslm=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        data_leg2 = data_leg2 or data
        if not self.gfield.is_cached(type='quad'):
            wflm = self.wf_filter.get_wflm(self.get_data(idx))
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(self.get_data(idx2), wflm))
            
            ivfmap = self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_in[0], self.lm_max_in[1], self.ffi.sht_tr)
            xwfmap = self.gradient_operator.act(wflm, spin=2)
 
            qlms = -4 * (ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms)
        return self.gfield.get_quad()
    

    def get_operator(self, filter_operator):
        return operator.joint([operator.multiply({"factor": -1j}), filter_operator], out='map')


class Joint(SharedFilters):
    quads: List[Union[LensingGradientQuad, BirefringenceGradientQuad]]
    def __init__(self, quads, ipriormatrix):
        super().__init__(quads[0]) # NOTE I am assuming the ivf and wf class are the same in all gradients
        self.quads = quads
        self.ipriormatrix = ipriormatrix
        self.component = [comp for quad in self.quads for comp in quad.component]
        self.comp2idx = {comp: i for i, comp in enumerate(self.component)}
        # self.quad1: LensingGradientQuad = object.__new__(LensingGradientQuad)()
        # self.quad2: BirefringenceGradientQuad = object.__new__(BirefringenceGradientQuad)()
    
    
    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_total')
    def get_gradient_total(self, data=None):
        ctx, isnew = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        # NOTE filtering it here as it could be that not all it in list are calculated, so need to calculate them
        ret = []
        if isinstance(it, (list,np.ndarray)):
            ctx.set(it=it-1)
            prior = self.get_gradient_prior()
            ctx.set(it=it)
            for iti, it_ in enumerate(it):
                ctx.set(it=it_)
                totgrad = []
                for quad in self.quads:
                    if quad.gfield.is_cached(type='total'):
                        totgrad.append(quad.gfield.get_total())
                    else:
                        log.info(f'calculating total gradient for {quad.ID}')
                        totgrad.append(quad.get_gradient_meanfield() - quad.get_gradient_quad(data=data))
                totgrad = [a + b for a, b in zip(totgrad, prior[iti])]
                ret.append(totgrad)
            ctx.set(it=it)
        else:
            totgrad = []
            for quad in self.quads:
                if quad.gfield.is_cached(type='total'):
                    totgrad.append(quad.gfield.get_total())
                else:
                    log.info(f'calculating total gradient for {quad.ID}')
                    totgrad.append(quad.get_gradient_meanfield() - quad.get_gradient_quad(data=data))
            ctx.set(it=it-1)
            prior = self.get_gradient_prior()
            ctx.set(it=it)
            totgrad = [a + b for a, b in zip(totgrad, prior)]
            ret = totgrad
        return ret
    

    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_quad: idx={idx}, it={it}, component={component}, data={data}, idx2={idx2}')
    def get_gradient_quad(self, data=None):
        # NOTE here I need to go through all quads and check it
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        ret = []
        if isinstance(it, (list,np.ndarray)):
            quadgrad = []
            for it_ in it:
                ctx.set(it=it_)
                for quad in self.quads:
                    if quad.gfield.is_cached(type='quad'):
                        quadgrad.append(quad.gfield.get_quad())
                    else:
                        log.info(f'calculating quad gradient for {quad.ID}')
                        quadgrad.append(quad.get_gradient_quad(data=data))
                ret.append(quadgrad)
            ctx.set(it=it)
        else:
            quadgrad = []
            for quad in self.quads:
                if quad.gfield.is_cached(type='quad'):
                    quadgrad.append(quad.gfield.get_quad())
                else:
                    log.info(f'calculating quad gradient for {quad.ID}')
                    quadgrad.append(quad.get_gradient_quad(data=data))
            ret = quadgrad
        return ret


    # @log_on_start(logging.DEBUG, 'Joint.get_gradient_prior: idx={idx}, it={it}, component={component}, idx2={idx2}')
    def get_gradient_prior(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(it, (list, np.ndarray)):
            results = []
            for it_ in it:
                ctx.set(it=it_)
                complambda = lambda quad: [comp for comp in component if comp in quad.component]
                orig = [self.get_est_for_prior(quad.gfield, complambda(quad)) for quad in self.quads]
                original_shapes = [arr.shape for arr in orig]
                est = np.vstack(orig)
        
                result = []
                for xi, x in enumerate(self.ipriormatrix):
                    prod_ = sum(almxfl(est[yi], self.ipriormatrix[xi, yi], None, False) for yi, y in enumerate(x))
                    result.append(prod_)

                sec_result = []
                start = 0
                for shape in original_shapes:
                    num_rows = shape[0]
                    sec_result.append(np.array(result[start:start + num_rows]))
                    start += num_rows
                results.append(sec_result)
            ctx.set(it=it)
            return results  # return list of results for each iteration in 'it'
        
        complambda = lambda quad: [comp for comp in component if comp in quad.component]
        orig = [self.get_est_for_prior(quad.gfield, complambda(quad)) for quad in self.quads]
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

    
    def get_gradient_meanfield(self):
        return [quad.get_gradient_meanfield() for quad in self.quads]

    
    def get_est_for_prior(self, gfield, component):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2 = ctx.it, ctx.idx, ctx.idx2 or ctx.idx
        if isinstance(component, str):
            component = [component]
        if isinstance(it, list):
            ret = []
            for it_ in it:
                ret.append([np.array([gfield.cacher_field.load(gfield.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it)).squeeze()]) for comp in component])
        else:
            ret = np.array([gfield.cacher_field.load(gfield.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it)).squeeze() for comp in component])
        return ret


    def update_operator(self, idx, it):
        self.ivf_filter.update_operator(idx, it)


    def __getattr__(self, name):
        # NOTE this forwards the method call to the gradient_lib
        def method_forwarder(*args, **kwargs):
            if hasattr(self.ivf_filter, name):
                return getattr(self.ivf_filter, name)(*args, **kwargs)
            elif hasattr(self.wf_filter, name):
                return getattr(self.wf_filter, name)(*args, **kwargs)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        return method_forwarder