import numpy as np
from os.path import join as opj
from typing import List, Type, Union
from lenspyx.remapping.deflection_028 import rtype, ctype

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy, almxfl_nd
from delensalot.utils import cli

from delensalot.core.MAP import field
from delensalot.core.MAP import operator

from delensalot.core.handler import DataContainer
from delensalot.core.MAP.filter import ivf, wf


class SharedFilters:
    def __init__(self, quad):
        self.ivf_filter = quad.ivf_filter
        self.wf_filter = quad.wf_filter


class GradQuad:
    def __init__(self, gradient_desc):
        self.ID = gradient_desc['ID']
        libdir = gradient_desc['libdir']

        self.data_key = gradient_desc['data_key']

        self.ivf_filter: ivf = gradient_desc.get('ivf_filter', None)
        self.wf_filter: wf = gradient_desc.get('wf_filter', None)

        self.ffi = gradient_desc['sec_operator'].operators[0].ffi # each sec operator has the same ffi, so can pick any
        self.chh = gradient_desc['chh']
        self.component = gradient_desc['component']

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


    def get_gradient_total(self, it, component=None, data=None, data_leg2=None, idx=None, idx2=None):
        data_leg2 = data_leg2 or data
        idx2 = idx2 or idx
        if component is None:
            component = self.gfield.component
        if isinstance(it, (list,np.ndarray)):
            return np.array([self.get_gradient_total(it_, component, data, data_leg2, idx=None, idx2=None, wflm=None, ivfreslm=None) for it_ in it])
        if self.gfield.cacher.is_cached(idx=idx, idx2=idx2, it=it):
            return self.gfield.get_total(it, self.LM_max, component)
        else:
            g = 0
            g += self.get_gradient_prior(it-1, idx, idx2, component)
            g += self.get_gradient_meanfield(it, idx, idx, component)
            g -= self.get_gradient_quad(it, idx, idx2, component, data)
            # self.gfield.cache_total(g, simidx, it) # NOTE this is implemented, but not used to save disk space
            return g


    def get_gradient_quad(self, it, idx, component=None, idx2=None):
        assert 0, "subclass this"


    def get_gradient_meanfield(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_meanfield(idx, it_, component, idx2=idx2) for it_ in it])
        return self.gfield.get_meanfield(idx, it, component=component, idx2=idx2)
    

    def update_operator(self, it):
        self.ivf_filter.update_operator(it)


    def get_gradients_prior(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(it_, component) for it_ in it])
        return self.gfield.get_gradient_prior(idx, it, component=component, idx2=idx2)
    

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
    def __init__(self, gradient_desc):
        super().__init__(gradient_desc)
        self.gradient_operator: operator.joint = self.get_operator(gradient_desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[0].lm_max_in


    def get_gradient_quad(self, it, component=None, data=None, data_leg2=None, idx=None, idx2=None, wflm=None, ivfreslm=None):
        # NOTE this function is equation 22 of the paper (for lensing).
        # Using property _2Y = _-2Y.conj
        # res = ivf.conj * gpmap(3) - ivf * gpmap(1).conj
        if self.data_container is None:
            assert wflm is not None and ivfreslm is not None, "wflm and ivfreslm must be provided as data container is missing"
        elif idx is not None:
            idx2 = idx2 or idx # NOTE these are the identifier to get the data from the container
        elif data is not None:
            data_leg2 = data_leg2 or data # NOTE these are the data to calculate ivfreslm and wf

        if not self.gfield.quad_is_cached(idx=idx, idx2=idx2, it=it):
            if wflm is None:
                assert self.ivf_filter is not None, "ivf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                assert self.wf_filter is not None, "wf_filter must be provided at instantiation in absence of wflm and ivfreslm"
                wflm = self.wf_filter.get_wflm(it, self.get_data(idx))
                ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2), wflm))

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
            self.gfield.cache_quad(gc, idx=idx, idx2=idx2, it=it)
        return self.gfield.get_quad(idx=idx, idx2=idx2, it=it, component=component)
    

    def get_operator(self, filter_operator):
        lm_max_out = filter_operator.operators[0].lm_max_out
        return operator.joint([operator.spin_raise(lm_max=lm_max_out), filter_operator], out='map')
    

    def cache(self, gfieldlm, idx, it, idx2=None):
        self.gfield.cache(gfieldlm, idx=idx, idx2=idx2, it=it)


    def is_cached(self, idx, it, idx2=None):
        return self.gfield.quad_is_cached(idx=idx, idx2=idx2, it=it)


class BirefringenceGradientQuad(GradQuad):

    def __init__(self, gradient_desc):
        super().__init__(gradient_desc)
        self.gradient_operator: operator.joint = self.get_operator(gradient_desc['sec_operator'])
        self.lm_max_in = self.gradient_operator.operators[-1].operators[0].lm_max_in
    

    def get_gradient_quad(self, it, component=None, data=None, data_leg2=None, idx=None, idx2=None, wflm=None, ivfreslm=None):
        idx2 = idx2 or idx
        data_leg2 = data_leg2 or data
        if not self.gfield.quad_is_cached(idx=idx, it=it, idx2=idx2):
            wflm = self.wf_filter.get_wflm(it, self.get_data(idx))
            ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(it, self.get_data(idx2), wflm))
            
            ivfmap = self.ffi.geom.synthesis(ivfreslm, 2, self.lm_max_in[0], self.lm_max_in[1], self.ffi.sht_tr)
            xwfmap = self.gradient_operator.act(wflm, spin=2)
 
            qlms = -4 * (ivfmap[0]*xwfmap[1] - ivfmap[1]*xwfmap[0])
            qlms = self.ffi.geom.adjoint_synthesis(qlms, 0, self.LM_max[0], self.LM_max[1], self.ffi.sht_tr)
            
            self.gfield.cache_quad(qlms, idx=idx, it=it, idx2=idx2)
        return self.gfield.get_quad(idx=idx, it=it, idx2=idx2, component=component)
    

    def get_operator(self, filter_operator):
        return operator.joint([operator.multiply({"factor": -1j}), filter_operator], out='map')

class Joint(SharedFilters):
    quads: List[LensingGradientQuad, BirefringenceGradientQuad]
    def __init__(self, quads: List[LensingGradientQuad, BirefringenceGradientQuad], ipriormatrix):
        super().__init__(quads[0]) # NOTE I am assuming the ivf and wf class are the same in all gradients
        self.quads:  List[LensingGradientQuad, LensingGradientQuad] = quads
        self.ipriormatrix = ipriormatrix
        self.component = [comp for quad in self.quads for comp in quad.component]
        self.quad1: LensingGradientQuad = object.__new__(LensingGradientQuad)()
        self.quad2: BirefringenceGradientQuad = object.__new__(BirefringenceGradientQuad)()
    
    
    def get_gradient_total(self, idx, it, component=None, data=None, idx2=None):
        idx2 = idx2 or idx
        component = component or self.component
        if isinstance(it, (list,np.ndarray)):
            return np.array([self.get_gradient_total(idx, it_, component, data, idx2) for it_ in it])
        totgrad = []
        for quad in self.quads:
            print(f'calculating total gradient for {quad.ID}')
            component_ = [comp for comp in component if comp in quad.component]
            if quad.gfield.is_cached(idx=idx, it=it, type='total', idx2=idx2):
                totgrad.append(quad.gfield.get_total(idx, it, component_, idx2=idx2))
            else:
                totgrad.append(quad.get_gradient_meanfield(idx, it, component_, idx2=idx2) - quad.get_gradient_quad(it, component_, data, idx=idx, idx2=idx2))
        prior = self.get_gradient_prior(idx, it-1, component, idx2=idx2)
        totgrad = [a + b for a, b in zip(totgrad, prior)]
        return totgrad 
    

    def get_gradient_quad(self, idx, it, component=None, data=None, idx2=None):
        idx2 = idx2 or idx
        component = component or self.component
        if isinstance(it, (list,np.ndarray)):
            return [self.get_gradient_total(idx, it_, component, data, idx2) for it_ in it]
        quadgrad = []
        for quad in self.quads:
            component_ = [comp for comp in component if comp in quad.component]
            if quad.gfield.is_cached(idx=idx, it=it, type='quad', idx2=idx2):
                quadgrad.append(quad.gfield.get_quad(idx, it, component_, idx2=idx2))
            else:
                quadgrad.append(quad.get_gradient_quad(it, component_, data, idx=idx, idx2=idx2))
        return quadgrad 


    def get_gradient_prior(self, idx, it, component=None, idx2=None):
        component = component or self.component
        idx2 = idx2 or idx
        if isinstance(it, (list, np.ndarray)):
            return np.array([self.get_gradient_prior(idx, it_, component, idx2) for it_ in it])


        complambda = lambda quad: [comp for comp in component if comp in quad.component]
        orig = [self.get_est(quad.gfield, idx, it, complambda(quad), idx2) for quad in self.quads]
        original_shapes = [arr.shape for arr in orig]
        est = np.vstack(orig)
    
        result = []
        for xi, x in enumerate(self.ipriormatrix):
            prod_ = 0
            for yi, y in enumerate(x):
                prod_ += almxfl(est[yi],self.ipriormatrix[xi,yi], None, False)
            result.append(prod_)

        final_result = []
        start = 0
        for shape in original_shapes:
            num_rows = shape[0]
            final_result.append(result[start:start + num_rows]) 
            start += num_rows
        return final_result
    

    def get_est(self, gfield, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        it = 0 # NOTE setting it=0 for now
        if isinstance(component, list):
            if len(component) == 1:
                component = component[0]
            elif len(component) >1:
                return np.atleast_2d([self.get_est(gfield, idx, it, component_, idx2=idx2).squeeze() for component_i, component_ in enumerate(component)])
        if component is None:
            return np.atleast_2d([self.get_est(gfield, idx, it, component_, idx2=idx2).squeeze() for component_i, component_ in enumerate(self.component)])
        if isinstance(it, list):
            return np.atleast_2d([self.get_est(gfield, idx, it_, component, idx2=idx2).squeeze() for it_ in it])
        if not gfield.cacher_field.is_cached(gfield.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it)):
            assert 0, "cannot find secondary for prior at {}".format(gfield.cacher_field.lib_dir+"/"+gfield.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it))
        else:
            priorlm = gfield.cacher_field.load(gfield.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it))
        return priorlm


    def update_operator(self, it):
        self.ivf_filter.update_operator(it)