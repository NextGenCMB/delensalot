import numpy as np
from os.path import join as opj

from lenspyx.remapping import deflection
from lenspyx.lensing import get_geom 

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy
from delensalot.utils import cli
from delensalot.core.cg import cd_solve

from . import field
from . import gradient
from . import curvature
from . import filter
from . import operator

template_secondaries = ['lensing', 'birefringence']  # Define your desired order
template_index_secondaries = {val: i for i, val in enumerate(template_secondaries)}

class base:
    # def __init__(self, simulationdata, estimator_key, secondaries, filter_desc, gradient_descs, curvature_desc, desc, simidx):
    def __init__(self, simulationdata, estimator_key, simidx, CLfids, itmax, curvature_desc, lenjob_info, lm_maxs, wf_info, noise_info, obs_info, libdir, startingpoint_desc):
        # # this class handles the filter, gradient, and curvature libs, nothing else
        self.data = None 

        self.libdir = libdir
        self.simulationdata = simulationdata
        self.estimator_key = estimator_key
        self.simidx = simidx
        self.CLfids = CLfids
        self.itmax = itmax

        # FIXME cleaner implementation
        analysis_secondary = {}
        if 'p' in estimator_key or 'w' in estimator_key:
            analysis_secondary['lensing'] = [c for c in ['p', 'w'] if c in estimator_key[:2]]
        if 'f' in estimator_key:
            analysis_secondary['birefringence'] = ['f']

        self.secondaries = {
            sec: field.secondary({
                "ID": sec,
                "libdir": opj(self.libdir, 'estimate/'),
                "component": val,
                'fns': {comp: f'klm_{comp}_simidx{{idx}}_it{{it}}' for comp in val},
                'increment_fns': {comp: f'kinclm_{comp}_simidx{{idx}}_it{{it}}' for comp in val},
                'meanfield_fns': {comp: f'kmflm_{comp}_simidx{{idx}}_it{{it}}' for comp in val},
        }) for sec, val in analysis_secondary.items()}
        self.sec2idx = {secondary_ID: idx for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.idx2sec = {idx: secondary_ID for idx, secondary_ID in enumerate(self.secondaries.keys())}
        self.seclist_sorted = sorted(list(self.sec2idx.keys()), key=lambda x: template_index_secondaries.get(x, ''))
        
        lenjob_geomlib = get_geom(('thingauss', {'lmax': 4500, 'smax': 3}))
        zbounds = lenjob_info['zbounds']
        thtbounds = (np.arccos(zbounds[1]), np.arccos(zbounds[0]))
        lenjob_geomlib.restrict(*thtbounds, northsouth_sym=False, update_ringstart=True)
        ffi = deflection(lenjob_geomlib, np.zeros(shape=Alm.getsize(*lm_maxs['LM_max'])), lm_maxs['LM_max'][1], numthreads=8, verbosity=False, epsilon=lenjob_info['epsilon'])
               
        filter_operators = []
        _MAP_operators_desc = {}
        for sec in self.seclist_sorted:
            _MAP_operators_desc[sec] = {
                "LM_max": lm_maxs['LM_max'],
                "component": analysis_secondary[sec],
                "libdir": opj(self.libdir, 'estimate/'),
                "field_fns": self.secondaries[sec].fns, # This must connect to the estimator fields
                "ffi": ffi,}
            if sec == "lensing":
                _MAP_operators_desc[sec]["perturbative"] = False
                filter_operators.append(operator.lensing(_MAP_operators_desc[sec]))
            else:  # birefringence
                filter_operators.append(operator.birefringence(_MAP_operators_desc[sec]))
        sec_operator = operator.secondary_operator(filter_operators)

        MAP_ivf_desc = {
            'ivf_operator': sec_operator,
            'libdir': opj(self.libdir, 'filter/'),
            'beam': operator.beam({"beamwidth": obs_info['beam'], "lm_max": lm_maxs['lm_max_sky']}), # FIXME rewrite in1el in2bl etc.
            "ttebl": obs_info['ttebl'], # FIXME rewrite in1el in2bl etc.
            "lm_max_pri": lm_maxs['lm_max_pri'],
            "lm_max_sky": lm_maxs['lm_max_sky'],
            "nlev": noise_info['nlev'], # FIXME this should belong to a noise model operator
        }
        # it_chain_descr = lambda p2, p5 : [[0, ["diag_cl"], p2, noisemodel_info['nivjob_geominfo'][1]['nside'], np.inf, p5, cd_solve.tr_cg, cd_solve.cache_mem()]]
        MAP_wf_desc = {
            'wf_operator': sec_operator,
            'libdir': opj(self.libdir, 'filter/'),
            'beam': operator.beam({"beamwidth": obs_info['beam'], "lm_max": lm_maxs['lm_max_pri']}), # FIXME rewrite in1el in2bl etc.
            'nlev': noise_info['nlev'], # FIXME rewrite in1el in2bl etc.
            "chain_descr": wf_info['chain_descr'](lm_maxs['lm_max_pri'][0], wf_info['cg_tol']),
            "ttebl": obs_info['ttebl'], # FIXME rewrite in1el in2bl etc.
            "cls_filt": simulationdata.cls_lib.Cl_dict,
            "lm_max_pri": lm_maxs['lm_max_pri'],
            "lm_max_sky": lm_maxs['lm_max_sky'],
            "nlev": noise_info['nlev'], # FIXME this should belong to a noise model operator
        }
        self.ivf_filter = filter.ivf(MAP_ivf_desc)
        self.wf_filter = filter.wf(MAP_wf_desc)
        filters = {'ivf': self.ivf_filter, 'wf': self.wf_filter}

        gradient_descs = {}
        for gradient_name in analysis_secondary.keys():
            gradient_descs.update({ 
                gradient_name: {
                    "ID": gradient_name,
                    'libdir': opj(self.libdir),
                    "lm_max_sky": lm_maxs['lm_max_sky'],
                    "lm_max_pri": lm_maxs['lm_max_pri'],
                    "LM_max": lm_maxs['LM_max'],
                    'itmax': itmax,
                    "ffi": ffi,
                    'sec_operator': sec_operator,
                    'component': analysis_secondary[gradient_name],
                    'chh': [self._chh(self.CLfids[gradient_name][comp*2], lmax=lm_maxs['LM_max'][0], gradient_name=gradient_name) for comp in analysis_secondary[gradient_name]],
            }})
        self.chh = {sec: {comp: self._chh(self.CLfids[sec][comp*2], lmax=lm_maxs['LM_max'][0], gradient_name=sec) for comp in analysis_secondary[sec]} for sec in analysis_secondary.keys()}
        self.gradients = []
        self.gradients.extend(
            getattr(gradient, sec)(gradient_descs[sec], filters)
            for sec in self.seclist_sorted if hasattr(gradient, sec))

        # FIXME this is not guaranteed to be correctly sorted (dicts are not ordered)
        curvature_desc["h0"] = np.array([v for val in self.__get_h0(curvature_desc["Runl0"]).values() for v in val.values()])
        self.curvature = curvature.base(curvature_desc, self.gradients)
        
    
    def get_est(self, simidx, request_it, secondary=None, component=None, scale='k', calc_flag=False):
        current_it = self.maxiterdone()
        if isinstance(request_it, (list,np.ndarray)):
            if all([current_it<reqit for reqit in request_it]): print(f"Cannot calculate new iterations if param 'it' is a list, maximum available iteration is {current_it}")
            # assert not calc_flag and any([current_it<reqit for reqit in request_it]), "Cannot calculate new iterations if it is a list, please set calc_flag=False"
            return [self.get_est(simidx, it, secondary, component, scale=scale, calc_flag=False) for it in request_it[request_it<=current_it]]
        if self.maxiterdone() < 0:
            assert 0, "Could not find the QE starting points, I expected them e.g. at {}".format(self.secondaries['lensing'].libdir)
        if request_it <= current_it: # data already calculated
            if secondary is None:
                return [self.secondaries[secondary].get_est(simidx, request_it, component, scale=scale) for secondary in self.secondaries.keys()]
            elif isinstance(secondary, list):
                return [self.secondaries[sec].get_est(simidx, request_it, component, scale=scale) for sec in secondary]
            else:
                return self.secondaries[secondary].get_est(simidx, request_it, component, scale=scale)
        elif (current_it < self.itmax and request_it >= current_it) or calc_flag:
            if self.data is None: self.data = self.get_data(self.ivf_filter.lm_max_sky)
            for it in range(current_it+1, request_it+1):
                # NOTE it=0 is QE and is implicitly skipped. current_it is the it we have a solution for already
                grad_tot, grad_prev = [], []
                print(f'---------- starting iteration {it} ----------')
                for gradient in self.gradients:
                    print(f'Calculating gradient for {gradient.ID}')
                    self.update_operator(simidx, it-1)
                    wflm = self.wf_filter.get_wflm(simidx, it, self.data)
                    ivfreslm = np.ascontiguousarray(self.ivf_filter.get_ivfreslm(simidx, it, self.data, wflm))
                    grad_tot.append(gradient.get_gradient_total(it, wflm=wflm, ivfreslm=ivfreslm)) #calculates the filtering, the sum, and the quadratic combination
                grad_tot = np.concatenate([np.ravel(arr) for arr in grad_tot])
                if it>=2: #NOTE it=1 cannot build the previous diff, as current diff is QE
                    for gradient in self.gradients:
                        grad_prev.append(gradient.get_gradient_total(it-1))
                    grad_prev = np.concatenate([np.ravel(arr) for arr in grad_prev])
                    self.curvature.add_yvector(grad_tot, grad_prev, simidx, it)
                increment = self.curvature.get_increment(grad_tot, simidx, it)
                prev_klm = np.concatenate([np.ravel(arr) for arr in self.get_est(simidx, it-1, scale=scale)])
                new_klms = self.curvature.grad2dict(increment+prev_klm)
                
                self.cache_klm(new_klms, simidx, it)
            # TODO return a list of requested secondaries and components, not dict
            # return self.load_klm(simidx, it, secondary, component)
            return new_klms if secondary is None else new_klms[secondary] if component is None else new_klms[secondary][component]
        elif current_it < self.itmax and request_it >= current_it and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')
        elif request_it > self.itmax and not calc_flag:
            print(f"Requested iteration {request_it} is beyond the maximum iteration")
            print('If you want to calculate it, set calc_flag=True')
    

    def isiterdone(self, it):
        if it >= 0:
            return np.all([val for sec in self.secondaries.values() for val in sec.is_cached(self.simidx, it)])
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr


    # FIXME this must go to QE
    def __get_h0(self, R_unl0):
        ret = {grad.ID: {} for grad in self.gradients}
        for seci, sec in enumerate(self.seclist_sorted):
            lmax = self.gradients[seci].LM_max[0]
            for comp in self.secondaries[sec].component:
                chh_comp = self.chh[sec][comp]
                buff = cli(R_unl0[sec][comp][:lmax+1] + cli(chh_comp)) * (chh_comp > 0)
                ret[sec][comp] = np.array(buff)
        return ret
    

    def update_operator(self, simidx, it):
        # NOTE updaing a single operator here is enough to update all operators,
        # as they all point to the same operator.lensing and birefringence
        self.ivf_filter.update_operator(simidx, it)
        # self.wf_filter.update_operator(simidx, it)
        # self.gradients[0].update_operator(simidx, it)


    def get_wflm(self, simidx, it):
        return self.wf_filter.get_wflm(simidx, it)

    
    def get_ivfreslm(self, simidx, it):
        return self.ivf_filter.get_ivfreslm(simidx, it)
    

    def get_gradient_quad(self, it=None, secondary=None, component=None, data=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_quad(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_quad(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return [self.gradients[idx].get_gradient_quad(it, component, data) for idx in sec_idx]


    def get_gradient_meanfield(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_meanfield(it, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_meanfield(it, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_meanfield(it, component) for idx in sec_idx])
    

    def get_gradient_prior(self, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_prior(it-1, component) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_prior(it-1, component)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_prior(it-1, component) for idx in sec_idx])
    

    def get_gradient_total(self, data, it=None, secondary=None, component=None):
        if it is None:
            it = self.maxiterdone()
        if secondary is None:
            return [grad.get_gradient_total(it, component, data) for grad in self.gradients]
        if isinstance(secondary, str):
            return self.gradients[self.sec2idx[secondary]].get_gradient_total(it, component, data)
        sec_idx = [self.sec2idx[sec] for sec in secondary]
        return np.array([self.gradients[idx].get_gradient_total(it, component, data) for idx in sec_idx])


    def get_template(self, simidx, it, secondary=None, component=None):
        return self.wf_filter.get_template(simidx, it, secondary, component)


    # exposed functions for job handler
    def cache_klm(self, new_klms, simidx, it):
        for secID, secondary in self.secondaries.items():
            for component in secondary.component:
                secondary.cache_klm(new_klms[secID][component], simidx, it, component=component)


    def get_data(self, lm_max):
        if True: # NOTE anisotropic data currently not supported
        # if self.noisemodel_coverage == 'isotropic':
            # NOTE dat maps must now be given in harmonic space in this idealized configuration. sims_MAP is not used here, as no truncation happens in idealized setting.
            if len(self.estimator_key.split('_'))==1:
                if len(self.estimator_key) == 3:
                    data_key = self.estimator_key[1:]
                elif len(self.estimator_key) == 1:
                    data_key = self.estimator_key
            else:
                data_key = self.estimator_key.split('_')[-1]
            if data_key in ['p', 'eb', 'be']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
            if data_key in ['ee']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)[0]
            elif data_key in ['tt']:
                return alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)
            elif data_key in ['p']:
                EBobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='polarization'),
                    None, *lm_max)
                Tobs = alm_copy(
                    self.simulationdata.get_sim_obs(self.simidx, space='alm', spin=0, field='temperature'),
                    None, *lm_max)         
                ret = np.array([Tobs, *EBobs])
                return ret
            else:
                assert 0, 'implement if needed'
        else:
            if self.k in ['p_p', 'p_eb', 'peb', 'p_be', 'pee']:
                return np.array(self.sims_MAP.get_sim_pmap(self.simidx), dtype=float)
            else:
                assert 0, 'implement if needed'


    def _chh(self, CL, lmax, gradient_name='lensing'):
        if gradient_name == 'lensing':
            return CL[:lmax+1] * (0.5 * np.arange(lmax+1) * np.arange(1, lmax+2))**2
        elif gradient_name == 'birefringence':
            return CL[:lmax+1]
        

    def _copyQEtoDirectory(self, QE_searchs):
        # copies fields and gradient starting points to MAP directory
        # NOTE this turns them into convergence fields
        for secname, secondary in self.secondaries.items():
            QE_searchs[self.sec2idx[secname]].init_filterqest()
            if not all(self.secondaries[secname].is_cached(self.simidx, it=0)):
                klm_QE = QE_searchs[self.sec2idx[secname]].get_est(self.simidx)
                self.secondaries[secname].cache_klm(klm_QE, self.simidx, it=0)
            
            if not self.gradients[self.sec2idx[secname]].gfield.is_cached(self.simidx, it=0):
                kmflm_QE = QE_searchs[self.sec2idx[secname]].get_kmflm(self.simidx)
                self.gradients[self.sec2idx[secname]].gfield.cache_meanfield(kmflm_QE, self.simidx, it=0)

            #TODO cache QE wflm into the filter directory
            if not self.wf_filter.wf_field.is_cached(self.simidx, it=0):
                wflm_QE = QE_searchs[self.sec2idx[secname]].get_wflm(self.simidx, self.ivf_filter.lm_max_pri)
                self.wf_filter.wf_field.cache_field(np.array(wflm_QE), self.simidx, it=0)