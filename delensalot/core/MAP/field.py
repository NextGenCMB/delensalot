from os.path import join as opj
import numpy as np

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy_nd

from delensalot.core import cachers

def get_secondary_fns(component):
    return {comp: f'klm_{comp}_idx{{idx}}_{{idx2}}_it{{it}}' for comp in component}

class secondary:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.component = field_desc['component']
        self.fns = get_secondary_fns(self.component)
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.component2idx = {component: i for i, component in enumerate(self.component)}


    def get_est(self, idx, it, component=None, scale='k', idx2=None):
        idx2 = idx2 or idx
        if self.ID != 'lensing':
            assert scale == 'k'
        # NOTE component are stored with leading dimension
        if isinstance(component, list):
            assert all([comp in self.component for comp in component]), "component must be in {}".format(self.component)
        if component is None:
            return np.array([self.get_est(idx, it, component, scale, idx2=idx2).squeeze() for component in self.component])
        if isinstance(component, str): assert component in self.component, f"component must be in {self.component}, but is {component}"
        if it < 0:
            return np.atleast_2d(np.zeros(1, dtype=complex))
        if isinstance(component, list):
            return np.atleast_2d([self.get_est(idx, it, component_, scale, idx2=idx2).squeeze() for component_i, component_ in enumerate(component)])
        if scale == 'd':
            return np.atleast_2d(self.klm2dlm(self.cacher.load(self.fns[component].format(idx=idx, idx2=idx2, it=it))[0]))
        elif scale == 'k':
            return np.atleast_2d(self.cacher.load(self.fns[component].format(idx=idx, idx2=idx2, it=it)))


    def sk2klm(self, idx, it, component):
        rlm = self.cacher.load(self.fns[component].format(idx=idx, it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm


    def cache_klm(self, klm, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if component is None:
            for ci, component in enumerate(self.component):
                self.cache_klm(np.atleast_2d(klm[ci]), idx, it, component, idx2=idx2)
            return
        self.cacher.cache(self.fns[component].format(idx=idx, idx2=idx2, it=it), np.atleast_2d(klm))


    def is_cached(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if component is None:
            return np.array([self.is_cached(idx, it, component_, idx2=idx2) for component_ in self.component])
        return self.cacher.is_cached(opj(self.fns[component].format(idx=idx, idx2=idx2, it=it)))
    

    def remove(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if component is None:
            if all(np.array([self.is_cached(idx, it, component_, idx2=idx2) for component_ in self.component])):
                [self.remove(idx, it, component_, idx2=idx2) for component_ in self.component]
            else:
                print("cannot find field to remove")
        else:
            if self.is_cached(idx, it, component):
                self.cacher.remove(opj(self.fns[component].format(idx=idx, it=it)))
            else:
                print('cannot find field to remove')
    

    def klm2dlm(self, klm):
        Lmax = Alm.getlmax(klm.shape[0])
        h2d = cli(0.5 * np.sqrt(np.arange(Lmax + 1) * np.arange(1, Lmax + 2)))
        return almxfl(klm, h2d, Lmax, False)
    

class gradient:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.libdir_prior = field_desc['libdir_prior']

        self.meanfield_fns = f'mf_glm_{self.ID}_idx{{idx}}_{{idx2}}_it{{it}}'
        self.quad_fns = f'quad_glm_{self.ID}_idx{{idx}}_{{idx2}}_it{{it}}'
        self.prior_fns = 'klm_{component}_idx{idx}_{idx2}_it{it}' # NOTE prior is just field, and then we do a simple divide by spectrum (almxfl)
        self.total_increment_fns = f'ginclm_{self.ID}_idx{{idx}}_{{idx2}}_it{{it}}'   
        self.total_fns = f'gtotlm_{self.ID}_idx{{idx}}_{{idx2}}_it{{it}}'

        self.chh = field_desc['chh']
        self.component = field_desc['component']
        self.component2idx = {component: i for i, component in enumerate(self.component)}

        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.cacher_field = cachers.cacher_npy(opj(self.libdir_prior))


    def get_gradient_prior(self, idx, it, component=None, idx2=None):
        idx2 = idx2 or idx
        if isinstance(component, list):
            if len(component) == 1:
                component = component[0]
            elif len(component) >1:
                return np.atleast_2d([self.get_gradient_prior(idx, it, component_, idx2=idx2).squeeze() for component_i, component_ in enumerate(component)])
        if component is None:
            return np.atleast_2d([self.get_gradient_prior(idx, it, component_, idx2=idx2).squeeze() for component_i, component_ in enumerate(self.component)])
        if isinstance(it, list):
            return np.atleast_2d([self.get_gradient_prior(idx, it_, component, idx2=idx2).squeeze() for it_ in it])
        if not self.cacher_field.is_cached(self.prior_fns.format(component=component, idx=idx, it=it)):
            assert 0, "cannot find secondary for prior at {}".format(self.cacher_field.lib_dir+"/"+self.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it))
        else:
            priorlm = self.cacher_field.load(self.prior_fns.format(component=component, idx=idx, idx2=idx2, it=it))
            Lmax = Alm.getlmax(priorlm.size, None)
            almxfl(priorlm.squeeze(), cli(self.chh[component]), Lmax, True)
        return priorlm

    
    def get_meanfield(self, idx, it, component=None, idx2=None):
        # NOTE this currently only uses the QE gradient meanfield
        if isinstance(it, list):
            return np.array([self.get_meanfield(idx, 0, component) for _ in it])
        it=0
        if self.cacher.is_cached(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it)):
            if component is None:
                return np.array(self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it)))
            else: 
                if isinstance(component, list):
                    buff = self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it))
                    ret = np.atleast_2d([buff[self.component2idx[component_]] for component_ in component])
                    return ret
                return self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it))[self.component2idx[component]]
        else:
            assert 0, f"cannot find meanfield at {self.meanfield_fns.format(idx=idx, idx2=idx2, it=it)}"
            

    def get_total(self, idx, it, component=None, idx2=None):
        if isinstance(it, list):
            np.array([self.get_total(idx, it_, component, idx2=idx2) for it_ in it]) 
        if self.cacher.is_cached(self.total_fns.format(idx=idx, idx2=idx2, it=it)):
            return self.cacher.load(self.total_fns.format(idx=idx, idx2=idx2, it=it))
        g += self.get_gradient_prior(idx, it-1, component, idx2=idx2)
        g += self.get_meanfield(idx, it, component, idx2=idx2)
        g -= self.get_quad(idx, it, component, idx2=idx2)
        return g
    

    def get_quad(self, idx, it, component=None, idx2=None):
        if isinstance(component, list):
            if len(component) == 1:
                component = component[0]
            elif len(component) >1:
                return np.atleast_2d([self.get_quad(idx, it, component_, idx2=idx2).squeeze() for component_i, component_ in enumerate(component)])
        if isinstance(it, list):
            np.array([self.get_quad(idx, it_, component, idx2=idx2) for it_ in it])
        if component is None:
            return self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it))
        else:
            return self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it))[self.component2idx[component]]


    def cache_total(self, totlm, idx, it, idx2=None):
        idx2 = idx2 or idx
        self.cacher_field.cache(self.total_fns.format(idx=idx, idx2=idx2, it=it), totlm)


    def cache_meanfield(self, kmflm, idx, it, idx2=None):
        idx2 = idx2 or idx
        self.cacher.cache(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it), kmflm)


    def cache_quad(self, quadlm, idx, it, idx2=None):
        idx2 = idx2 or idx
        self.cacher.cache(self.quad_fns.format(idx=idx, idx2=idx2, it=it), quadlm)


    def quad_is_cached(self, idx, it, idx2=None):
        idx2 = idx2 or idx
        return self.cacher.is_cached(self.quad_fns.format(idx=idx, idx2=idx2, it=it))
    

    def is_cached(self, idx, it, type=None, idx2=None):
        idx2 = idx2 or idx
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=it),
        }
        if type is None:
            return all(self.cacher.is_cached(filename) for filename in file_map.values())
        elif type in file_map:
            return self.cacher.is_cached(file_map[type])


    def remove(self, idx, it, type=None, idx2=None):
        idx2 = idx2 or idx
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=it),
        }
        if type is None:
            for typ, filename in file_map.items():
                if self.is_cached(idx, it, typ):
                    self.cacher.remove(filename)
        elif type in file_map:
            if self.is_cached(idx, it, type):
                self.cacher.remove(file_map[type])
    

class filter:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_field(self, idx, it, idx2=None):
        return self.cacher.load(self.fns.format(idx=idx, it=it))
    

    def cache(self, fieldlm, idx, it, idx2=None):
        self.cacher.cache(self.fns.format(idx=idx, it=it), fieldlm)

    
    def is_cached(self, idx, it, idx2=None):
        return self.cacher.is_cached(self.fns.format(idx=idx, it=it))
    

    def remove(self, idx, it, idx2=None):
        if self.is_cached(idx=idx, it=it):
            self.cacher.remove(self.fns.format(idx=idx, it=it))
    

class curvature:
    def __init__(self, field_desc):
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']
        self.component = ['grad1d']
        self.increment_fns = {comp: f'kinclm_{comp}_idx{{idx}}_it{{it}}' for comp in self.component},
        self.meanfield_fns = {comp: f'kmflm_{comp}_idx{{idx}}_it{{it}}' for comp in self.component},
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.types = list(self.fns.keys())


    def get_field(self, type, idx, it, idx2=None):
        idx2 = idx2 or idx
        return self.cacher.load(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def cache_field(self, fieldlm, type, idx, it, idx2=None):
        idx2 = idx2 or idx
        self.cacher.cache(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1), fieldlm)

    
    def is_cached(self, type, idx, it, idx2=None):
        idx2 = idx2 or idx
        return self.cacher.is_cached(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def remove(self, idx, it, type=None, idx2=None):
        idx2 = idx2 or idx
        if type is None:
            for type in self.types:
                if self.is_cached(type, idx, it, idx2=None):
                    self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
                else:
                    print("cannot find field to remove")
        else:
            if self.is_cached(type, idx, it, idx2=None):
                self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
            else:
                print("cannot find field to remove")