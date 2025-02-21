from os.path import join as opj
import numpy as np

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy_nd

from delensalot.core import cachers

class secondary:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.component = field_desc['component']
        self.fns = {comp: f'klm_{comp}_simidx{{idx}}_it{{it}}' for comp in self.component}
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.component2idx = {component: i for i, component in enumerate(self.component)}


    def get_est(self, idx, it, component=None, scale='k'):
        if self.ID != 'lensing':
            assert scale == 'k'
        # NOTE component are stored with leading dimension
        if isinstance(component, list):
            assert all([comp in self.component for comp in component]), "component must be in {}".format(self.component)
        if component is None:
            return np.array([self.get_est(idx, it, component, scale).squeeze() for component in self.component])
        if isinstance(component, str): assert component in self.component, f"component must be in {self.component}, but is {component}"
        if it < 0:
            return np.atleast_2d(np.zeros(1, dtype=complex))
        if isinstance(component, list):
            return np.atleast_2d([self.get_est(idx, it, component_, scale).squeeze() for component_i, component_ in enumerate(component)])
        if scale == 'd':
            return np.atleast_2d(self.klm2dlm(self.cacher.load(self.fns[component].format(idx=idx, it=it))[0]))
        elif scale == 'k':
            return np.atleast_2d(self.cacher.load(self.fns[component].format(idx=idx, it=it)))


    def sk2klm(self, idx, it, component):
        rlm = self.cacher.load(self.fns[component].format(idx=idx, it=0))
        for i in range(it):
            rlm += self.hess_cacher.load(self.sk_fns(i))
        return rlm


    def cache_klm(self, klm, idx, it, component=None):
        if component is None:
            for ci, component in enumerate(self.component):
                self.cache_klm(np.atleast_2d(klm[ci]), idx, it, component)
            return
        self.cacher.cache(self.fns[component].format(idx=idx, it=it), np.atleast_2d(klm))


    def is_cached(self, simidx, it, component=None):
        if component is None:
            return np.array([self.is_cached(simidx, it, component_) for component_ in self.component])
        return self.cacher.is_cached(opj(self.fns[component].format(idx=simidx, it=it)))
    

    def remove(self, simidx, it, component=None):
        if component is None:
            if all(np.array([self.is_cached(simidx, it, component_) for component_ in self.component])):
                [self.remove(simidx, it, component_) for component_ in self.component]
            else:
                print("cannot find field to remove")
        else:
            if self.is_cached(simidx, it, component):
                self.cacher.remove(opj(self.fns[component].format(idx=simidx, it=it)))
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
        self.meanfield_fns = field_desc['meanfield_fns']
        self.prior_fns = field_desc['prior_fns']
        self.quad_fns = field_desc['quad_fns']
        self.total_fns = field_desc['total_fns']
        self.total_increment_fns = field_desc['total_increment_fns']
        self.chh = field_desc['chh']
        self.component = field_desc['component']
        self.component2idx = {component: i for i, component in enumerate(self.component)}

        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.cacher_field = cachers.cacher_npy(opj(self.libdir_prior))


    def get_gradient_prior(self, simidx, it, component=None):
        if isinstance(component, list):
            if len(component) == 1:
                component = component[0]
            elif len(component) >1:
                return np.atleast_2d([self.get_gradient_prior(simidx, it, component_).squeeze() for component_i, component_ in enumerate(component)])
        if component is None:
            return np.atleast_2d([self.get_gradient_prior(simidx, it, component_).squeeze() for component_i, component_ in enumerate(self.component)])
        if isinstance(it, list):
            return np.atleast_2d([self.get_gradient_prior(simidx, it_, component).squeeze() for it_ in it])
        if not self.cacher_field.is_cached(self.prior_fns.format(component=component, idx=simidx, it=it)):
            assert 0, "cannot find prior at {}".format(self.cacher_field.lib_dir+"/"+self.prior_fns.format(component=component, idx=simidx, it=it))
        else:
            priorlm = self.cacher_field.load(self.prior_fns.format(component=component, idx=simidx, it=it))
            Lmax = Alm.getlmax(priorlm.size, None)
            almxfl(priorlm.squeeze(), cli(self.chh[component]), Lmax, True)
        return priorlm

    
    def get_meanfield(self, simidx, it, component=None):
        # NOTE this currently only uses the QE gradient meanfield
        if isinstance(it, list):
            return np.array([self.get_meanfield(simidx, 0, component) for _ in it])
        it=0 
        if self.cacher.is_cached(self.meanfield_fns.format(idx=simidx, it=it)):
            if component is None:
                return np.array(self.cacher.load(self.meanfield_fns.format(idx=simidx, it=it)))
            else: 
                if isinstance(component, list):
                    buff = self.cacher.load(self.meanfield_fns.format(idx=simidx, it=it))
                    ret = np.atleast_2d([buff[self.component2idx[component_]] for component_ in component])
                    return ret
                return self.cacher.load(self.meanfield_fns.format(idx=simidx, it=it))[self.component2idx[component]]
        else:
            assert 0, "cannot find meanfield"
            

    def get_total(self, simidx, it, component=None):
        if isinstance(it, list):
            np.array([self.get_total(simidx, it_, component) for it_ in it]) 
        if self.cacher.is_cached(self.total_fns.format(idx=simidx, it=it)):
            return self.cacher.load(self.total_fns.format(idx=simidx, it=it))
        g += self.get_gradient_prior(simidx, it-1, component)
        g += self.get_meanfield(simidx, it, component)
        g -= self.get_quad(simidx, it, component)
        return g
    

    def get_quad(self, simidx, it, component=None):
        if isinstance(component, list):
            if len(component) == 1:
                component = component[0]
            elif len(component) >1:
                return np.atleast_2d([self.get_quad(simidx, it, component_).squeeze() for component_i, component_ in enumerate(component)])
        if isinstance(it, list):
            np.array([self.get_quad(simidx, it_, component) for it_ in it])
        if component is None:
            return self.cacher.load(self.quad_fns.format(idx=simidx, it=it))
        else:
            return self.cacher.load(self.quad_fns.format(idx=simidx, it=it))[self.component2idx[component]]
    

    def isiterdone(self, it):
        return self.cacher.is_cached(self.klm_fns.format(it=it))
    

    def maxiterdone(self):
        it = -2
        isdone = True
        while isdone:
            it += 1
            isdone = self.isiterdone(it + 1)
        return it


    def cache_total(self, totlm, simidx, it):
        self.cacher_field.cache(self.total_fns.format(idx=simidx, it=it), totlm)


    def cache_meanfield(self, kmflm, simidx, it):
        self.cacher.cache(self.meanfield_fns.format(idx=simidx, it=it), kmflm)


    def cache_quad(self, quadlm, simidx, it):
        self.cacher.cache(self.quad_fns.format(idx=simidx, it=it), quadlm)


    def quad_is_cached(self, simidx, it):
        return self.cacher.is_cached(self.quad_fns.format(idx=simidx, it=it))
    

    def is_cached(self, simidx, it, type=None):
        file_map = {
            'total': self.total_fns.format(idx=simidx, it=it),
            'quad': self.quad_fns.format(idx=simidx, it=it),
            'meanfield': self.meanfield_fns.format(idx=simidx, it=it),
        }
        if type is None:
            return all(self.cacher.is_cached(filename) for filename in file_map.values())
        elif type in file_map:
            return self.cacher.is_cached(file_map[type])


    def remove(self, simidx, it, type=None):
        file_map = {
            'total': self.total_fns.format(idx=simidx, it=it),
            'quad': self.quad_fns.format(idx=simidx, it=it),
            'meanfield': self.meanfield_fns.format(idx=simidx, it=it),
        }
        if type is None:
            for typ, filename in file_map.items():
                if self.is_cached(simidx, it, typ):
                    self.cacher.remove(filename)
        elif type in file_map:
            if self.is_cached(simidx, it, type):
                self.cacher.remove(file_map[type])
    

class filter:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_field(self, simidx, it):
        return self.cacher.load(self.fns.format(idx=simidx, it=it))
    

    def cache_field(self, fieldlm, simidx, it):
        self.cacher.cache(self.fns.format(idx=simidx, it=it), fieldlm)

    
    def is_cached(self, simidx, it):
        return self.cacher.is_cached(self.fns.format(idx=simidx, it=it))
    

    def remove(self, simidx, it):
        if self.is_cached(simidx, it):
            self.cacher.remove(self.fns.format(idx=simidx, it=it))
    

class curvature:
    def __init__(self, field_desc):
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']
        self.component = ['grad1d']
        self.increment_fns = {comp: f'kinclm_{comp}_simidx{{idx}}_it{{it}}' for comp in self.component},
        self.meanfield_fns = {comp: f'kmflm_{comp}_simidx{{idx}}_it{{it}}' for comp in self.component},
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.types = list(self.fns.keys())


    def get_field(self, type, simidx, it):
        return self.cacher.load(self.fns[type].format(idx=simidx, it=it, itm1=it-1))
    

    def cache_field(self, fieldlm, type, simidx, it):
        self.cacher.cache(self.fns[type].format(idx=simidx, it=it, itm1=it-1), fieldlm)

    
    def is_cached(self, type, simidx, it):
        return self.cacher.is_cached(self.fns[type].format(idx=simidx, it=it, itm1=it-1))
    

    def remove(self, simidx, it, type=None):
        if type is None:
            for type in self.types:
                if self.is_cached(type, simidx, it):
                    self.cacher.remove(self.fns[type].format(idx=simidx, it=it, itm1=it-1))
                else:
                    print("cannot find field to remove")
        else:
            if self.is_cached(type, simidx, it):
                self.cacher.remove(self.fns[type].format(idx=simidx, it=it, itm1=it-1))
            else:
                print("cannot find field to remove")