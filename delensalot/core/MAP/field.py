import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from os.path import join as opj
import numpy as np

from delensalot.utils import cli
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy_nd

from delensalot.core import cachers
from delensalot.core.MAP.context import get_computation_context

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
        self


    def get_est(self, scale='k'):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if self.ID != 'lensing':
            assert scale == 'k' # FIXME this is only true between lensing and birefringence.. for other fields this needs to be changed
        # NOTE component are stored with leading dimension
        if isinstance(component, (np.ndarray, list)):
            assert all([comp in self.component for comp in component]), "component must be in {}".format(self.component)
        if isinstance(it, (np.ndarray, list)):
                assert not np.any(np.array(it)<0), 'it must be negative'
  
        if isinstance(it, (np.ndarray, list)):
            ret = []
            for it_ in it:
                if scale == 'd':
                    ret.append(np.array([self.klm2dlm(self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_))[0]) for compi, comp in enumerate(component)]))
                elif scale == 'k':
                    ret.append(np.array([self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_)) for compi, comp in enumerate(component)]))
        else:
            if scale == 'd':
                ret = np.array([self.klm2dlm(self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_))[0]) for compi, comp in enumerate(component)])
            elif scale == 'k':
                ret = np.array([self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_)) for compi, comp in enumerate(component)]) 
        return ret



    def cache_klm(self, klm):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        for ci, comp in enumerate(self.component):
            self.cacher.cache(self.fns[comp].format(idx=idx, idx2=idx2, it=it), np.atleast_2d(klm[ci]))


    def is_cached(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        return [self.cacher.is_cached(self.fns[comp].format(idx=idx, idx2=idx2, it=it)) for comp in component if comp in self.component]
    

    def remove(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        for comp in component:
            self.cacher.remove(opj(self.fns[component].format(idx=idx, idx2=idx2, it=it)))
    

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
        self.comp2idx = {component: i for i, component in enumerate(self.component)}

        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.cacher_field = cachers.cacher_npy(opj(self.libdir_prior))


    def get_prior(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        # NOTE here i can check for all components and idx and it
        if isinstance(it, (np.ndarray, list)):
            ret = []
            for compi, comp in enumerate(component):
                priorlm = [self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it_)) for it_ in it]
                Lmax = Alm.getlmax(priorlm[compi].size, None)
                almxfl(priorlm[compi].squeeze(), cli(self.chh[comp]), Lmax, True)
                ret.append(priorlm)
        else:
            for compi, comp in enumerate(component):
                priorlm = self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it))
                Lmax = Alm.getlmax(priorlm[compi].size, None)
                almxfl(priorlm[compi].squeeze(), cli(self.chh[comp]), Lmax, True)
                ret = priorlm
        return ret


    def get_meanfield(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        component = component or self.component
        if isinstance(component, str):
            component = [component]
        it_ = 0 # NOTE this currently only uses the QE gradient meanfield
        if self.is_cached(type='meanfield'):
            indices = [self.comp2idx[comp] for comp in component]
            if isinstance(it, (list, np.ndarray)):
                meanlm = np.array([self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_)) for _ in it])
                return meanlm[:,indices]
            else:
                meanlm = self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_))
                return meanlm[indices]
        else:
            assert 0, f"cannot find meanfield at {self.libdir}/{self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_)}"
            

    def get_total(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if self.is_cached(type='total'):
            totallm = self.cacher.load(self.total_fns.format(idx=idx, idx2=idx2, it=it))
            return np.atleast_2d([totallm[self.comp2idx[comp]] for comp in component])
        g += self.get_meanfield()
        g -= self.get_quad()
        ctx.set(it=it-1)
        g += self.get_prior()
        ctx.set(it=it)
        return g
    

    def get_quad(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        # NOTE here i can check for all components and idx and it
        indices = [self.comp2idx[comp] for comp in component]
        if isinstance(it, (np.ndarray, list)):
            return [self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it_))[:, indices] for it_ in it]
        else:
            ret = self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it))
            return ret[indices]


    def cache_total(self, totlm):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        self.cacher_field.cache(self.total_fns.format(idx=idx, idx2=idx2, it=it), totlm)


    def cache_meanfield(self, kmflm):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        self.cacher.cache(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it), kmflm)


    def cache_quad(self, quadlm):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        self.cacher.cache(self.quad_fns.format(idx=idx, idx2=idx2, it=it), quadlm)
    

    def is_cached(self, type=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=0),
        }
        if type is None:
            return all(self.cacher.is_cached(filename) for filename in file_map.values())
        elif type in file_map:
            return self.cacher.is_cached(file_map[type])


    def remove(self, type=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=0),
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


    def get_field(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component
        return self.cacher.load(self.fns.format(idx=idx, it=it))
    

    def cache(self, fieldlm):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component
        self.cacher.cache(self.fns.format(idx=idx, it=it), fieldlm)

    
    def is_cached(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component
        return self.cacher.is_cached(self.fns.format(idx=idx, it=it))
    

    def remove(self):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component
        if self.is_cached():
            self.cacher.remove(self.fns.format(idx=idx, it=it))
    

class curvature:
    def __init__(self, field_desc):
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']
        self.component = ['grad1d']
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.types = list(self.fns.keys())


    def get_field(self, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        return self.cacher.load(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def cache(self, fieldlm, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        self.cacher.cache(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1), fieldlm)

    
    def is_cached(self, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        return self.cacher.is_cached(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def remove(self, type=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        it, idx, idx2, component = ctx.it, ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if type is None:
            for type in self.types:
                if self.is_cached(type):
                    self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
                else:
                    log.info("cannot find field to remove")
        else:
            if self.is_cached(type):
                self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
            else:
                log.info("cannot find field to remove")