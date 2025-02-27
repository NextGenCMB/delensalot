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

class Secondary:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.component = field_desc['component']
        self.fns = get_secondary_fns(self.component)
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.comp2idx = {component: i for i, component in enumerate(self.component)}


    def get_est(self, it, scale='k'):
        # print(f'getting est for {self.ID} with scale {scale}')
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        if self.ID == 'birefringence':
            scale = 'k' # FIXME this is only true between lensing and birefringence.. for other fields this needs to be changed
        # NOTE component are stored with leading dimension
        if isinstance(component, (np.ndarray, list)):
            assert all([comp in self.component for comp in component]), "component must be in {}".format(self.component)
        if isinstance(it, (np.ndarray, list)):
                assert not np.any(np.array(it)<0), 'it must be negative'
  
        if isinstance(it, (np.ndarray, list)):
            ret = []
            for it_ in it:
                if scale == 'd':
                    ret.append([self.klm2dlm(self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_))[0]) for compi, comp in enumerate(component)])
                elif scale == 'k':
                    ret.append(np.array([self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it_))[0] for compi, comp in enumerate(component)]))
        else:
            if scale == 'd':
                ret = np.array([self.klm2dlm(self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it))[0]) for compi, comp in enumerate(component)])
            elif scale == 'k':
                ret = np.array([self.cacher.load(self.fns[comp].format(idx=idx, idx2=idx2, it=it))[0] for compi, comp in enumerate(component)]) 
        return ret


    def cache_klm(self, klm, it):
        ctx, _ = get_computation_context()
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        for ci, comp in enumerate(self.component):
            klm_ = klm[ci] if isinstance(klm, (list, np.ndarray)) else klm[comp] # TODO this can be removed one the get_est from minimizer is fixed
            self.cacher.cache(self.fns[comp].format(idx=idx, idx2=idx2, it=it), np.atleast_2d(klm_))


    def is_cached(self, it):
        ctx, _ = get_computation_context()
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        return [self.cacher.is_cached(self.fns[comp].format(idx=idx, idx2=idx2, it=it)) for comp in component if comp in self.component]
    

    def remove(self, it):
        ctx, _ = get_computation_context()
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        for comp in component:
            self.cacher.remove(opj(self.fns[comp].format(idx=idx, idx2=idx2, it=it)))
    

    def klm2dlm(self, klm):
        Lmax = Alm.getlmax(klm.shape[0], None)
        h2d = cli(0.5 * np.sqrt(np.arange(Lmax + 1) * np.arange(1, Lmax + 2)))
        return almxfl(klm, h2d, Lmax, False)
    

class Gradient:
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


    def _get_est(self, it):
        ctx, _ = get_computation_context()
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        indices = [self.comp2idx[comp] for comp in component]
        if isinstance(it, (np.ndarray, list)):
            ret = []
            for it_ in it:
                retcomp = []
                for compi, comp in enumerate(component):
                    priorlm = self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it_)).squeeze()
                    retcomp.append(priorlm)
                ret.append(np.array(retcomp)[indices])
            return np.array(ret)
        else:
            ret = []
            for compi, comp in enumerate(component):
                priorlm = self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it)).squeeze()
                ret.append(priorlm)
            return np.array(ret)[indices]


    def get_prior(self, it):
        ctx, _ = get_computation_context()
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        indices = [self.comp2idx[comp] for comp in component]
        if isinstance(it, (np.ndarray, list)):
            ret = []
            for it_ in it:
                itret = []
                priorlm = self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it_))[:,indices]
                for compi, comp in enumerate(component):
                    Lmax = Alm.getlmax(priorlm[compi].size, None)
                    almxfl(priorlm[compi].squeeze(), cli(self.chh[comp]), Lmax, True)
                    itret.append(priorlm)
                ret.append(itret)
            return ret
        else:
            priorlm = self.cacher_field.load(self.prior_fns.format(component=comp, idx=idx, idx2=idx2, it=it))[indices]
            for compi, comp in enumerate(component):
                Lmax = Alm.getlmax(priorlm[compi].size, None)
                almxfl(priorlm[compi].squeeze(), cli(self.chh[comp]), Lmax, True)
                return priorlm
        return ret


    def get_meanfield(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        component = component or self.component
        if isinstance(component, str):
            component = [component]
        indices = [self.comp2idx[comp] for comp in component]
        it_ = 0 # NOTE this currently only uses the QE gradient meanfield
        if self.is_cached(it=it, type='meanfield'):
            if isinstance(it, (list, np.ndarray)):
                return np.array([self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_))[:,indices] for _ in it])
            else:
                return self.cacher.load(self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_))[indices]

        else:
            assert 0, f"cannot find meanfield at {self.libdir}/{self.meanfield_fns.format(idx=idx, idx2=idx2, it=it_)}"
            

    def get_total(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        indices = [self.comp2idx[comp] for comp in component]
        if self.is_cached(type='total'):
            if isinstance(it, (np.ndarray, list)):
                return [self.cacher.load(self.total_fns.format(idx=idx, idx2=idx2, it=it_))[:,indices] for it_ in it]
            else:
                return self.cacher.load(self.total_fns.format(idx=idx, idx2=idx2, it=it))[indices]

        g += self.get_meanfield(it=it)
        g -= self.get_quad(it=it)
        g += self.get_prior(it=it-1)
        return g
    

    def get_quad(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2, component = ctx.idx, ctx.idx2 or ctx.idx, ctx.component or self.component
        if isinstance(component, str):
            component = [component]
        indices = [self.comp2idx[comp] for comp in component]
        if self.is_cached(it=it, type='quad'):
            if isinstance(it, (np.ndarray, list)):
                return [self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it_))[:,indices] for it_ in it]
            else:
                ret = self.cacher.load(self.quad_fns.format(idx=idx, idx2=idx2, it=it))
                return ret[indices]
        else:
            assert 0, f"cannot find quad at {self.libdir}/{self.quad_fns.format(idx=idx, idx2=idx2, it=it)}"


    def cache(self, data, it, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx

        file_map = {
            'total': (self.cacher_field, self.total_fns.format(idx=idx, idx2=idx2, it=it)),
            'quad': (self.cacher, self.quad_fns.format(idx=idx, idx2=idx2, it=it)),
            'meanfield': (self.cacher, self.meanfield_fns.format(idx=idx, idx2=idx2, it=it)),
        }

        if type in file_map:
            cacher, filename = file_map[type]
            cacher.cache(filename, data)
        else:
            raise ValueError(f"Unknown cache type: {type}")
    

    def is_cached(self, it, type=None):
        ctx, _ = get_computation_context()
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=0),
        }
        if type is None:
            return all(self.cacher.is_cached(filename) for filename in file_map.values())
        elif type in file_map:
            return self.cacher.is_cached(file_map[type])


    def remove(self, it, type=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        file_map = {
            'total': self.total_fns.format(idx=idx, idx2=idx2, it=it),
            'quad': self.quad_fns.format(idx=idx, idx2=idx2, it=it),
            'meanfield': self.meanfield_fns.format(idx=idx, idx2=idx2, it=0),
        }
        if type is None:
            for typ, filename in file_map.items():
                if self.is_cached(it, typ):
                    self.cacher.remove(filename)
        elif type in file_map:
            if self.is_cached(it, type):
                self.cacher.remove(file_map[type])
    

class Filter:
    def __init__(self, field_desc):
        self.ID = field_desc['ID']
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']

        self.cacher = cachers.cacher_npy(opj(self.libdir))


    def get_field(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        return self.cacher.load(self.fns.format(idx=idx, it=it, idx2=idx2))
    

    def cache(self, fieldlm, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.cacher.cache(self.fns.format(idx=idx, it=it, idx2=idx2), fieldlm)

    
    def is_cached(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        return self.cacher.is_cached(self.fns.format(idx=idx, it=it, idx2=idx2))
    

    def remove(self, it):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        if self.is_cached(it):
            self.cacher.remove(self.fns.format(idx=idx, it=it, idx2=idx2))
    

class Curvature:
    def __init__(self, field_desc):
        self.libdir = field_desc['libdir']
        self.fns =  field_desc['fns']
        self.cacher = cachers.cacher_npy(opj(self.libdir))
        self.types = list(self.fns.keys())


    def get_field(self, it, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        return self.cacher.load(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def cache(self, fieldlm, it, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        self.cacher.cache(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1), fieldlm)

    
    def is_cached(self, it, type):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        return self.cacher.is_cached(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
    

    def remove(self, it, type=None):
        ctx, _ = get_computation_context()  # NOTE getting the singleton instance for MPI rank
        idx, idx2 = ctx.idx, ctx.idx2 or ctx.idx
        if type is None:
            for type in self.types:
                if self.is_cached(it, type):
                    self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
                else:
                    log.info("cannot find field to remove")
        else:
            if self.is_cached(it, type):
                self.cacher.remove(self.fns[type].format(idx=idx, idx2=idx2, it=it, itm1=it-1))
            else:
                log.info("cannot find field to remove")