import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import almxfl

from . import gradient
from . import curvature

class base:
    def __init__(self, fields, filter_desc, gradient_descs, curvature_desc, desc, simidx):
        # this class handles the filter, gradient, and curvature libs, nothing else
        self.fields = fields
        # NOTE gradient and curvature share the field increments, so naming must be consistent. Can use the gradient_descs['inc_fn'] for this
        self.gradients = [gradient(gradient_desc, filter_desc) for gradient_desc in gradient_descs]
        self.curvature = curvature(curvature_desc, self.gradients)
        self.itmax = desc.get('itmax')
        self.simidx = simidx
        self.template_cacher = cachers.cache_npy(desc['template_cacher'])
        self.template_operators = desc['filter_operators']


    # i need get_klm for certain iteration, current iteration, and final iteration, 
    def get_klm_it(self, it, field):
        # if field a parameter, do i want to iterate over param only, or the full iterator?
        current_iter = self.maxiterdone()
        for it in range(current_iter, self.itmax):
            for gradient, field in zip(self.gradients, self.fields):
                curr_klm = self.get_klm_it(it, field)
                self.update_operators(curr_klm)
                gradient = self.get_gradient_total(it) #calculates the filtering, the sum, and the quadratic combination
                self.cac
            H = self.update_curvature(gradient)
            self.update_MAP(H)
        return self.fields[field].get_klm(idx=self.simidx, it=it)


    def get_template(self, field):
        fn_blt = self.template_cacher.get_fn(self.template_operators[field])
        if not self.template_cacher.is_cached(self.simidx):
            self.template_operator.update_field(self.template_operators[field])
            # almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)
            blm = self.template_operator.act(field)
            self.blt_cacher.cache(fn_blt, blm)
        return self.template_cacher.load(fn_blt)
    

    def get_meanfield_it(self, it, calc=False):
        # fn = opj(self.mf_dirname, 'mf%03d_it%03d.npy'%(self.Nmf, it))
        # if not calc:
        #     if os.path.isfile(fn):
        #         mf = np.load(fn)
        #     else:
        #         mf = self.get_meanfield_it(self, it, calc=True)
        # else:
        #     plm = rec.load_plms(self.libdir_MAP(self.k, self.simidxs[0], self.version), [0])[-1]
        #     mf = np.zeros_like(plm)
        #     for simidx in self.simidxs_mf:
        #         log.info("it {:02d}: adding sim {:03d}/{}".format(it, simidx, self.Nmf-1))
        #         mf += rec.load_plms(self.libdir_MAP(self.k, simidx, self.version), [it])[-1]
        #     np.save(fn, mf/self.Nmf)
        return None
    


    def get_WF(self, field):
        self.gradients[field].get_WF(field)

    
    def get_ivf(self, field):
        self.gradients[field].get_ivf(field)


    def isiterdone(self, it):
        return self.cacher.is_cached(self.klm_fns.format(it=it))
    

    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr
    

    def _get_current_MAPpoint(self):
        for field in self.fields:
            comps = []
            for component in field.components:
                comps.append(field.get_klm(self.maxiterdone() - 1, component))
        self.klm_currs = np.array(comps)
        return self.klm_currs


    def _get_current_klms(self, it):
        buff = []
        for field in self.fields:
            for component in field.components:
                buff.append(field.get_klm(it, component))
        return np.array(buff)


    def _update_operators(self, fields):
        # For each operator that is dependent on a field, we need to update the field
        for gradient in self.gradients:
            gradient.update_field(fields)
        for curvature in self.curvature:
            curvature.update_field(fields)
            

    def _get_gradient(self, curr_MAPp):
        gradients = []
        for gi, gradient in enumerate(self.gradients):
            gradients.append(gradient.calc_gradient(len(curr_MAPp[gi]), self.maxiterdone() - 1), self.maxiterdone() - 1) # self.klm_currs[gi]
        return np.array(gradients)


    def _update_curvature(self, gradient):
        self.curvature.update_curvature(gradient) # This updates the vectors to be used for the curvature calculation


    def _update_MAP(self, H):
        deltag = self.curvature.get_gradient_inc(self.klm_currs) # This calls the 2-loop curvature update
        for field in self.fields:
            for component in field.components:
                increment = field.calc_increment(deltag, component)
                field.update_klm(increment, component) 