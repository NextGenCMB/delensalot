import numpy as np

from delensalot.core import cachers
from delensalot.utility.utils_hp import almxfl, alm_copy, Alm
from delensalot.utils import cli

from . import gradient
from . import curvature
from . import filter

class base:
    def __init__(self, simulationdata, fields, filter_desc, gradient_descs, curvature_desc, desc, template_desc, simidx):
        # this class handles the filter, gradient, and curvature libs, nothing else
        self.fields = fields
        self.simulationdata = simulationdata
        # NOTE gradient and curvature share the field increments, so naming must be consistent. Can use the gradient_descs['inc_fn'] for this
        self.filter = filter.base(filter_desc)
        
        # TODO does not work if ONLY lensing or birefringence is used
        self.gradients = []
        self.gradients.append(gradient.lensing(gradient_descs['lensing'], self.filter, simidx))
        self.gradients.append(gradient.birefringence(gradient_descs['birefringence'], self.filter, simidx))
        
        # TODO i order them here as h0arr needs a specific ordering, but this is not a good solution
        h0dict = self.get_h0(desc["Runl0"])
        h0arr = np.vstack([h0dict[field] for field in ['lensing', 'birefringence']])
        curvature_desc.update({"h0": h0arr})
        self.curvature = curvature.base(curvature_desc, self.gradients)

        self.itmax = desc['itmax']
        self.simidx = simidx
        self.template_cacher = cachers.cacher_npy(template_desc['libdir'])
        self.template_operators = template_desc['template_operators']
        

    # get_klm is used for certain iteration, current iteration, and final iteration 
    # can also ask for certain field and component. gradient search is always done with
    # the joint operators
    def get_klm(self, simidx, request_it, field_ID=None, component=None):
        current_it = self.maxiterdone()
        if self.maxiterdone() < 0:
            assert 0, "Could not find the QE starting points, I expected them e.g. at {}".format(self.fields['lensing'].libdir)
        if request_it <= current_it: # data already calculated
            if field_ID is None:
                return [self.get_klm(simidx, request_it, fieldID, component) for fieldID, field in self.fields.items()]
            return np.array(self.fields[field_ID].get_klm(simidx, request_it, component))
        elif current_it < self.itmax and request_it > current_it:
            for it in range(current_it, request_it):
                grad_tot, grad_prev = [], []
                print('starting iteration ', it+1, 'taking result from iteration', it, '. maxiterdone:', self.maxiterdone())
                for gradient in self.gradients:
                    gradient.update_operator(simidx, it)
                    grad_tot.append(gradient.get_gradient_total(it)) #calculates the filtering, the sum, and the quadratic combination
                grad_tot = np.vstack(grad_tot)
                
                if it-1>=0: #NOTE it=0 cannot build the previous diff, as current diff is QE
                    for gradient in self.gradients:
                        grad_prev.append(gradient.get_gradient_total(it-1))
                    grad_prev = np.vstack(grad_prev)
                    self.curvature.add_yvector(grad_tot, grad_prev, simidx, it)
                
                N = 0 # NOTE it=0 uses h0 for the curvature
                new_klms = self.curvature.grad2dict(self.curvature.get_new_gradient(grad_tot, simidx, it))
                # new_klms = self.step(new_klms)
                self.cache_klm(new_klms, simidx, it+1)
        else:
            assert False, "Requested iteration is beyond the maximum iteration"


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


    def isiterdone(self, it):
        if it >= 0:
            return self.fields['lensing'].is_cached(self.simidx, it, 'alpha')
        return False    


    def maxiterdone(self):
        itr = -2
        isdone = True
        while isdone:
            itr += 1
            isdone = self.isiterdone(itr + 1)
        return itr


    def get_new_MAP(self, H, gtot):
        pass
        # self.curvature.get_new_MAP(H, gtot)
        # deltag = self.curvature.get_gradient_inc(self.klm_currs) # This calls the 2-loop curvature update
        # for field in self.fields:
        #     for component in field.components:
        #         increment = field.calc_increment(deltag, component)
        #         field.update_klm(increment, component) 


    def step(self, klms):
        ret = []
        fl = np.ones_like(klms[0])
        for fieldID, field in self.fields.items():
            for component in field.components.split("_"):
                ret.append(almxfl(klms[fieldID][component], fl, None, False))
        return ret
        # steplen=1
        # return almxfl(MAP, steplen)


    def cache_klm(self, new_klms, simidx, it):
        for fieldID, field in self.fields.items():
            for component in field.components.split("_"):
                field.cache_klm(new_klms[fieldID][component], simidx, it, component=component)


    # exposed functions
    def get_WF(self, field):
        self.gradients[field].get_WF(self.get_datmaps(), field)

    
    def get_ivf(self, field):
        self.gradients[field].get_ivf(field)


    def get_h0(self, R_unl0):
        ret = {}
        idx2gradient = {grad.ID: idx for idx, grad in enumerate(self.gradients)}
        for field_id, field in self.fields.items():
            h0 = []
            for componenti, component in enumerate(field.components.split("_")):
                self.ckk_prior = field.CLfids[component][:self.gradients[idx2gradient[field_id]].lm_max_qlm[0]+1]
                buff = cli(R_unl0[field.ID][componenti][:self.gradients[idx2gradient[field_id]].lm_max_qlm[0]+1] + cli(self.ckk_prior))   #~ (1/Cpp + 1/N0)^-1
                buff *= (self.ckk_prior > 0)
                h0.append(buff)
            ret.update({'{}'.format(field.ID): np.array(h0)}) 
        return ret