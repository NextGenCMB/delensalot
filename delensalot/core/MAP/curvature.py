from delensalot.core.iterator import bfgs

class base:
    def __init__(self, bfgs_desc, curvature_desc):
        pass
        # self.BFGS_H = bfgs.BFGS_Hessian(bfgs_desc)

    
    def get_current_curvature(self):
        pass


    def update_curvature(self, gtot):
        pass
        # deltag = self.curvature.get_gradient_inc(self.klm_currs) # This calls the 2-loop curvature update
        # for field in self.fields:
        #     for component in field.components:
        #         increment = field.calc_increment(deltag, component)
        #         field.update_klm(increment, component) 

    
    def get_new_MAP(self):
        pass
        # self._new_MAP(H, gtot)