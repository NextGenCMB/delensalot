from delensalot.core.iterator import bfgs

class base:
    def __init__(self, bfgs_desc, **curvature_desc):
        self.BFGS_H = bfgs(bfgs_desc)
        self.value = curvature_desc['value']

    
    def get_current_curvature(self):
        pass


    def update_curvature(self):
        pass