from . import field
from . import gradient
from . import curvature

class base:
    def __init__(self, fields_desc, gradient_descs, curvature_desc, **kwargs):
        self.fields = field(fields_desc)
        self.gradients = [gradient(gradient_desc, field) for gradient_desc, field in zip(gradient_descs, self.fields)]
        self.curvature = curvature(curvature_desc, self.gradients)


    def run(self):
        self.get_current_MAP()
        self.get_gradient()
        self.get_curvature()
        self.get_step()
        self.update_MAP()


    def get_current_MAP(self):
        pass


    def get_gradient(self):
        for gradient in self.gradients:
            gradient.get_gradient()


    def get_curvature(self):
        pass


    def get_step(self):
        pass


    def update_MAP(self):
        pass

