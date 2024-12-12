from . import filter

class base:
    def __init__(self, fields, gradient_desc, filter_desc, **kwargs):
        self.fields = fields
        self.filter = filter(filter_desc)


    def get_current_gradient(self):
        pass


    def get_gradient_quad(self, field):
        pass


    def get_gradient_meanfield(self):
        pass


    def get_gradient_prior(self):
        pass


    def update_gradient(self):
        pass


