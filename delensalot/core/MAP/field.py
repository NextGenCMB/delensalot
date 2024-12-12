class base:
    def __init__(self, **field_desc):
        # TODO not sure if field should have components as list, or its own class
        self.prior = field_desc['prior']
        self.id = field_desc['id']
        self.lm_max = field_desc['lm_max']
        self.f0 = field_desc['f0']
        self.components = field_desc['components']


class component:
    def __init__(self, **component_desc): 
        pass