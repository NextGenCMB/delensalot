from delensalot.core.cg import cd_solve, cd_monitors, multigrid
from . import operator

class base:
    def __init__(self, operator_descs, **filter_desc):
        self.operators = [operator(operator_desc) for operator_desc in operator_descs]
        pass


    def get_XWF(self, field):
        # self.filter.
        pass


    def get_ivf(self, field):
        # self.filter.
        pass