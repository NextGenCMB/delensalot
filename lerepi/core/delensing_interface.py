"""Interface delensing algorithms
"""
from lenscarf.core import handler


class Dlensalot():
    def __init__(self, sc, lc, rc):
        self.params = self.get_dlensalot_params(sc, lc, rc)
        self.dlensalot = handler.MAP_delensing(sc, lc, rc)


    
    def get_dlensalot_params(self, sc, lc, rc):
        params = dict()
        """Here we extract all parameters which are needed for dlensalot

        Args:
            sc (_type_): _description_
            lc (_type_): _description_
            rc (_type_): _description_
        """
        return params


    def run(self):
        self.dlensalot.collect_jobs()
        self.dlensalot.run()
