"""Interface delensing algorithms
"""
from lenscarf.core import handler


class Dlensalot():
    def __init__(self, sc, lc, rc):
        self.dlensalot = handler.MAP_delensing(sc, lc, rc)


    def run(self):
        self.dlensalot.collect_jobs()
        self.dlensalot.run()
