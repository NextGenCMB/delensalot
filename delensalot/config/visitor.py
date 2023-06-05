#!/usr/bin/env python

"""visitor.py: This module implements a double dispatch in order to
mimic some kind of visitor pattern.

Based on:
https://stackoverflow.com/a/56098472
"""



class _Visitor:
    def __init__(self, t_fct):
        """"Initialise with a transformation function object."""
        self.transform = t_fct
        self.cases = {}

    def case(self, type1, type2):
        """Do double dispatch."""
        def call(fun):
            self.cases[(type1, type2)] = fun
        return call

    def __call__(self, arg1, arg2):
        fun = self.cases[type(arg1), type(arg2)]
        return fun(arg1, arg2)
    
class _Visitor3d:
    def __init__(self, t_fct):
        """"Initialise with a transformation function object."""
        self.transform = t_fct
        self.cases = {}
        
    def case(self, type1, type2, type3):
        """Do triple dispatch."""
        def call(fun):
            self.cases[(type1, type2, type3)] = fun
        return call

    def __call__(self, arg1, arg2, arg3):
        fun = self.cases[type(arg1), type(arg2), type(arg3)]
        return fun(arg1, arg2, arg3)


@_Visitor
def transform(x, y): # pylint: disable=unused-argument,missing-function-docstring
    pass

@_Visitor3d
def transform3d(x, y, z): # pylint: disable=unused-argument,missing-function-docstring
    pass
