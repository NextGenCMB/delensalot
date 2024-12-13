import numpy as np

from . import operator


class lensing:

    def __init__(self, operator_kwargs, **lensing_desc):
        self.lensing_desc = lensing_desc
        self.op1 = operator.spin_raise(operator_kwargs)
        self.op2 = operator.lensing(operator_kwargs)
        self.operators = [self.op1, self.op2]

    def inner_derivative_lensing(self, obj):
        buff = None
        for operator in self.operators:
            buff = operator.act(obj)
        return buff
    

    def filter_operator(self, obj):
        return self.op2.act(obj)
    

    def update_field(self, fields):
        for oi, operator in enumerate(self.operators):
            operator.update_field(fields[oi])



class lensingplusbirefringence:


    def __init__(self, operator_kwargs, **lensing_desc):
        self.lensing_desc = lensing_desc
        self.op1 = operator.spin_raise(operator_kwargs)
        self.op2 = operator.lensing(operator_kwargs)
        self.op3 = operator.birefringence(operator_kwargs)
        self.operators = [self.op1, self.op2, self.op2]


    def update_field(self, fields):
        for oi, operator in enumerate(self.operators):
            operator.update_field(fields[oi])


    def inner_derivative_lensing(self, obj):
        buff = None
        for operator in self.operators:
            buff = operator.act(obj)
            obj = buff
        return obj 
    
    
    def inner_derivative_birefringence(self, obj):
        buff = None
        for operator in self.operators[1:]:
            buff = operator.act(obj)
            obj = buff
        return -np.imag * obj
    
    
    def filter_operator(self, obj):
        buff = None
        for operator in self.operators[1:]:
            buff = operator.act(obj)
            obj = buff
        return obj
    

    def filter_operator_adjoint(self):
        buff = None
        for operator in self.operators[1:][::-1]:
            buff = operator.act(obj, adjoint=True)
            obj = buff
        return obj
    

    def update_field(self, fields):
        for oi, operator in enumerate(self.operators):
            operator.update_field(fields[oi])


class birefringence:
    
    def inner_derivative_lensing(kwargs, obj):
        buff = operator.birefringence(kwargs).act(obj)
        return buff 
    

    def inner_derivative_birefringence(kwargs, obj):
        buff = operator.birefringence(kwargs).act(obj)
        return -np.imag * buff 
    

    def filter_operator(kwargs, obj):
        buff = operator.birefringence(kwargs).act(obj)
        return buff
    

    def update_field(self, fields):
        for oi, operator in enumerate(self.operators):
            operator.update_field(fields[oi])