import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'typ': [],
    'lmax_qlm': [],
    'mmax_qlm': [],
    'a': [],
    'b': [],
    'xa': [],
    'xb': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'typ': [],
    'lmax_qlm': [],
    'mmax_qlm': [],
    'a': [],
    'b': [],
    'xa': [],
    'xb': [],
}

# if [], doesn't check for type
valid_type = {
    'typ': [],
    'lmax_qlm': [],
    'mmax_qlm': [],
    'a': [],
    'b': [],
    'xa': [],
    'xb': [],
}

def typ(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lmax_qlm(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def mmax_qlm(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def a(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def b(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def xa(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def xb(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
