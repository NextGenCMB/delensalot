import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'libdir': [],
    'rescale': [],
    'tpl': [],
    'nlev_dep': [],
    'lmax': [],
    'beam': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'libdir': [],
    'rescale': [],
    'tpl': [],
    'nlev_dep': [],
    'lmax': [],
    'beam': [],
}

# if [], doesn't check for type
valid_type = {
    'libdir': [],
    'rescale': [],
    'tpl': [],
    'nlev_dep': [],
    'lmax': [],
    'beam': [],
}

def libdir(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def rescale(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def tpl(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def nlev_dep(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def nside(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lmax(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def beam(instance, attribute, value):
    if valid_bound[attribute.name] != []:
        if len(valid_bound[attribute.name]) == 1:
            assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
        if len(valid_bound[attribute.name]) == 2:
            assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))
