import numpy as np

from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'p0': [],
    'p1': [],
    'p2': [],
    'p3': [],
    'p4': [],
    'p5': [],
    'p6': [],
    'p7': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'p0': [],
    'p1': [],
    'p2': [],
    'p3': [],
    'p4': [],
    'p5': [],
    'p6': [],
    'p7': [],
}

# if [], doesn't check for type
valid_type = {
    'p0': [],
    'p1': [],
    'p2': [],
    'p3': [],
    'p4': [],
    'p5': [],
    'p6': [],
    'p7': [],
}

def p0(instance, attribute, value):
    if type(value) != int:
        raise ValueError('Must be int')
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p1(instance, attribute, value):
    valid_bound[attribute.name] = [['diag_cl'], DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p2(instance, attribute, value):
    valid_bound[attribute.name] = [None, DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p3(instance, attribute, value):
    valid_bound[attribute.name] = [2048, DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p4(instance, attribute, value):
    valid_bound[attribute.name] = [np.inf, DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p5(instance, attribute, value):
    valid_bound[attribute.name] = [None, DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p6(instance, attribute, value):
    valid_bound[attribute.name] = ['tr_cg', DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def p7(instance, attribute, value):
    valid_bound[attribute.name] = ['cache_mem', DEFAULT_NotAValue]
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
