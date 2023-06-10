import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'lmax_filt': [],
    'lm_max_len': [],
    'lm_max_unl': [],
    'lm_max_ivf': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'lmax_filt': [],
    'lm_max_len': [],
    'lm_max_unl': [],
    'lm_max_ivf': [],
}

# if [], doesn't check for type
valid_type = {
    'lmax_filt': [],
    'lm_max_len': [],
    'lm_max_unl': [],
    'lm_max_ivf': [],
}

def lmax_filt(instance, attribute, value):
    if type(value) != int:
        raise ValueError('Must be int')
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lm_max_len(instance, attribute, value):
    valid_bound[attribute.name] = [tuple, int, list]
    if np.all(value != DEFAULT_NotAValue):
        assert type(value) in valid_bound[attribute.name], ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], type(value)))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lm_max_unl(instance, attribute, value):
    valid_bound[attribute.name] = [tuple, int, list]
    if np.all(value != DEFAULT_NotAValue):
        assert type(value) in valid_bound[attribute.name], ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], type(value)))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lm_max_ivf(instance, attribute, value):
    valid_bound[attribute.name] = [tuple, int, list]
    if np.all(value != DEFAULT_NotAValue):
        assert type(value) in valid_bound[attribute.name], ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], type(value)))