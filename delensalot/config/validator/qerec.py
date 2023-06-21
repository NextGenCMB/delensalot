import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'tasks': ["calc_phi", "calc_meanfield", "calc_blt"],
    'qlm_type': [],
    'cg_tol': [],
    'filter_directional': [],
    'lm_max_qlm': [],
    'chain': [],
    'cl_analysis': [],
    'blt_pert': [],
}

# if [], doesn't check for bounds
valid_bound = {
    'tasks': [],
    'qlm_type': [],
    'cg_tol': [],
    'filter_directional': [],
    'lm_max_qlm': [],
    'chain': [],
    'cl_analysis': [],
    'blt_pert': [],
}

# if [], doesn't check for type
valid_type = {
    'tasks': [],
    'qlm_type': [],
    'cg_tol': [],
    'filter_directional': [],
    'lm_max_qlm': [],
    'chain': [],
    'cl_analysis': [],
    'blt_pert': [],
}

def tasks(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def simidxs(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert type(value) in valid_bound[attribute.name], TypeError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def simidxs_mf(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert type(value) in valid_bound[attribute.name], TypeError('Must be in {}, but is {}'.format(valid_bound[attribute.name], type(value)))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def btemplate_perturbative_lensremap(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def ivfs(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def qlms(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def filter_directional(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def cg_tol(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))

    
def ninvjob_qe_geominfo(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def lm_max_qlm(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
        if valid_bound[attribute.name] != []:
            if len(valid_bound[attribute.name]) == 1:
                assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
            if len(valid_bound[attribute.name]) == 2:
                assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def chain(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def cl_analysis(instance, attribute, value):
    if np.all(value != DEFAULT_NotAValue):
        assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
