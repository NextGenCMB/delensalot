import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
}
# if [], doesn't check for bounds
valid_bound = {
}

# if [], doesn't check for type
valid_type = {
}

def meta(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def job(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def analysis(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def data(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def noisemodel(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def qerec(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def itrec(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def computing(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def config(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def madel(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
    #     if valid_bound[attribute.name] != []:
    #         if len(valid_bound[attribute.name]) == 1:
    #             assert np.all(value >= valid_bound[attribute.name][0]), ValueError('Must be leq {}, but is {}'.format(valid_bound[attribute.name][0], value))
    #         if len(valid_bound[attribute.name]) == 2:
    #             assert np.all(value <= valid_bound[attribute.name][1]), ValueError('Must be seq {}, but is {}'.format(valid_bound[attribute.name][1], value))

def obd(instance, attribute, value):
    pass
    # if np.all(value != DEFAULT_NotAValue):
    #     assert value in valid_value[attribute.name] if valid_value[attribute.name] != [] else 1, ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))
