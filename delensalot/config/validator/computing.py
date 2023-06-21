import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'OMP_NUM_THREADS': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'OMP_NUM_THREADS': [],
}

# if [], doesn't check for type
valid_type = {
    'OMP_NUM_THREADS': [],
}

def OMP_NUM_THREADS(instance, attribute, value):
    if type(value) != int:
        raise ValueError('Must be int, is {}'.format(type(value)))
