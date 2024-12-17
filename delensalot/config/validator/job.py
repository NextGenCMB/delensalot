import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'jobs': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'jobs': [],
}

# if [], doesn't check for type
valid_type = {
    'jobs': [],
}

def jobs(instance, attribute, value):
    valid_bound[attribute.name] = ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'MAP_lensrec_operator', 'delens']
    if np.all(value != DEFAULT_NotAValue):
        assert all(val in valid_bound[attribute.name] for val in value), ValueError('Must be in {}, but is {}'.format(valid_bound[attribute.name], value))