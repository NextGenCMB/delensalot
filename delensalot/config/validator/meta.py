import numpy as np
from delensalot.config.metamodel import DEFAULT_NotAValue

# if [], doesn't check for value
valid_value = {
    'version': [],
}
# if [], doesn't check for bounds
valid_bound = {
    'version': [],
}

# if [], doesn't check for type
valid_type = {
    'version': [],
}

def version(instance, attribute, value):
    if type(attribute) != str:
        raise ValueError('Must be str')
