import numpy as np

def simidxs(instance, attribute, value):
    desc = [list, int, np.ndarray]
    assert type(value) in desc, TypeError('Must be in {}, but is {}'.format(desc, type(value)))

def class_parameters(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def package_(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def module_(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def class_(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def data_type(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def maps(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def phi(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def data_field(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def beam(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def nside(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def transferfunction(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def epsilon(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def lmax_transf(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def nlev_t(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def nlev_p(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))