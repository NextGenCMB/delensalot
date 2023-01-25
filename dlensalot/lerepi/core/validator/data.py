def simidxs(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def class_parameters(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def package_(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def module_(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def class_(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def data_type(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def data_field(instance, attribute, value):
    desc = ['p_p', 'p_eb']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def beam(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def nside(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def transferfunction(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))