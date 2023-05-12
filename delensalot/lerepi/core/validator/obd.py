from delensalot.lerepi.core.metamodel import DEFAULT_NotAValue


def libdir(instance, attribute, value):
    if type(value) != str and value != DEFAULT_NotAValue:
        raise ValueError('Must be str')

def rescale(instance, attribute, value):
    if type(value) not in [float, int]:
        raise ValueError('Must be float or int')

def tpl(instance, attribute, value):
    desc = ['template_dense']
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

def nlev_dep(instance, attribute, value):
    if type(value) not in [float, int]:
        raise ValueError('Must be float or int')

def nside(instance, attribute, value):
    if type(value) not in [float, int]:
        raise ValueError('Must be float or int')

def lmax(instance, attribute, value):
    if type(value) not in [float, int]:
        raise ValueError('Must be float or int')

def beam(instance, attribute, value):
    if type(value) not in [float, int]:
        raise ValueError('Must be float or int')
