def libdir(instance, attribute, value):
    if type(value) != str:
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
