def libdir(instance, attribute, value):
    if type(attribute) != int:
        raise ValueError('Must be int')


def rescale(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def tpl(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def nlev_dep(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
