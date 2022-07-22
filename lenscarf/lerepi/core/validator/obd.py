def libdir(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def rescale(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def tpl(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def nlev_dep(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
