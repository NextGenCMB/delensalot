def key(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def version(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def simidxs_mf(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def TEMP_suffix(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lens_res(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def Lmin(instance, attribute, value):
    desc = [int]
    assert type(attribute) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def zbounds(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def zbounds_len(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def pbounds(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
