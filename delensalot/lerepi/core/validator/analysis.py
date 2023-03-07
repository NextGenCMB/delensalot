from delensalot.lerepi.core.metamodel import DEFAULT_NotAValue

def key(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def version(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def simidxs_mf(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def TEMP_suffix(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def zbounds(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def zbounds_len(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))


def pbounds(instance, attribute, value):
    desc = [value]
    assert value in desc or value == DEFAULT_NotAValue, ValueError('Must be in {}, but is {}'.format(desc, value))

def Lmin(instance, attribute, value):
    desc = [int]
    assert type(value) in desc or value == DEFAULT_NotAValue, TypeError('Must be one of {}, but is {}'.format(desc, type(value)))
