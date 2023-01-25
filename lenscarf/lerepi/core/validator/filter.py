def lmax_filt(instance, attribute, value):
    if type(attribute) != int:
        raise ValueError('Must be int')


def lm_max_len(instance, attribute, value):
    desc = [tuple, int, list]
    assert type(attribute) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lm_max_unl(instance, attribute, value):
    desc = [tuple, int, list]
    assert type(attribute) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lm_ivf(instance, attribute, value):
    desc = [tuple, int, list]
    assert type(attribute) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))