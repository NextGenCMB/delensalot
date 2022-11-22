def lmax_filt(instance, attribute, value):
    if type(attribute) != int:
        raise ValueError('Must be int')


def lmax_len(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def mmax_len(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lmax_unl(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def mmax_unl(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
