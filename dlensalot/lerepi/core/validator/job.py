def QE_lensrec(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def MAP_lensrec(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def inspect_result(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def map_delensing(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def build_OBD(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
