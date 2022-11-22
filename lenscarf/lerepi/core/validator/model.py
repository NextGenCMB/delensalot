def meta(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def job(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def analysis(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def data(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def noisemodel(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def qerec(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def itrec(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def madel(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
