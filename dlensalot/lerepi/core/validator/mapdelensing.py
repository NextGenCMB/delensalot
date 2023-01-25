def edges(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def dlm_mod(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def iterations(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def masks(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lmax(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def Cl_fid(instance, attribute, value):
    desc = ['sepTP', 'simple']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

    
def libdir_it(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def binning(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def spectrum_calculator(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
