def simidxs(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def simidxs_mf(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def ivfs(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def qlms(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def cg_tol(instance, attribute, value):
    desc = ['sepTP', 'simple']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

    
def ninvjob_qe_geometry(instance, attribute, value):
    desc = ['nmr_relative', 'mr_relative']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lmax_qlm(instance, attribute, value):
    desc = ['max', 'extend']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def mmax_qlm(instance, attribute, value):
    desc = ['map', 'alm']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def chain(instance, attribute, value):
    desc = ['eb', 'qu']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def cl_analysis(instance, attribute, value):
    desc = ['gauss', 'gauss_with_pixwin']
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
