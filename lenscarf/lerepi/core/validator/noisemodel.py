def spectrum_type(instance, attribute, value):
    desc = ['white', 'non-white']
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def sky_coverage(instance, attribute, value):
    desc = ['isotropic', 'masked']
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def OBD(instance, attribute, value):
    desc = [True, False]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lmin_teb(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def nlev_t(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def nlev_p(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def rhits_normalised(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def mask(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def ninvjob_geometry(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))