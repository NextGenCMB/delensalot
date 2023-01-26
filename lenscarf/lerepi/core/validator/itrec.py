def tasks(instance, attribute, value):
    desc = ["calc_phi", "calc_meanfield", "calc_blt"]
    assert all(val in desc for val in value), ValueError('Must be in {}, but is {}'.format(desc, value))


def simidxs(instance, attribute, value):
    desc = [list]
    assert type(value) in desc, TypeError('Must be in {}, but is {}'.format(desc, type(value)))


def itmax(instance, attribute, value):
    desc = [int]
    assert type(value) in desc, TypeError('Must be in {}, but is {}'.format(desc, type(value)))


def lensres(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lm_max_qlm(instance, attribute, value):
    desc = [int, tuple]
    assert type(value) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
    

def filter(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def cg_tol(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def filter_directional(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lenjob_geometry(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

    
def lenjob_pbgeometry(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def iterator_typ(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def mfvar(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def soltn_cond(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def stepper(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
