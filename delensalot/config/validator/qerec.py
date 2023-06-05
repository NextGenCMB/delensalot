import numpy as np

def tasks(instance, attribute, value):
    desc = ["calc_phi", "calc_meanfield", "calc_blt"]
    assert all(val in desc for val in value), ValueError('Must be in {}, but is {}'.format(desc, value))


def simidxs(instance, attribute, value):
    desc = [int]
    assert type(value) in desc, TypeError('Must be in {}, but is {}'.format(desc, value))


def simidxs_mf(instance, attribute, value):
    desc = [list, np.ndarray]
    assert type(value) in desc, TypeError('Must be in {}, but is {}'.format(desc, type(value)))


def btemplate_perturbative_lensremap(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def ivfs(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def qlms(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def filter_directional(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def cg_tol(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

    
def ninvjob_qe_geometry(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def lm_max_qlm(instance, attribute, value):
    desc = [int, tuple]
    assert type(value) in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def chain(instance, attribute, value):
    desc = [value]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def cl_analysis(instance, attribute, value):
    desc = [True, False]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
