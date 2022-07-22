import os
import numpy as np

from lenscarf.lerepi.core.metamodel.dlensalot import *


def lowell_treat(instance, attribute, value):
    desc = ['OBD', 'trunc', None]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def OBD(instance, attribute, value):
    desc = [type(value)]
    assert type(value) in desc, ValueError('Must be in {}, but is {}'.format(desc, type(value)))


def lmin_tlm(instance, attribute, value):
    desc_min = 0
    desc_max = np.inf
    assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


def lmin_elm(instance, attribute, value):
    desc_min = 0
    desc_max = np.inf
    assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


def lmin_blm(instance, attribute, value):
    desc_min = 0
    desc_max = np.inf
    assert desc_min<value<desc_max, ValueError('Must be in [{}-{}], but is {}'.format(desc_min, desc_max, value))


def nlev_t(instance, attribute, value):
    desc_dtype = [int, np.float, np.float64]
    desc = [0.,np.inf]
    assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
    assert desc[0]<value<desc[1], ValueError('Must be in {}, but is {}'.format(desc, value))


def nlev_p(instance, attribute, value):
    desc_dtype = [int, np.float, np.float64]
    desc = [0.,np.inf]
    assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
    assert desc[0]<value<desc[1], ValueError('Must be in {}, but is {}'.format(desc, value))


def rhits_normalised(instance, attribute, value):
    desc_dtype = [str, type(None)]
    assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
    if type(value) is not type(None):
        assert os.path.isfile(value), OSError("File doesn't exist: {}".format(value))


def mask(instance, attribute, value):
    desc_dtype = [str, type(None)]
    assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
    if type(value) is not type(None):
        assert os.path.isfile(value), OSError("File doesn't exist: {}".format(value))


def ninvjob_geometry(instance, attribute, value):
    desc_dtype = [str, type(None)]
    desc = ['healpix_geometry', 'healpix_geometry_qe', 'thin_gauss', 'pbdGeometry']
    assert type(value) in desc_dtype, TypeError('Must be in {}, but is {}'.format(desc_dtype, type(value)))
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
