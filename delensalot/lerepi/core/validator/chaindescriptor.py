import numpy as np

from delensalot.lerepi.core.metamodel import DEFAULT_NotAValue

def p0(instance, attribute, value):
    if type(value) != int:
        raise ValueError('Must be int')


def p1(instance, attribute, value):
    desc = [['diag_cl'], DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p2(instance, attribute, value):
    desc = [None, DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p3(instance, attribute, value):
    desc = [2048, DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p4(instance, attribute, value):
    desc = [np.inf, DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p5(instance, attribute, value):
    desc = [None, DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p6(instance, attribute, value):
    desc = ['tr_cg', DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def p7(instance, attribute, value):
    desc = ['cache_mem', DEFAULT_NotAValue]
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
