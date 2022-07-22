def QE_lensrec(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def MAP_lensrec(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def inspect_result(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def map_delensing(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def build_OBD(instance, attribute, value):
    desc = [attribute]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
