def version(instance, attribute, value):
    if type(value) != str:
        raise ValueError('Must be int')
    desc = ['0.9']
    assert value in desc, ValueError('Must be in {}, but is {}'.format(desc, value))

