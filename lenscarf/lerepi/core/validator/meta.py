def version(instance, attribute, value):
    if type(attribute) != str:
        raise ValueError('Must be int')
