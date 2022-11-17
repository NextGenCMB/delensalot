def OMP_NUM_THREADS(instance, attribute, value):
    if type(attribute) != int:
        raise ValueError('Must be int')
