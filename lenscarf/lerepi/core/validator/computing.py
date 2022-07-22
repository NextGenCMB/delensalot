def OMP_NUM_THREADS(instance, attribute, value):
    if type(value) != int:
        raise ValueError('Must be int')
