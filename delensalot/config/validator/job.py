from delensalot.config.metamodel import DEFAULT_NotAValue

def jobs(instance, attribute, value):
    desc = ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'delens']
    if value != DEFAULT_NotAValue:
        assert all(val in desc for val in value), ValueError('Must be in {}, but is {}'.format(desc, value))