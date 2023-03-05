def jobs(instance, attribute, value):
    desc = ['generate_sim', 'build_OBD', 'QE_lensrec', 'MAP_lensrec', 'inspect_result', 'map_delensing']
    assert all(val in desc for val in value), ValueError('Must be in {}, but is {}'.format(desc, value))