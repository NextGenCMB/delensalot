def meta(instance, attribute, value):
    desc = [dlmeta]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def computing(instance, attribute, value):
    desc = [DLENSALOT_Computing]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def job(instance, attribute, value):
    desc = [DLENSALOT_Job]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def analysis(instance, attribute, value):
    desc = [DLENSALOT_Analysis]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def data(instance, attribute, value):
    desc = [DLENSALOT_Data]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def noisemodel(instance, attribute, value):
    desc = [DLENSALOT_Noisemodel]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def qerec(instance, attribute, value):
    desc = [DLENSALOT_Qerec]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def itrec(instance, attribute, value):
    desc = [DLENSALOT_Itrec]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))


def madel(instance, attribute, value):
    desc = [DLENSALOT_Mapdelensing]
    assert attribute in desc, ValueError('Must be in {}, but is {}'.format(desc, value))
