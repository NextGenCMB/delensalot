import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='lenscarf',
    version='0.0.1',
    packages=['lenscarf'],
    data_files=[('lenscarf/data/cls', ['lenscarf/data/cls/FFP10_wdipole_lensedCls.dat',
                                'lenscarf/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                'lenscarf/data/cls/FFP10_wdipole_params.ini'])],
    url='https://github.com/carronj/lenscarf',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='Iterative CMB lensing curved-sky package based on scarf SHTs',
    install_requires=['numpy'], #removed mpi4py for travis tests
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

