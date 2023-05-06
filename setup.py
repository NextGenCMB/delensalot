import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='delensalot',
    version='0.0.1',
    packages=['delensalot'],
    data_files=[('delensalot/data/cls', ['delensalot/data/cls/FFP10_wdipole_lensedCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_params.ini'])],
    url='https://github.com/NextGenCMB/delensalot',
    author='Julien Carron, Sebastian Belkner',
    author_email='to.jcarron@gmail.com, to.sebastianbelkner@gmail.com',
    description='Iterative CMB lensing reconstruction on curved-sky',
    install_requires=['numpy', 'pyfftw', 'healpy', 'logdecorator', 'psutil', 'ducc0', 'lenspyx'], #removed mpi4py for travis tests
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

