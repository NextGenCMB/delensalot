import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    modules = glob.glob('lenscarf/fortran/*.f95')
    for modu in modules:
        nam = modu.split('/')[-1].replace('.f95', '')
        config.add_extension('lenscarf.fortran.' + nam, ['lenscarf/fortran/%s.f95'%nam],
                extra_link_args=['-lgomp'],libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w' , '-O3', '-ffast-math'])
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
    install_requires=['numpy', 'pyfftw'], #removed mpi4py for travis tests
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

