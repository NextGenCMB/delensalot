import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    for ext in ['.f90', '.f95']:
        modules = glob.glob('delensalot/fortran/*' + ext)
        for modu in modules:
            nam = modu.split('/')[-1].replace(ext, '')
            config.add_extension('delensalot.fortran.' + nam, ['delensalot/fortran/%s'%nam + ext],
                extra_link_args=['-lgomp'],libraries=['gomp'], extra_f90_compile_args=['-fopenmp', '-w' , '-O3', '-ffast-math'])
    return config

setup(
    name='delensalot',
    version='0.0.1',
    packages=['delensalot'],
    data_files=[('delensalot/data/cls', ['delensalot/data/cls/FFP10_wdipole_lensedCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_params.ini'])],
    url='https://github.com/carronj/delensalot',
    author='Julien Carron',
    author_email='to.jcarron@gmail.com',
    description='Iterative CMB lensing curved-sky package based on scarf SHTs',
    install_requires=['numpy', 'pyfftw', 'healpy', 'logdecorator', 'psutil'], #removed mpi4py for travis tests
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

