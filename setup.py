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
    version='0.1',
    packages=[
        'delensalot',
        'delensalot.data', 'delensalot.data.cls',
        'delensalot.biases',
        'delensalot.utility',
        'delensalot.config', 'delensalot.config.default', 'delensalot.config.etc', 'delensalot.config.metamodel', 'delensalot.config.transformer',
        'delensalot.core', 'delensalot.core.cg', 'delensalot.core.decorator', 'delensalot.core.helper', 'delensalot.core.iterator', 'delensalot.core.ivf', 'delensalot.core.opfilt', 'delensalot.core.power',
        'delensalot.sims',],
    data_files=[('delensalot/data/cls', ['delensalot/data/cls/FFP10_wdipole_lensedCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat',
                                'delensalot/data/cls/FFP10_wdipole_params.ini'])],
    url='https://github.com/NextGenCMB/delensalot',
    author='Julien Carron, Sebastian Belkner',
    author_email='to.jcarron@gmail.com, to.sebastianbelkner@gmail.com',
    description='Iterative CMB lensing reconstruction on curved-sky',
    install_requires=[
        'numpy',
        'healpy',
        'logdecorator',
        'psutil',
        'plancklens @ git+https://github.com/carronj/plancklens@plancklensdev',
        'lenspyx @ git+https://github.com/carronj/lenspyx',
        'attrs'
    ],
    requires=['numpy'],
    long_description=long_description,
    configuration=configuration)

