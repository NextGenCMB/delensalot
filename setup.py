import setuptools
from setuptools import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='delensalot',
    version='0.1',
    packages=setuptools.find_packages(),
    data_files=[('delensalot/data/cls', [
        'delensalot/data/cls/FFP10_wdipole_lensedCls.dat',
        'delensalot/data/cls/FFP10_wdipole_lenspotentialCls.dat',
        'delensalot/data/cls/FFP10_wdipole_params.ini'
    ])],
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Example of an extension in case you need it, change or remove accordingly:
    # ext_modules=[
    #     Extension('delensalot.some_extension', ['path/to/source_file.c'], extra_compile_args=['-O3'])
    # ],
)
