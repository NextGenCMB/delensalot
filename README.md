![D.lensalot logo](res/dlensalot2.PNG)
# D.Lensalot 



(formerly known as Lenscarf)
Curved-sky iterative CMB lensing tools

## Installation

Download the project to your computer, navigate to the root folder and execute the command,

``` 
python setup.py install
```

## Dependencies


 based on
  * [Scarf](https://github.com/samuelsimko/scarf)
  * [Plancklens](https://github.com/carronj/plancklens)


## Doc

Documentation may be found [HERE]

# Usage

simply type in your favourite `bash`,

```
python3 run.py -p <path-to-config>
```
where, `<path-to-config>` is a relative path, pointing to a config file in config/, e.g.

```
python3 run.py -p examples/example_c08b.py
```

runs the example configuration for `c08b`. See [lenscarf/lerepi/README](https://github.com/NextGenCMB/lenscarf/blob/f/mergelerepi/lenscarf/lerepi/README.rst) for a description of the configuation parameters


## Use on NERSC


To use D.lensalot on NERSC, you need to load some libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
Type these lines in the terminal or include them into your `~/.bash_profile`:

```
module load fftw
module load gsl
module load cfitsio
module swap PrgEnv-intel PrgEnv-gnu
module load python
```


