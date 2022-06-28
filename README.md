![D.lensalot logo](res/dlensalot2.PNG)
# D.Lensalot 
(formerly known as Lenscarf)
Curved-sky iterative CMB lensing tools

## Installation
Download the project to your computer, navigate to the root folder and execute the command,

``` 
python setup.py install
```
For this to work, an older gnu compiler, `gcc 7` is currently needed, as a newer version is more restrictive to type checking.


# Usage

Type `run.py [-h]` for quickhelp,
```
usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS]

D.lensalot entry point.

optional arguments:
  -h, --help  show this help message and exit
  -p NEW      Relative path to config file to run analysis.
  -r RESUME   Absolute path to config file to resume.
  -s STATUS   Absolute path for the analysis to write a report.
```


To run a configutation file `<path-to-config>`, type in your favourite `bash`,
```
python3 run.py -p <path-to-config>
```
`<path-to-config>` is a relative path, pointing to a config file in `lenscarf/lerepi/config/`.

For example,
```
python3 run.py -p examples/example_c08b.py
```
runs the example configuration for `c08b`. See [lenscarf/lerepi/README](https://github.com/NextGenCMB/lenscarf/blob/f/mergelerepi/lenscarf/lerepi/README.rst) for a description of the configuation parameters

If you already have an analysis, located at `$path`, with config file `conf.py`, you may resume this analysis with,
```
python3 run.py -r $path/conf.py
```

If you'd like to know the status of the analysis done with `$path/conf.py`, run,
```
python3 run.py -s $path/conf.py
```


## Use on NERSC
D.lensalot is computationally demanding and therefore needs NERSC.
Alternative, add the above lines to your `~/.bash_profile`

To use D.lensalot on NERSC, you need to load some libraries as well as the GNU compilers (the default ones being Intel), before installing the module.
Type these lines in the terminal or include them into your `~/.bash_profile`:

```
module load fftw
module load gsl
module load cfitsio
module swap PrgEnv-intel PrgEnv-gnu
module load python
```


# Dependencies

 based on
  * [Scarf](https://github.com/samuelsimko/scarf)
  * [Plancklens](https://github.com/carronj/plancklens)

## Doc
Documentation may be found [HERE]
