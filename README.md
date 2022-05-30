![D.lensalot logo](res/dlensalot2.PNG)
# D.Lensalot 


(formerly known as Lenscarf)
Curved-sky iterative CMB lensing tools

## Installation
D.lensalot is computationally demanding and therefore needs NERSC.

### Use on NERSC

First load the dependent libraries, and swap to the gnu gcc compiler.
To do so, open a terminal on NERSC and execute,
```
module load fftw
module load gsl
module load cfitsio
module swap PrgEnv-intel PrgEnv-gnu
module load python
```

alternative, add the above lines to your `~/.bash_profile`


Download the project to your computer, navigate to the root folder and execute the command,

``` 
python setup.py install
```

For this to work, an older gnu compiler, `gcc 7` is currently needed, as a newer version is more restrictive to type checking.


## Dependencies

 based on
  * [Scarf](https://github.com/samuelsimko/scarf)
  * [Plancklens](https://github.com/carronj/plancklens)


## Doc

Documentation may be found [HERE]
