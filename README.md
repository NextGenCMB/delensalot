![delensalot logo](res/dlensalot2.PNG)
[![Install](https://github.com/NextGenCMB/delensalot/actions/workflows/install.yaml/badge.svg?branch=sims&event=pull_request)](https://github.com/NextGenCMB/delensalot/actions/workflows/install.yaml)
[![Documentation Status](https://readthedocs.org/projects/delensalot/badge/?version=latest)](https://delensalot.readthedocs.io/en/latest/?badge=latest)
[![Integration filter](https://github.com/NextGenCMB/delensalot/actions/workflows/integration_filter.yaml/badge.svg?branch=sims&event=pull_request)](https://github.com/NextGenCMB/delensalot/actions/workflows/integration_filter.yaml)
[![Integration reconstruction](https://github.com/NextGenCMB/delensalot/actions/workflows/unit_reconstruction.yaml/badge.svg?branch=sims&event=pull_request)](https://github.com/NextGenCMB/delensalot/actions/workflows/integration_reconstruction.yaml)
[![Integration tutorial](https://github.com/NextGenCMB/delensalot/actions/workflows/integration_tutorial.yaml/badge.svg?branch=sims&event=pull_request)](https://github.com/NextGenCMB/delensalot/actions/workflows/integration_tutorial.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2302.01942-red)](https://arxiv.org/abs/2310.06729)

# delensalot
Curved-sky iterative CMB lensing reconstruction and bias calculation.

If you use delensalot for your work, please consider citing [CMB-S4: Iterative internal delensing and r constraints](https://arxiv.org/abs/2310.06729).

# Installation
Download the project, navigate to the root folder and execute the command,

``` 
python setup.py install
```

Make sure that you have the latest `plancklens` and that you are using the `plancklensdev` branch
```
cd plancklens
git checkout plancklensdev
```
This is needed to give delensalot control over plancklens's mpi implementation, which we conveniently set up at this branch.

You will need to install `jupyter` for the tutorials found in `first_steps/notebooks/`, and possibly an `ipykernel` to create a jupyter-kernel out of the environment in which you install `delensalot`.

### Set up a jupyter-kernel with delensalot

To run the tutorials with a jupyter kernel, you will have to install delensalot in it. Assuming you are using conda for your package management,

```
conda create --name delensalot
conda activate delensalot
```

Then, install your favourite packages,

```
conda install pip, numpy
```

Now, go to the delensalot directory, and install, including its requirements,

```
cd </path/to/delensalot>
python3 -m pip install -r requirements .
python3 setup.py develop
```

Eventually, create your kernel using the environment at which you just installed all packages,
```
python3 -m ipykernel install --user --name=delensalot
```

## Installation troubles

Frequent problems are
 * `attrs`. If there are errors related to the metamodel, make sure you have `attrs` (not `attr`, which is a different package) installed and updated/upgraded (version 23.1.0 should do)
 * `astropy`. If there are errors related to ducc0, chances are it's because of an old `astropy` installation. Make sure you have the latest `astropy`.

# Usage

## The quickest way: `anafast()`, `map2delblm()` or `map2tempblm()`

`delensalot` comes with two handy functions to get you started very easily.
To get a delensed power spectrum, simply import `delensalot` and run `anafast()`:
```
import delensalot
delensedmap = delensalot.anafast(obsmaps, lmax_cmb=lmax_cmb, beam=beam, itmax=itmax, noise=noise, verbose=True)
```

To get a delensed B map, simply import `delensalot` and run `map2delblm()`:
```
import delensalot
delensedmap = delensalot.map2delblm(obsmaps, lmax_cmb=lmax_cmb, beam=beam, itmax=itmax, noise=noise, verbose=True)
```

here `obsmaps` is the observed Q and U map you may have... gotten from somewhere. Then all what is left to do is to tell `delensalot` about,
 * the maximum \ell (`lmax_cmb`) of your CMB map you'd like to use,
 * the beam (`sims_beam`) of the transfer function of the observed maps,
 * how many iterations (`itmax`) you'd like to perform,
 * and the noise level (`noise`) of the observation.

If you are interested in the B-lensing template, instead use `map2tempblm()`,
```
import delensalot
Blenstemplate = delensalot.map2tempblm(obsmaps, lmax_cmb=lmax_cmb, beam=beam, itmax=itmax, noise=noise, verbose=True)
```

## Run a configuration file

To run a configuration file `<path-to-config/conf.py>`, type in your favorite `bash`,
``` 
python3 run.py -r <path-to-config/conf.py>
```

delensalot supports MPI,

```
srun --nodes <nnodes> -n <taskspernode> python3 run.py -r <path-to-config/conf.py>
```

If you'd like to know the status of the analysis done with `<path-to-config/conf.py>`, run,
```
python3 run.py -s <path-to-config/conf.py>
```

## interactive mode

delensalot supports interactive mode. See `first_steps/notebooks/` for our tutorials.


## help

Type `python3 run.py [-h]` for quickhelp,
```
usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

delensalot entry point.

optional arguments:
  -h, --help            show this help message and exit
  -r RESUME             Absolute path to config file to run/resume.
  -s STATUS             Absolute path for the analysis to write a report.

```


# Dependencies

 uses
  * [Plancklens](https://github.com/carronj/plancklens)
  * [lenspyx](https://github.com/carronj/lenspyx)
  * [DUCC](https://github.com/mreineck/ducc)

## Doc
Documentation may be found [HERE]


## Use with HPC
`delensalot` can be computationally demanding.
We have parallelized the computations across the simulation index in most cases. Assuming you have MPI set up and `srun` is available, you can simply run MPI-supported `delensalot` via,

```
srun -MPI_paramX X -MPI_paramY Y python3 <path-to-delensalot>/run.py -r <path-to-config-file>
```

If you have troubles, your HPC-center can help.
