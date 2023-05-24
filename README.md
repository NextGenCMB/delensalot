![delensalot logo](res/dlensalot2.PNG)
# delensalot
Curved-sky iterative CMB lensing tools

## Installation
Download the project to your computer, navigate to the root folder and execute the command,

``` 
python setup.py install
```

You will need to install `jupyter` for the tutorials found in `first_steps/notebooks/`, and possibly an `ipykernel` to create a jupyter-kernel out of the environment in which you install `delensalot`.
<!-- TODO: Add explicit instructions -->


# Usage

Type `python3 run.py [-h]` for quickhelp,
```
usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

delensalot entry point.

optional arguments:
  -h, --help            show this help message and exit
  -p NEW                Relative path to config file to run analysis.
  -r RESUME             Absolute path to config file to resume.
  -s STATUS             Absolute path for the analysis to write a report.

```

## The quickest way: map2map_del()

The following shows how to obtain a delensed B map from observed Q and U maps, with just one line of code and five parameters!

First we generate some mock data, which is beamed and contains noise.
```

import healpy as hp
import numpy as np

import delensalot
from delensalot.utility.utils_hp import gauss_beam

noise = np.sqrt(2)
lmax_sims = 4096
lmax_cmb = 3000
nside = 2048
sims_beam = 1.

transf = gauss_beam(sims_beam/180/60 * np.pi, lmax=lmax_sims)
elm = hp.almxfl(hp.synalm(delensalot.cls_len['ee'], lmax=lmax_sims), transf)
blm = hp.almxfl(hp.synalm(delensalot.cls_len['bb'], lmax=lmax_sims), transf)
skymaps = hp.alm2map_spin([elm, blm], nside=nside, spin=2, lmax=lmax_sims)

noise_cl = cls=np.ones(shape=lmax_sims)*180/60*np.pi*noise

vamin = np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60
obsmaps = noise/vamin * np.array([hp.alm2map(hp.synalm(noise_cl),nside=2048)+skymaps[0],hp.alm2map(hp.synalm(noise_cl),nside=2048)+skymaps[1]])
```

Finally, all we have to do is call `map2map_del()`,
```
delensedmap = delensalot.map2map_del(obsmaps, lmax_cmb=lmax_cmb, beam=sims_beam, itmax=5, noise=noise, verbose=True)
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

delensalot supports interactive mode. See `first_steps/notebooks/` for or tutorials.



# Dependencies

 uses
  * [Plancklens](https://github.com/carronj/plancklens)
  * [lenspyx](https://github.com/carronj/lenspyx)
  * [DUCC](https://github.com/mreineck/ducc)

## Doc
Documentation may be found [HERE]


## Use with HPC
delensalot is computationally demanding.
We have parallelized the computations across the simulation index in most cases.
To use delensalot on any HPC infrastructure, set up MPI accordingly. Your HPC-center can help.