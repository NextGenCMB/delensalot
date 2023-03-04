![D.lensalot logo](res/dlensalot2.PNG)
# D.Lensalot
Curved-sky iterative CMB lensing tools

## Installation
Download the project to your computer, navigate to the root folder and execute the command,

``` 
python setup.py install
```


# Usage

Type `python3 run.py [-h]` for quickhelp,
```
usage: run.py [-h] [-p NEW] [-r RESUME] [-s STATUS] [-purgehashs PURGEHASHS]

D.lensalot entry point.

optional arguments:
  -h, --help            show this help message and exit
  -p NEW                Relative path to config file to run analysis.
  -r RESUME             Absolute path to config file to resume.
  -s STATUS             Absolute path for the analysis to write a report.

```

## Run a configuration file

To run a configutation file `<path-to-config/conf.py>`, type in your favourite `bash`,
```
python3 run.py -r <path-to-config/conf.py>
```

Delensalot supports MPI,

```
srun --nodes <nnodes> -n <taskspernode> python3 run.py -r <path-to-config/conf.py>
```

If you'd like to know the status of the analysis done with `<path-to-config/conf.py>`, run,
```
python3 run.py -s <path-to-config/conf.py>
```

## interactive mode

D.lensalot supports interactive mode. See `delensalot/notebooks/examples/` for guidance.




# Dependencies

 based on
  * [Scarf](https://github.com/samuelsimko/scarf)
  * [Plancklens](https://github.com/carronj/plancklens)

## Doc
Documentation may be found [HERE]


## Use with HPC
D.lensalot is computationally demanding.
Alternative, add the above lines to your `~/.bash_profile`

To use D.lensalot on any HPC infrastructure, set up MPI accordingly. Your HPC-center can help.