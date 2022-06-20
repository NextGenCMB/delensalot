#!/bin/bash
#SBATCH -N 250
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J D.lensalot
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 00:45:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lerepi/run.py -p /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_09_OBD/c08b_v2.py