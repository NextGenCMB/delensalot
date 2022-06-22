#!/bin/bash
#SBATCH -N 50
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J 08b09_D.lensalot
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 01:30:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lerepi/run.py -r /global/homes/s/sebibel/git/lerepi/lerepi/config/examples/c08b.py