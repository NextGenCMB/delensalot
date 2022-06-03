#!/bin/bash
#SBATCH -N 25
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J D.lensalot
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00

#OpenMP settings:
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 100 -c 16 --cpu_bind=cores global/homes/s/sebibel/git/lerepi/run.py -r $1