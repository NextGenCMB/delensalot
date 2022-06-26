#!/bin/bash
#SBATCH -N 25
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J 8dv2tol5e5_D.lensalot
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 02:30:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_00_OBD_rinf_tol5e5/c08d_v2.py'

#run the application:
srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lerepi/run.py -r $file

echo $file