#!/bin/bash
#SBATCH -N 25
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J 8dv2tol5e5_D.lensalot
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 01:30:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lerepi/run.py -r /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_00_OBD_cnv035/c08b_v2.py

echo 'data_08d/ILC_May2022_00_OBD_rinf_tol5e5/c08d_v2.py'