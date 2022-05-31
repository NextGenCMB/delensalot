#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J s08d_delensing
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00

#OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
echo $SLURM_ARRAY_TASK_ID
srun -n 1 -c 64 --cpu_bind=cores python3 lerepi/params/<parfile> -imin <imin> -imax <imax> -itmax <itmax> -btempl