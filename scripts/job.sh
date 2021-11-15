#!/bin/bash
#SBATCH -N 25
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J 00MAPB_tmplt
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 5:00:00


#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#params/s08b_00/par_s08b00_cILC_4000.py -imin 30 -imax 30 -itmax 12 -btempl
#run the application:
echo $SLURM_ARRAY_TASK_ID
# srun -n 1 --cpu_bind=cores /global/homes/s/sebibel/git/cmbs4/qlmsdd.sh $SLURM_ARRAY_TASK_ID
srun -n 50 -c 32 --cpu_bind=cores python3 lerepi/params/90.91/par_90.91_cILC_4000.py -imin 450 -imax 499 -itmax 12 -btempl
# srun -n 100 -c 32 --cpu_bind=cores python3 params/s08b_07/par_s08b07_cILC_4000.py -imin $((4*$SLURM_ARRAY_TASK_ID)) -imax $((4*($SLURM_ARRAY_TASK_ID+1))) -itmax 12 -btempl
# srun -n 8 python3 params/s08b_00/par_s08b00_cILC_4000.py -imin 100 -imax 100 -itmax 12 -btempl
# salloc

# S B A T C H - - a r r a y=0-100
##### how to
# OMP_NUM_THREADS must be set BEFORE salloc - i hope that is not true
# -c in srun doesnt seem to have an effect, perhaps due to -n
# -n tasks per node
# a task is a process in SLURM-language. a process could either be a srun, or a python3 call
# Haswell regular has 32 CPUs per node, haswell interactive has 64?