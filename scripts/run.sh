#!/bin/bash
#SBATCH -N 50
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J examples/c08b.py
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# file='/global/cscratch1/sd/sebibel/cmbs4/08b_07_OBD_MF100_example/config_mfvar.py'
# file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_00_OBD_cnv035/c08b_v2.py'

#run the application:
# srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lenscarf/run.py -r $file
srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lenscarf/run.py -p examples/c08b.py
echo $file
