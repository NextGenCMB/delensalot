#!/bin/bash
#SBATCH -N 25
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J 8bmfvar
#SBATCH --mail-user=sebastian.belkner@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 04:00:00
#SBATCH -o ./add/slurm/slurm-%j.out

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# --test-only as sbatch# looks helpful

# file='/global/cscratch1/sd/sebibel/cmbs4/08b_07_OBD_MF100_noQEmfsub/c08b.py'

# file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_00_OBD_cnv035/c08b_v2.py'

# file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_07_OBD_r10_tol5e5/c08d_v2.py'
# file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_00_OBD_r100_tol5e5/c08d_v2.py'
# file='/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_00_OBD_r10_tol5e5/c08d_v2.py'

file='/global/cscratch1/sd/sebibel/cmbs4/08b_07_OBD_MF100_example/config_mfvar.py'
echo $file
cat $file

srun -c 32 --cpu_bind=cores python3 /global/homes/s/sebibel/git/lenscarf/run.py -r $file

echo $file
cat $file