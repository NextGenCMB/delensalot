#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J rdn0_ideal_cstMF
#SBATCH --mail-user=louis.legrand@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
#SBATCH --output=slurm-%x-%J-%A.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""


# module load fftw
# module load gsl
# module load cfitsio
# module swap PrgEnv-intel PrgEnv-gnu
# module load python
# source activate scarf37gnu

# #OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=false 

# srun -n 64 -c 16 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_planckmask -datidx 0 -tol 4
# srun -n 64 -c 16 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_planckmask -datidx ${SLURM_ARRAY_TASK_ID} -tol 4
# srun -n 64 -c 16 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_idealized_noaberration -datidx ${SLURM_ARRAY_TASK_ID} -tol 4
# srun -n 1 -c 256 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_planckmask_lminsim2_cstMF -datidx ${SLURM_ARRAY_TASK_ID} -tol 4 -itmax 50 -Nsims 96 -Nroll 8
# srun -n 16 -c 16 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_idealized_noaberration_lminsim2_cstMF -datidx ${SLURM_ARRAY_TASK_ID} -tol 4 -itmax 50 -Nsims 96 -Nroll 8
srun -n 16 -c 16 python /global/homes/l/llegrand/lenscarf/lenscarf/rdn0_cs.py -par cmbs4wide_idealized_noaberration_lminsim2_cstMF -datidx ${SLURM_ARRAY_TASK_ID} -tol 4 -itmax 50 -Nsims 96 -Nroll 8


echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
