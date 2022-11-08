#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J get_mf_it
#SBATCH --mail-user=louis.legrand@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 30:00
#SBATCH --output=slurm-%x-%J-%s-%t.out


echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo ""
echo "***** LAUNCHING *****"
echo `date '+%F %H:%M:%S'`
echo ""


module load fftw
module load gsl
module load cfitsio
module swap PrgEnv-intel PrgEnv-gnu
module load python
source activate scarf37gnu

#OpenMP settings:
# export OMP_NUM_THREADS=16
# export OMP_PLACES=threads
# export OMP_PROC_BIND=false 

#run the application:
srun -n 1 -c 32 -u --cpu_bind=cores python $HOME/lenscarf/scripts/get_mf.py

echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
