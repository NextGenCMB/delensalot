#!/bin/bash
#SBATCH -N 4
##SBATCH -C haswell # for cori
#SBATCH -C cpu  # for perlmutter
#SBATCH -q regular
#SBATCH -J ideal_cstMF
#SBATCH --mail-user=louis.legrand@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%x-%J.out


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

# module load PrgEnv-gnu
# module load e4s/21.11-tcl
# module load fftw/3.3.10-gcc-11.2.0-mpi

# source activate scarf37gnu
module load python
module load cudatoolkit
source activate scarf37gnu

# module use /global/common/software/m3169/perlmutter/modulefiles
# module load openmpi

# #OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#run the application:

# srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized.py -k p_p -itmax 1000  -imin 0 -imax 31 -tol 5
# srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_cstMF.py -k p_p -itmax 1000  -imin 0 -imax 31 -tol 5
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration.py -k p_p -itmax 1000  -imin 0 -imax 3 -tol 5
# srun -n 8 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_lmindlm2_lminsim30.py -k p_p -itmax 1000  -imin 0 -imax 7 -tol 5
# srun -n 8 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_lmindlm2.py -k p_p -itmax 100  -imin 0 -imax 7 -tol 5
# srun -n 8 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_lminsim30.py -k p_p -itmax 100  -imin 0 -imax 7 -tol 5
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_lminsim2_cstMF.py -k p_p -itmax 100  -imin 0 -imax 7 -tol 5
# srun -n 8 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_cstMF.py -k p_p -itmax 1000  -imin 0 -imax 7 -tol 5

# srun -n 8 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim30_lmindlm2.py -k p_p -itmax 1000  -imin 0 -imax 7 -tol 5
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -k p_p -itmax 1000  -imin 1 -imax 4 -tol 5
# srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim30.py -k p_p -itmax 1000  -imin 0 -imax 31 -tol 5
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2.py -k p_p -itmax 1000  -imin 0 -imax 3 -tol 5
# srun -n 32 -c 8 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2_cstMF.py -k p_p -itmax 50  -imin 8 -imax 39 -tol 5
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2_simMF.py -k p_p -itmax 100  -imin 0 -imax 3 -tol 5
# srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminivf30.py -k p_p -itmax 15  -imin 0 -imax 31 -tol 4
# srun -n 1 -c 256 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_simMF.py -k p_p -itmax 15  -imin 0 -imax 0 -tol 4
# srun -n 4 -c 64 --cpu_bind=cores python   $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lmindlm1.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 4
# srun -n 4 -c 64 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_noaberration_bis.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 5
# srun -n 32 -c 32 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_noaberration.py -k p_p -itmax 15  -imin 0 -imax 31 -tol 5

# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -k p_eb -itmax 15  -imin 0 -imax 3 -tol 3 -v noMF
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -k p_eb -itmax 15  -imin 0 -imax 3 -tol 3
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized.py -k p_eb -itmax 15  -imin 0 -imax 3 -tol 5 -v noMF
# srun -n 32 -c 32 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -k p_p -itmax 15  -imin 0 -imax 254 -tol 5
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 5
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 3 -v noRespMF

# srun -n 4 -c 1024 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing.py -k p_p -itmax 20  -imin 0 -imax 3 -tol 5

# srun -n 4 -c 64 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_maskpoles.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 4
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_lminb30.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 5
# srun -n 4 -c 64 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_lmindlm1.py -k p_p -itmax 30  -imin 0 -imax 3 -tol 5
srun -n 1 -c 32 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_lmindlm1.py -k p_p -itmax 30  -imin 0 -imax 0 -tol 5
# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_delensing_fixedphi.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 5
# srun -n 1 -c 64 -u --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/par_s4P_fs.py -k p_p -itmax 15  -imin 0 -imax 0 -tol 5

# srun -n 4 -c 16 -u --cpu_bind=cores python  $HOME/lenscarf/tests/par_s4P_fs.py -k p_p -itmax 15  -imin 0 -imax 3 -tol 4 -scarf p
# srun -n 1 -c 64  -u --cpu_bind=cores python  $HOME/n32/n32/params/SO.py -imin 0 -imax 0 -itmax 6 -k ptt


# srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2_cstMF.py -k p_p -itmax 50  -imin 8 -imax 39 -tol 5
srun -n 32 -c 32 --cpu_bind=cores python  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized_noaberration_lminsim2_cstMF.py -k p_p -itmax 50  -imin 8 -imax 39 -tol 5



echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
