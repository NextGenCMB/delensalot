#!/bin/bash
#SBATCH -N 10
##SBATCH -C haswell # for cori
#SBATCH -C cpu  # for perlmutter
#SBATCH -q regular
#SBATCH -J mask_QE_simMF
#SBATCH --mail-user=louis.legrand@unige.ch
#SBATCH --mail-type=ALL
#SBATCH -t 1:00:00
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
# source activate scarf37gnu

#OpenMP settings:
# export OMP_NUM_THREADS=8
# export OMP_PLACES=threads
# export OMP_PROC_BIND=false 
# export OMP_PROC_BIND=spread

#run the application:
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_noaberration_bis.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lmindlm1.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminivf30.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim30.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim30_lmindlm2.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2_cstMF.py -imin 0 -imax 319 -k p_p -ivp -dd
srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_planckmask_lminsim2_simMF.py -imin 0 -imax 319 -k p_p -ivp -dd
# srun -n 320 -c 8 -u --cpu_bind=cores python  $HOME/plancklens/examples/run_qlms.py  $HOME/lenscarf/lenscarf/params/cmbs4wide_idealized.py -imin 0 -imax 319 -k p_p -ivp -dd


echo ""
echo "***** DONE *****"
echo `date '+%F %H:%M:%S'`
echo ""
