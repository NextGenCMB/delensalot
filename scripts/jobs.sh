# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/cmbs4/08b_00_OBD_MF100_example/config_mfvar.py
# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/cmbs4/08b_00_OBD_MF100_example/example_c08b.py

# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/cmbs4/08b_07_OBD_MF100_example/config_mfvar.py
# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/cmbs4/08b_07_OBD_MF100_example/example_c08b.py

# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_00_OBD/c08b_v2.py
# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_07_OBD/c08b_v2.py
# srun -c 4 python3 run.py -r /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_09_OBD/c08b_v2.py

srun -c 32 python3 run.py -r /global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08b/caterinaILC_May12_00_OBD_cnv035/c08b_v2.py # (in sbatch queue)
srun -c 32 python3 run.py -r '/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_00_OBD_rinf_tol5e5/c08d_v2.py' # (in sbatch queue)
srun -c 32 python3 run.py -r /global/homes/s/sebibel/gi/global/cscratch1/sd/sebibel/dlensalot/lerepi/data_08d/ILC_May2022_00_OBD_rinf_tol5e5/c08d_v2.pyt/lerepi/lerepi/config/examples/c08d_rediimdi.py # TBD