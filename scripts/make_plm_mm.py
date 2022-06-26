"""Script for calculating meanfield subtracted plms, using precaulculated MF over 500 simulations. Use 'Generate_plm_mf.py' for plm MF calculation.
"""

import sys, os
import argparse
import numpy as np

from plancklens.helpers import mpi

from itercurv.iterators.statics import rec as Rec

parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=500, help='Maximal simulation index')
index')
parser.add_argument('-itmax', dest='itmax', type=str, default=500, help='Maximal iteration index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
parser.add_argument('-bs', dest='blm_suffix', type=str, default='', help='Suffix string. Defines where the plms are found and where the blm will be stored.')
args = parser.parse_args()

simid_lower = args.simid_lower
simid_upper = args.simid_upper
fg = args.fg
iteration = [0,int(args.itmax)]
simids = np.arange(simid_lower,simid_upper)
averages = simid_upper-simid_lower
blm_suffix = args.blm_suffix
TEMP = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/' %fg
plm_mf1 = np.load('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/plm_fg%s_mf1o2_itmax%s.npy'%(fg,str(iteration[-1])))
plm_mf2 = np.load('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/plm_fg%s_mf2o2_itmax%s.npy'%(fg,str(iteration[-1])))

plm_mm = np.zeros(shape=(len(iteration), 8394753), dtype=complex)
print('Plm ..')
for simidx, simid in enumerate(simids[mpi.rank::mpi.size]):
    lib_dir_iterator = TEMP + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % simid
    plm_mm_dir = lib_dir_iterator+'/ffi_p_it%d%s/'
    if not(os.path.isdir(plm_mm_dir%(iteration[0], blm_suffix))):
        os.mkdir(plm_mm_dir%(iteration[0], blm_suffix))
    if not(os.path.isdir(plm_mm_dir%(iteration[-1], blm_suffix))):
        os.mkdir(plm_mm_dir%(iteration[-1], blm_suffix))
    plm = Rec.load_plms(lib_dir_iterator, iteration)
    if 'fake' in blm_suffix:
        plm_mm = np.array(plm)
    else:
        plm_mm = averages/(averages-1.) * (np.array(plm) - (plm_mf1 + plm_mf2))
    np.save(plm_mm_dir%(iteration[0], blm_suffix)+'plm_mm_%04d_it%d.npy'%(simid, iteration[0]), plm_mm[0])
    np.save(plm_mm_dir%(iteration[-1], blm_suffix)+'plm_mm_%04d_it%d.npy'%(simid, iteration[-1]), plm_mm[-1])
    print('  .. {}/{} done.'.format(simidx+1, len(simids)))