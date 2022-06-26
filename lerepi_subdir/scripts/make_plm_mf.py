"""Script for calculating lensing potential meanfield
"""

import sys, os
from os.path import join as opj
import argparse
import numpy as np
from itercurv.iterators.statics import rec as Rec

parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=500, help='Maximal simulation index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
args = parser.parse_args()

simid_lower = args.simid_lower
simid_upper = args.simid_upper
fg = args.fg
iteration = [0,11]
simids = np.arange(simid_lower,simid_upper)
averages = simid_upper-simid_lower
TEMP = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/' %fg

plm = np.zeros(shape=(len(iteration),8394753), dtype=complex)
plm_mf1 = np.zeros(shape=(len(iteration),8394753), dtype=complex)
plm_mf2 = np.zeros(shape=(len(iteration),8394753), dtype=complex)

for simidx, simid in enumerate(simids):
    lib_dir_iterator = TEMP + '/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20' % simid
    plm = Rec.load_plms(lib_dir_iterator, iteration)
    if simidx <= averages/2.:
        plm_mf1 += plm
    else:
        plm_mf2 += plm
    if simidx % 10 == 0:    
        print('Plm {}/{} added.'.format(simidx+1, len(simids)))

np.save('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/plm_fg%s_mf1o2_itmax%s.npy'%(fg,str(iteration[-1])), plm_mf1/(averages/2.))
np.save('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/plm_fg%s_mf2o2_itmax%s.npy'%(fg,str(iteration[-1])), plm_mf2/(averages/2.))