"""Script for calculating Btemplates 08d using precalculated plms as input. Use 'Generate_plm_mm.py' for calculating plm input
"""

import os, sys
import argparse

import healpy as hp
import numpy as np

from plancklens.helpers import mpi
from itercurv.iterators.statics import rec as Rec
from itercurv.remapping import remapping

from lerepi.data.dc08 import data_08d as sims_if
from lerepi.params.s08 import s08d as paramfile

sims = paramfile.sims

parser = argparse.ArgumentParser(description='Full-sky iterative delensing.')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=500, help='Maximal simulation index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
parser.add_argument('-bs', dest='blm_suffix', type=str, default='', help='Suffix string. Defines where the plms are found and where the blm will be stored.')
parser.add_argument('-it', dest='itid', type=str, default='QEMAP', help='iteration identifier, either QE, MAP, QEMAP or All')
args = parser.parse_args()

simid_lower = args.simid_lower
simid_upper = args.simid_upper
fg = args.fg
blm_suffix = args.blm_suffix
if args.itid == 'QE':
    iteration = [0]
elif args.itid == 'QEMAP':
    iteration = [0,12]
elif args.itid == 'MAP':
    iteration = [12]
elif args.itid == 'All':
    iteration = [1]
simids = np.arange(simid_lower,simid_upper+1)


for simid in simids[mpi.rank::mpi.size]:
    print('starting simid {}'.format(simid))
    for iti, it in enumerate(iteration):
        print('starting iteration {}'.format(it))
        TEMP_simit_loc = paramfile.TEMP_it%simid
        TEMP_it_loc = TEMP_simit_loc + '/ffi_p_it{}{}'.format(it, blm_suffix)
        
        ivfs = paramfile.ivfs_raw
        rm = remapping.cached_deflection(TEMP_it_loc, paramfile.nside, 1, facres=-1, zbounds=paramfile.zbounds_len)
            # plm0 = hp.almxfl(alm_copy(qlms_dd.get_sim_qlm(qe_key, DATIDX), lmax=lmax_qlm) - mf0, qnorm * clwf)
        plm_lensc = Rec.load_plms(TEMP_simit_loc, [it])[0]
        print('0 ok')
        rm.prepare_filtering(plm_lensc)
        print('1 ok')
        if it == 0:
            wflm0 = lambda : alm_copy(ivfs.get_sim_emliklm(simid), lmax=paramfile.lmax_filt)
            elm = wflm0()
            print('2 ok')
        else:
            elm = Rec.load_elm(TEMP_simit_loc, it-1)
            print('3 ok')
        blm = Rec.get_btemplate(TEMP_simit_loc, elm, it, paramfile.pbounds_len, paramfile.zbounds_len, cache=True, lmax_b=1024, ffi_suffix=blm_suffix)
        print('4 ok')
        np.save(lib_dir+'/blm%s_%04d_it%d.npy'%(blm_suffix, simid, it), blm)
        print('  .. it {}/{}'.format(iti, len(iteration)))
    print('simulation {}/{} done.'.format(simid+1, len(simids)))



