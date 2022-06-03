"""Builds OBD matrix for polarisation CMB-S4 08d
adapted from PL2018.

This scripts uses a number of processes to calculate all rows of the dense OBD matrix
After calculting the rows, the rows are collected by the first process, who then regularize it,  inverse it, and cache the inverse under the name tniti.npy in the specified folder

Note:
    It takes about 60min to calculate all rows when using 2 nodes with -c 4, for bmarg_lmax=200,
    30min to collect, and 60min to invert    
"""

import os
import numpy as np
import healpy as hp

from lenscarf import utils_scarf
from lenscarf.opfilt import bmodes_ninv
from lenscarf.core.decorators.io import iohelper

from lerepi.data.dc08 import data_08d as sims_if
from plancklens.helpers import mpi


data = sims_if.ILC_May2022('00')
lmax_marg = 200  # max marged multipole
prefix = ''
lib_dir = os.path.join('/global/cscratch1/sd/sebibel/cmbs4/s08d/', 'OBD')


def calc_ninvp(centralnoiselevel = 0.59):
    """
    Central noise level comes from notebook
    """
    mask = data.get_mask()
    mask = np.where(mask<0.0001,0,mask)
    pixlev = centralnoiselevel / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    ninv_p = 1./ pixlev ** 2 * mask
    return ninv_p


def cleanup():
    """
    Remove all 'row'-files. As a safety layer, will only execute if tniti.py has been calculated
    and can be found in the same directory
    """
    assert os.path.exists(lib_dir + '/tniti.npy')
    import glob
    fns = glob.glob(lib_dir + '/row*.npy')
    for fn in fns:
        os.remove(fn)


@iohelper
def build_OBD(lib_dir, mpi):
    """
    Calculates..
    1. NiT
    2. tnit
    3. tniti
    """
    
    ninv_p = calc_ninvp() # ninv_p = NiT

    mpi.barrier()
    geom = utils_scarf.Geom.get_healpix_geometry(data.nside)
    bpl = bmodes_ninv.template_bfilt(lmax_marg, geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=lib_dir)
    if not os.path.exists(lib_dir + '/tnit.npy'):
        bpl._get_rows_mpi(ninv_p, prefix)  # builds all rows in parallel
    else:
        mpi.barrier()
        if mpi.rank == 0:
            tnit = bpl._build_tnit()
            np.save(lib_dir + '/tnit.npy', tnit)
        
    nlev_deproj_modes = 10000.  # regularization, saying the modes have in fact huge noise
    tniti = np.linalg.inv(tnit + np.diag((1. / (nlev_deproj_modes / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
    np.save(lib_dir + '/tniti.npy', tniti)
    mpi.barrier()
    mpi.finalize()


if __name__ == '__main__':
    pol = True
    if pol == True:
        build_OBD(lib_dir, mpi)
    elif pol == False:
        assert 0, "Implement if needed"