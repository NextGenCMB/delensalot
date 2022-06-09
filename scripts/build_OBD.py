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

from plancklens.helpers import mpi

from lenscarf import utils_scarf
from lenscarf.opfilt import bmodes_ninv

from lerepi.config.cmbs4.data import data_08d as sims_if
from lerepi.config.helper import data_functions as df


lib_dir = os.path.join('/global/cscratch1/sd/sebibel/cmbs4/s08d/', 'OBD_matrix', 'r_inf_normalisedmask')


def calc_ninvp(centralnoiselevel, noisemodel_rhits, nside, nlev_p):
    """
    Central noise level comes from notebook
    """
    
    noisemodel_norm = np.max(noisemodel_rhits)
    ninv_p = np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_p ** 2])/noisemodel_norm * noisemodel_rhits

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


def build_OBD(lib_dir, mpi):
    """
    Calculates..
    1. NiT
    2. tnit
    3. tniti
    """
    
    ninv_p = calc_ninvp() # ninv_p = NiT
    mpi.barrier()
    geom = utils_scarf.Geom.get_healpix_geometry(data.nside_mask)
    bpl = bmodes_ninv.template_bfilt(lmax_marg, geom, int(os.environ.get('OMP_NUM_THREADS', 8)), _lib_dir=lib_dir)
    if not os.path.exists(lib_dir + '/tnit.npy'):
        bpl._get_rows_mpi(ninv_p)  # builds all rows in parallel
    mpi.barrier()
    if mpi.rank == 0:
        tnit = bpl._build_tnit()
        np.save(lib_dir + '/tnit.npy', tnit)
        nlev_deproj_modes = 10000.  # regularization, saying the modes have in fact huge noise
        tniti = np.linalg.inv(tnit + np.diag((1. / (nlev_deproj_modes / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
        np.save(lib_dir + '/tniti.npy', tniti)
        readme = 'This tniti has been created with the following settings: '
        np.save(lib_dir + '/README.txt', readme)
        mpi.barrier()
        mpi.finalize()


def get_noise_model_rhits(ratio, rhits):
    
    return  df.get_nlev_mask(np.inf, rhits)
    

def get_rhits(rh_p, inf):
    
    rhits = hp.read_map(rh_p)
    rhits[rhits == np.inf] = inf

    return rhits


if __name__ == '__main__':
    rhits_path = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'
    rhits_map = get_rhits(rhits_path)
    ratio = np.inf
    inf = 1e4
    nside = 2048
    noisemodel_rhits = get_noise_model_rhits(ratio, rhits_map, nside)
    nlev_deproj_modes = 10000.
    central_noiselevel = 0.59
    nlev_p = central_noiselevel / (np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.)


    setting = {
        'rhits_path': rhits_path,
        'nside': nside,
        'np.inf': inf,
        'central_noiselevel': central_noiselevel,
        'noisemodel_rhits_ratio': ratio,
        'noisemodel_norm': np.max(noisemodel_rhits),
        'nlev_p': nlev_p,
        'nlev_deproj_modes': nlev_deproj_modes,
        'lmax_marg': lmax_marg
}

    if mpi.rank == 0:
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
            print('created dir {}'.format(lib_dir))
    pol = True
    if pol == True:
        build_OBD(lib_dir, mpi)
    elif pol == False:
        assert 0, "Implement if needed"