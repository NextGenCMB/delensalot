"""Builds OBD matrix for polarisation CMB-S4 08d
adapted from PL2018.

Note:
    It takes about 60min to calculate all rows when using 2 nodes with -c 4, for bmarg_lmax=200,
    30min to collect, and 60min to store    
"""

import os
import numpy as np
import healpy as hp

from lenscarf import utils_scarf
from lenscarf.utils import read_map
from lenscarf.opfilt import bmodes_ninv

from plancklens.helpers import mpi


lmax_marg = 200  # max marged multipole
nside = 2048     # healpix res of the pixel inverse variance map
geom = utils_scarf.Geom.get_healpix_geometry(nside)
prefix = ''

def get_tpl(rescal=1.):
    """This returns the template instance.
    
        We can only call this after running the __main__ part below

    """
    if False:
        from PL2018.params import smicadx12_planck2018 as parfile
        libdir = os.path.join(parfile.TEMP, 'OBD')
        if not os.path.exists(libdir) and mpi.rank == 0:
            os.makedirs(libdir)
        mpi.barrier()
        tpl = tmodes_ninv.template_dense(lmax_marg, geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=libdir, rescal=rescal)
        return tpl



if __name__ == '__main__':
    # This scripts uses a number of processes to calculate all rows of the dense OBD matrix
    # After calculting the rows, the rows are collected by the first process, who then regularize it,  inverse it,
    # and cache the inverse under the name tniti.npy in the specified folder
    TEMP =  '/global/cscratch1/sd/sebibel/cmbs4/s08d/OBDmatrix/'
    libdir = os.path.join(TEMP, 'OBD')


    pol = True
    if pol == True:

        """
        Find ninv_p
        THIS NEEDS TO BE CHANGED FOR A NEW SIM DATA SET
        """
        centralnoiselevel = .59
        rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits'))
        pixlev = centralnoiselevel / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
        print("Pmap center pixel pol noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
        ninv_p = 1./ pixlev ** 2 * rhits


        libdir = os.path.join(TEMP, 'OBD')
        if not os.path.exists(libdir) and mpi.rank == 0:
            os.makedirs(libdir)
        mpi.barrier()

        NiT = ninv_p  #NiT = read_map(ninv_p) # inverse pixel variance map
        bpl = bmodes_ninv.template_bfilt(lmax_marg, geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=libdir)
        bpl._get_rows_mpi(NiT, prefix)  # builds all rows in parallel
        mpi.barrier()
        if mpi.rank == 0:
            assert not os.path.exists(bpl.lib_dir + '/tniti.npy')
            tnit = bpl._build_tniti()
            nlevp = 10000.  # regularization, saying the modes have in fact huge noise
            tniti = np.linalg.inv(tnit + np.diag((1. / (nlevp / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
            np.save(bpl.lib_dir + '/tniti.npy', tniti)
            # Now cleaning up. We delete all these rows that were previously saved
            import glob
            fns = glob.glob(bpl.lib_dir + '/row*.npy')
            for fn in fns:
                os.remove(fn)
        mpi.barrier()
        mpi.finalize()


        """
        1. create ninv_p
        2. run this
        """


    """This is the base
    if __name__ == '__main__':
    # This scripts uses a number of processes to calculate all rows of the dense OBD matrix
    # After calculting the rows, the rows are collected by the first process, who then regularize it,  inverse it,
    # and cache the inverse under the name tniti.npy in the specified folder
    from lerepi.params.s08d import s08d as parfile

    libdir = os.path.join(parfile.TEMP, 'OBD')
    if not os.path.exists(libdir) and mpi.rank == 0:
        os.makedirs(libdir)
    mpi.barrier()


    NiT = read_map(parfile.ninv_t)  # inverse pixel variance map
    tpl = tmodes_ninv.template_tfilt(lmax_marg, geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=libdir)
    tpl._get_rows_mpi(NiT, prefix)  # builds all rows in parallel
    mpi.barrier()
    if mpi.rank == 0:
        assert not os.path.exists(tpl.lib_dir + '/tniti.npy')
        tnit = tpl._build_tnit('')
        nlevt = 10000.  # regularization, saying the modes have in fact huge noise
        tniti = np.linalg.inv(tnit + np.diag((1. / (nlevt / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
        np.save(tpl.lib_dir + '/tniti.npy', tniti)
        # Now cleaning up. We delete all these rows that were previously saved
        import glob
        fns = glob.glob(tpl.lib_dir + '/row*.npy')
        for fn in fns:
            os.remove(fn)
    mpi.barrier()
    mpi.finalize()
    """