"""Builds OBD matrix for temperature smicadx12 2018
adapted from PL2018.
"""

import os
import numpy as np
from lenscarf import utils_scarf
from plancklens.helpers import mpi
from lenscarf.opfilt import tmodes_ninv
from lenscarf.utils import read_map

lmax_marg = 200  # max marged multipole
nside = 2048     # healpix res of the pixel inverse variance map
geom = utils_scarf.Geom.get_healpix_geometry(nside)
prefix = ''

def get_tpl(rescal=1.):
    """This returns the template instance.
    
        We can only call this after running the __main__ part below

    """
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