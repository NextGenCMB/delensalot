"""mpi4py wrapper module, supporting send/receive.

"""

from __future__ import print_function
import os

import logging
log = logging.getLogger(__name__)

verbose = True
has_key = lambda key : key in os.environ.keys()
cond4mpi4py = not has_key('NERSC_HOST') or (has_key('NERSC_HOST') and has_key('SLURM_SUBMIT_DIR'))

if cond4mpi4py:
    # try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
    send = MPI.COMM_WORLD.Send
    receive = MPI.COMM_WORLD.Recv
    barrier = MPI.COMM_WORLD.Barrier
    finalize = MPI.Finalize
    log.info('mpi.py : setup OK, rank %s in %s' % (rank, size))
else:
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    receive = lambda val, src : 0
    send = lambda val, dst : 0