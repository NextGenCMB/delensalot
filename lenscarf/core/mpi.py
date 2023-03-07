"""mpi4py wrapper module, supporting send/receive.

"""

from __future__ import print_function
import logging
log = logging.getLogger(__name__)

import os
import platform
import multiprocessing


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
    
def is_local() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
    

def check_MPI(func):
    def inner_function(*args, **kwargs):
        log.info("rank: {}, size: {}, name: {}".format(rank, size, name))
        return func(*args, **kwargs)
    return inner_function

def check_MPI_inline():
    log.info("rank: {}, size: {}, name: {}".format(rank, size, name))
    

verbose = True
has_key = lambda key : key in os.environ.keys()
cond4mpi4py = not has_key('NERSC_HOST') or (has_key('SLURM_SUBMIT_DIR') and has_key('NERSC_HOST'))
if not is_notebook() and cond4mpi4py:
    print('cond4mpi exists')
    from mpi4py import MPI


    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    barrier = MPI.COMM_WORLD.Barrier
    ANY_SOURCE = MPI.ANY_SOURCE
    send = MPI.COMM_WORLD.send
    receive = MPI.COMM_WORLD.recv
    bcast = MPI.COMM_WORLD.bcast
    status = MPI.Status
    iprobe = MPI.COMM_WORLD.Iprobe
    Ibcast = MPI.COMM_WORLD.Ibcast
    Isend = MPI.COMM_WORLD.Isend
    Ireceive = MPI.COMM_WORLD.Irecv
    finalize = MPI.Finalize
    name = "{} with {} cpus".format( platform.processor(),multiprocessing.cpu_count())
    log.info('mpi.py : setup OK, rank %s in %s' % (rank, size))
else:
    print('cond4mpi does not exists')
    log.info("No MPI loaded")
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    bcast = lambda _: 0
    receive = lambda val, src : 0
    send = lambda val, dst : 0

    
    name = "{} with {} cpus".format( platform.processor(),multiprocessing.cpu_count())