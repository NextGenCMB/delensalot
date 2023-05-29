"""mpi4py wrapper module, supporting send/receive.

"""

from __future__ import print_function
import logging
log = logging.getLogger(__name__)

import os, sys, importlib
import platform
import multiprocessing


def check_MPI(func):
    global name, rank, size
    def inner_function(*args, **kwargs):
        log.info("rank: {}, size: {}, name: {}".format(rank, size, name))
        return func(*args, **kwargs)
    return inner_function

def check_MPI_inline():
    global name, rank, size
    log.info("rank: {}, size: {}, name: {}".format(rank, size, name))


def isinstalled():
    # For illustrative purposes.
    name = 'MPI'

    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")


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


def enable():
    global disabled, verbose, has_key, cond4mpi4py, name
    disabled = False
    verbose = True
    has_key = lambda key : key in os.environ.keys()
    cond4mpi4py = not has_key('NERSC_HOST') or (has_key('SLURM_SUBMIT_DIR') and has_key('NERSC_HOST'))
    name = "{} with {} cpus".format(platform.processor(),multiprocessing.cpu_count())

    if not is_notebook() and cond4mpi4py and isinstalled():
        print(cond4mpi4py)
        print('cond4mpi exists')
        init()
        print('mpi.py : setup OK, rank %s in %s' % (rank, size))
    else:
        print('cond4mpi does not exists. No MPI loaded')
        disable()



def disable():
    global barrier, send, receive, bcast, ANY_SOURCE, name, rank, size, finalize, disabled
    barrier = lambda: -1
    send = lambda _, dest: 0
    receive = lambda _, source: 0
    bcast = lambda _: 0
    ANY_SOURCE = 0
    disabled = True
    rank = 0
    size = 1
    finalize = lambda: -1

def init():
    global barrier, send, receive, bcast, ANY_SOURCE, name, rank, size, finalize, disabled
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    barrier = MPI.COMM_WORLD.Barrier
    ANY_SOURCE = MPI.ANY_SOURCE
    send = MPI.COMM_WORLD.send
    receive = MPI.COMM_WORLD.recv
    bcast = MPI.COMM_WORLD.bcast
    finalize = MPI.Finalize
    log.info('mpi.py : setup OK, rank %s in %s' % (rank, size))

enable()