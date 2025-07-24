import os, sys, platform, logging
from importlib.util import find_spec

log = logging.getLogger(__name__)
rank, size, disabled = 0, 1, True
barrier = send = receive = bcast = finalize = lambda *a, **kw: None
ANY_SOURCE = 0
n_cpus = os.cpu_count()
OMP_threads = os.environ.get("OMP_NUM_THREADS", "not set")
hostname = platform.node()

def is_notebook():
    try: return 'ZMQ' in get_ipython().__class__.__name__
    except: return False

def is_installed(pkg="mpi4py"):
    return pkg in sys.modules or find_spec(pkg)

def detect_env():
    if "SLURM_JOB_ID" in os.environ:
        return "slurm_compute_node"
    elif any(env in os.environ for env in ["SLURM_CLUSTER_NAME", "SLURM_CONF"]):
        return "slurm_login_node"
    else:
        return "home_station"

def init():
    global rank, size, barrier, send, receive, bcast, finalize, ANY_SOURCE, disabled
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        barrier, send, receive, bcast = comm.Barrier, comm.send, comm.recv, comm.bcast
        finalize, ANY_SOURCE = MPI.Finalize, MPI.ANY_SOURCE
        disabled = False
        log.info(f"MPI initialized: rank {rank}, size {size}")
    except Exception as e:
        log.warning(f"MPI init failed: {e}")
        disable()

def disable():
    global rank, size, disabled
    rank, size, disabled = 0, 1, True
    log.info("MPI disabled")

def enable(verbose=True):
    if is_notebook():
        disable()
        return
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        global rank, size, barrier, send, receive, bcast, finalize, ANY_SOURCE, disabled
        rank, size = comm.Get_rank(), comm.Get_size()
        barrier, send, receive, bcast = comm.Barrier, comm.send, comm.recv, comm.bcast
        finalize, ANY_SOURCE = MPI.Finalize, MPI.ANY_SOURCE
        disabled = False
        if verbose:
            print(f"[env: {detect_env()}] mpi4py available: {is_installed('mpi4py')} | OMP_NUM_THREADS={OMP_threads}")
    except Exception as e:
        if verbose:
            print(f"[env: {detect_env()}] mpi4py load failed: {e} | OMP_NUM_THREADS={OMP_threads}")
        disable()

def print_mpi_info():
    print(f"[Rank {rank}/{size}] Host: {hostname} | CPUs: {n_cpus} | Threads: {OMP_threads} | MPI: {'enabled' if not disabled else 'disabled'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    enable()
    print_mpi_info()
    barrier()
    print(f"Hello from rank {rank}/{size}")
