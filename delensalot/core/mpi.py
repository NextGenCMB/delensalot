"""mpi4py wrapper module, supporting send/receive.

"""

from __future__ import print_function
import logging
log = logging.getLogger(__name__)

import os, sys, importlib
import platform
import multiprocessing
import importlib.util


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
    name = 'mpi4py'
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
        return True
    spec = importlib.util.find_spec(name)
    if spec is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
        return True
    else:
        print(f"can't find the {name!r} module")
        return False


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
    global disabled, verbose, has_key, mpisupport, name
    disabled = False
    verbose = True
    has_key = lambda key : key in os.environ.keys()
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        mpisupport = size > 1
    except ImportError:
        mpisupport = False
    pmisupport = 'PMI_CRAY_NO_SMP_ORDER' in os.environ.keys()
    # mpisupport = not has_key('NERSC_HOST') or (has_key('SLURM_SUBMIT_DIR') and has_key('NERSC_HOST'))
    name = "{} with {} cpus".format(platform.processor(),multiprocessing.cpu_count())

    if not is_notebook() and (mpisupport or pmisupport) and isinstalled():
        print('mpisupport: {}, pmisupport: {}'.format(mpisupport, pmisupport))
        init()
    else:
        print('mpisupport: {}, pmisupport: {}'.format(mpisupport, pmisupport))
        disable()



def disable():
    
    global barrier, send, receive, bcast, ANY_SOURCE, name, rank, size, finalize, disabled
    print('disabling mpi')
    barrier = lambda: -1
    send = lambda _, dest: 0
    receive = lambda _, source: _
    bcast = lambda _, root=0: _
    ANY_SOURCE = 0
    disabled = True
    rank = 0
    size = 1
    finalize = lambda: -1
    log.info('mpi.py : disabled, rank %s in %s' % (rank, size))

def init():

    global barrier, send, receive, bcast, ANY_SOURCE, name, rank, size, finalize, disabled
    print('enabling mpi')
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
    print('mpi.py : setup OK, rank %s in %s' % (rank, size))

enable()


# import os, sys, platform, logging
# import multiprocessing
# import warnings
# from importlib.util import find_spec
# import re

# log = logging.getLogger(__name__)
# rank, size, disabled = 0, 1, True
# barrier = send = receive = bcast = finalize = lambda *a, **kw: None
# ANY_SOURCE = 0
# n_cpus = os.cpu_count()
# OMP_threads = os.environ.get("OMP_NUM_THREADS", "not set")
# hostname = platform.node()


# def is_notebook():
#     try: return 'ZMQ' in get_ipython().__class__.__name__
#     except: return False

# def is_installed(pkg="mpi4py"):
#     return pkg in sys.modules or find_spec(pkg)

# def detect_env():
#     if "SLURM_JOB_ID" in os.environ:
#         return "slurm_compute_node"
#     elif any(env in os.environ for env in ["SLURM_CLUSTER_NAME", "SLURM_CONF"]):
#         return "slurm_login_node"
#     else:
#         return "home_station"

# def init():
#     global rank, size, barrier, send, receive, bcast, finalize, ANY_SOURCE, disabled
#     try:
#         from mpi4py import MPI
#         comm = MPI.COMM_WORLD
#         rank, size = comm.Get_rank(), comm.Get_size()
#         barrier, send, receive, bcast = comm.Barrier, comm.send, comm.recv, comm.bcast
#         finalize, ANY_SOURCE = MPI.Finalize, MPI.ANY_SOURCE
#         disabled = False
#         log.info(f"MPI initialized: rank {rank}, size {size}")
#     except Exception as e:
#         log.warning(f"MPI init failed: {e}")
#         disable()

# def disable():
#     global rank, size, disabled
#     rank, size, disabled = 0, 1, True
#     log.info("MPI disabled")

# def enable(verbose=True):
#     if is_notebook():
#         disable()
#         return
#     try:
#         from mpi4py import MPI
#         comm = MPI.COMM_WORLD
#         global rank, size, barrier, send, receive, bcast, finalize, ANY_SOURCE, disabled
#         rank, size = comm.Get_rank(), comm.Get_size()
#         barrier, send, receive, bcast = comm.Barrier, comm.send, comm.recv, comm.bcast
#         finalize, ANY_SOURCE = MPI.Finalize, MPI.ANY_SOURCE
#         disabled = False
#     except Exception as e:
#         if verbose:
#             if not rank: print(f"[env: {detect_env()}] mpi4py load failed: {e} | OMP_NUM_THREADS={OMP_threads}")
#         disable()

# def print_mpi_info():
#     if not rank: print(f"MPI:\t\t\t{'enabled' if not disabled else 'disabled'}\nMPI task-size:\t\t{size} (the number of simulations processed in parallel)\nOMP_NUM_Threads:\t{OMP_threads} (CPUs per simulation)\nTotal system CPUs:\t{n_cpus}\nEstimated idle CPUs:\t{int(n_cpus)-int(size)*int(OMP_threads)}\nHost:\t\t\t{hostname}\n --------- ")


# def parse_slurm_script(script_path):
#     config = {
#         "nodes": 1,
#         "ntasks": 1,
#         "ntasks_per_node": None,
#         "cpus_per_task": 1,
#         "mem_per_task": None,
#         "gres": None,
#         "env_vars": {}
#     }

#     with open(script_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("#SBATCH"):
#                 tokens = line[7:].strip().split()
#                 for tok in tokens:
#                     if tok.startswith("--nodes="):
#                         config["nodes"] = int(tok.split("=")[1])
#                     elif tok.startswith("--ntasks="):
#                         config["ntasks"] = int(tok.split("=")[1])
#                     elif tok.startswith("--ntasks-per-node="):
#                         config["ntasks_per_node"] = int(tok.split("=")[1])
#                     elif tok.startswith("--cpus-per-task="):
#                         config["cpus_per_task"] = int(tok.split("=")[1])
#                     elif tok.startswith("--mem="):
#                         config["mem_per_task"] = tok.split("=")[1]
#                     elif tok.startswith("--gres="):
#                         config["gres"] = tok.split("=")[1]
#             elif line.startswith("export "):
#                 import re
#                 match = re.match(r"export\s+(\w+)\s*=\s*(.*)", line)
#                 if match:
#                     var, val = match.groups()
#                     config["env_vars"][var] = val

#     if config["ntasks_per_node"] is None and config["nodes"] > 0:
#         config["ntasks_per_node"] = config["ntasks"] // config["nodes"]

#     omp_threads = config["env_vars"].get("OMP_NUM_THREADS")
#     if omp_threads and omp_threads.isdigit():
#         config["cpus_per_task"] = int(omp_threads)

#     print("Parsed SLURM script settings:")
#     print(f"{'nodes':>16}: {config['nodes']}")
#     print(f"{'ntasks':>16}: {config['ntasks']}  # number of simulations processed in parallel (MPI tasks)")
#     print(f"{'ntasks_per_node':>16}: {config['ntasks_per_node']}")
#     print(f"{'cpus_per_task':>16}: {config['cpus_per_task']}  # number of CPU cores used per simulation (OpenMP threads)")
#     print(f"{'mem_per_task':>16}: {config['mem_per_task']}")
#     # print(f"{'gres':>16}: {config['gres']}")
#     # print(f"{'env_vars':>16}: {config['env_vars']}")
#     return config

# def check_hardware_compliance(config, rank, env_name):
#     if rank != 0 or env_name not in ["home_station", "slurm_compute_node"]:
#         print("Compliance can not be checked on login nodes. Please refer to the HPC documentation.")
#         return
#     if env_name == "home_station":
#         print("Checking compliance on home station is not sensible")
#     total_cpus_requested = config["ntasks"] * config["cpus_per_task"]

#     sys_cpus = multiprocessing.cpu_count()

#     if total_cpus_requested > sys_cpus:
#         print(f"Total CPUs per node requested ({total_cpus_requested}) exceed available CPUs ({sys_cpus}).")

#     if config["mem_per_task"]:
#         try:
#             import psutil
#             sys_mem_mb = psutil.virtual_memory().total // (1024*1024)

#             mem_str = config["mem_per_task"].lower()
#             if mem_str.endswith('g'):
#                 total_mem_requested = int(mem_str[:-1]) * config["ntasks"] * 1024
#             elif mem_str.endswith('m'):
#                 total_mem_requested = int(mem_str[:-1]) * config["ntasks"]
#             else:
#                 total_mem_requested = None

#             if total_mem_requested and total_mem_requested > sys_mem_mb:
#                 print(f"Total memory requested ({total_mem_requested} MB) exceeds available system memory ({sys_mem_mb} MB).")
#         except ImportError:
#             print("psutil not installed; skipping memory check.")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.WARNING)
#     enable()
#     print_mpi_info()
#     barrier()
#     print(f"Hello from rank {rank}/{size}")

#     if len(sys.argv) == 2:
#         if not rank:
#             parsed = parse_slurm_script(sys.argv[1])
#             check_hardware_compliance(parsed, rank, detect_env())

