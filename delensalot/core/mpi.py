import os, sys, platform, logging
import multiprocessing
import warnings
from importlib.util import find_spec
import re

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
            if not rank: print(f"[env: {detect_env()}] mpi4py available: {is_installed('mpi4py')} | OMP_NUM_THREADS={OMP_threads}")
    except Exception as e:
        if verbose:
            if not rank: print(f"[env: {detect_env()}] mpi4py load failed: {e} | OMP_NUM_THREADS={OMP_threads}")
        disable()

def print_mpi_info():
    if not rank: print(f"MPI task-size: {size}\nHost: {hostname}\nTotal system CPUs: {n_cpus}\nThreads (CPUs per task): {OMP_threads} | MPI: {'enabled' if not disabled else 'disabled'}")


def parse_slurm_script(script_path):
    config = {
        "nodes": 1,
        "ntasks": 1,
        "ntasks_per_node": None,
        "cpus_per_task": 1,
        "mem_per_task": None,
        "gres": None,
        "env_vars": {}
    }

    with open(script_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#SBATCH"):
                tokens = line[7:].strip().split()
                for tok in tokens:
                    if tok.startswith("--nodes="):
                        config["nodes"] = int(tok.split("=")[1])
                    elif tok.startswith("--ntasks="):
                        config["ntasks"] = int(tok.split("=")[1])
                    elif tok.startswith("--ntasks-per-node="):
                        config["ntasks_per_node"] = int(tok.split("=")[1])
                    elif tok.startswith("--cpus-per-task="):
                        config["cpus_per_task"] = int(tok.split("=")[1])
                    elif tok.startswith("--mem="):
                        config["mem_per_task"] = tok.split("=")[1]
                    elif tok.startswith("--gres="):
                        config["gres"] = tok.split("=")[1]
            elif line.startswith("export "):
                import re
                match = re.match(r"export\s+(\w+)\s*=\s*(.*)", line)
                if match:
                    var, val = match.groups()
                    config["env_vars"][var] = val

    if config["ntasks_per_node"] is None and config["nodes"] > 0:
        config["ntasks_per_node"] = config["ntasks"] // config["nodes"]

    omp_threads = config["env_vars"].get("OMP_NUM_THREADS")
    if omp_threads and omp_threads.isdigit():
        config["cpus_per_task"] = int(omp_threads)

    return config

def check_hardware_compliance(config, rank, env_name):
    if rank != 0 or env_name not in ["home_station", "slurm_compute_node"]:
        print("Compliance can not be checked on login nodes. Please refer to the HPC documentation.")
        return
    if env_name == "home_station":
        print("Checking compliance on home station is not sensible")
    total_cpus_requested = config["ntasks"] * config["cpus_per_task"]

    sys_cpus = multiprocessing.cpu_count()

    if total_cpus_requested > sys_cpus:
        print(f"Total CPUs per node requested ({total_cpus_requested}) exceed available CPUs ({sys_cpus}).")

    if config["mem_per_task"]:
        try:
            import psutil
            sys_mem_mb = psutil.virtual_memory().total // (1024*1024)

            mem_str = config["mem_per_task"].lower()
            if mem_str.endswith('g'):
                total_mem_requested = int(mem_str[:-1]) * config["ntasks"] * 1024
            elif mem_str.endswith('m'):
                total_mem_requested = int(mem_str[:-1]) * config["ntasks"]
            else:
                total_mem_requested = None

            if total_mem_requested and total_mem_requested > sys_mem_mb:
                print(f"Total memory requested ({total_mem_requested} MB) exceeds available system memory ({sys_mem_mb} MB).")
        except ImportError:
            print("psutil not installed; skipping memory check.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    enable()
    print_mpi_info()
    barrier()
    print(f"Hello from rank {rank}/{size}")

    if len(sys.argv) == 2:
        if not rank:
            parsed = parse_slurm_script(sys.argv[1])
            print("\nParsed SLURM script settings:")
            for k, v in parsed.items():
                print(f"{k:>16}: {v}")
            check_hardware_compliance(parsed, rank, detect_env())

