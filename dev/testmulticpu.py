from multiprocessing import Pool, cpu_count
import math
import psutil
import os

def f(i):
    return math.sqrt(i)

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    # p.nice(0)

if __name__ == '__main__':
    # start "number of cores" processes
    # pool = Pool(None, limit_cpu)
    print(max(cpu_count()//1,1))
    pool = Pool(max(cpu_count()//1, 1))
    for p in pool.imap(f, range(10**6)):
        pass
    