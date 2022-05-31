def iohelper(func):
    def func_wrapper(libdir, mpi):
        if not os.path.exists(libdir) and mpi.rank == 0:
            os.makedirs(libdir)
        return func(libdir, mpi)
    return func_wrapper