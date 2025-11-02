import random
import sys

from delensalot.core import mpi
from delensalot.core.mpi import check_MPI


import functools


class ComputationContext:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        rank = mpi.rank
        # unique_id = random.randint(000000, 999999)
        # key = (rank, unique_id)
        # Check if running inside a Jupyter Notebook
        if "ipykernel" in sys.modules:
            rank = 0
        else:
            rank = mpi.rank  # Normal MPI rank
        if rank not in cls._instances:
            # Initialize a new context for this (rank, unique_id) pair
            instance = super().__new__(cls)
            instance.initialize(rank)
            cls._instances[rank] = instance
            print(f"ComputationContext initialized for rank {rank}")# with ID {unique_id}")
            return instance, True
        else:
            return cls._instances[rank], False  # This should rarely happen now

    def initialize(self, rank):
        self.rank = rank
        # self.unique_id = unique_id
        self.idx = None
        self.idx2 = None
        self.component = None
        self.secondary = None

    def get_context_instance(self):
        return f"Context for rank {self.rank}"#, ID {self.unique_id}"

    def reset(self):
        self.idx = None
        self.idx2 = None
        self.component = None
        self.secondary = None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Global function to get the context
def get_computation_context():
    return ComputationContext()



def preserve_context(func):
    """Decorator: stash and restore computation context around func."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assumes first argument has get_computation_context() method or import scope
        ctx, _ = get_computation_context()
        stash = (ctx.idx, ctx.idx2, ctx.component, ctx.secondary)
        try:
            return func(*args, **kwargs)
        finally:
            # restore context parameters even if func raises
            ctx.set(idx=stash[0], idx2=stash[1],
                    component=stash[2], secondary=stash[3])
    return wrapper