from delensalot.core import mpi
from delensalot.core.mpi import check_MPI


class ComputationContext:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        rank = mpi.rank
        if rank not in cls._instances:
            # Initialize a new context for this rank
            instance = super(ComputationContext, cls).__new__(cls, *args, **kwargs)
            instance.initialize(rank)
            cls._instances[rank] = instance
            print(f"ComputationContext initialized for rank {rank}")
            return instance, True
        else:
            return cls._instances[rank], False

    def initialize(self, rank):
        self.rank = rank
        self.idx = None
        self.idx2 = None
        self.component = None
        self.secondary = None

    def get_context_info(self):
        return f"Context for rank {self.rank}"

    def reset(self):
        self.idx = None
        self.idx2 = None
        self.component = None
        self.secondary = None

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# NOTE global function to get the context
def get_computation_context():
    ctx, is_new = ComputationContext()
    return ctx, is_new