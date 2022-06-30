from lenscarf.core import mpi
import numpy as np
import time

randNum = np.zeros(1)
if  mpi.rank == 0:
        randNum = np.random.random_sample(1)
        [mpi.send(randNum, dest=dest) for dest in range(1,mpi.size)]
        print("{}: sent a number".format(mpi.rank))

if mpi.rank >=1:
        mpi.receive(randNum, source=0)
        print("Process", mpi.rank, "received the number", randNum[0])