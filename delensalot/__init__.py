from pathlib import Path
import os
from os.path import join as opj

os.environ['USE_PLANCKLENS_MPI'] = ""

if 'SCRATCH' not in os.environ:
    os.environ['SCRATCH'] = opj(Path(__file__).resolve().parent.parent, 'delensalot_temp')
