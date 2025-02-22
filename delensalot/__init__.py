from pathlib import Path
import os
from os.path import join as opj

if 'SCRATCH' not in os.environ:
    os.environ['SCRATCH'] = opj(Path(__file__).resolve().parent.parent, 'reconstruction')