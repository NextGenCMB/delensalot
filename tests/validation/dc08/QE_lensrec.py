from IPython.display import clear_output
from datetime import datetime

import os
from os.path import join as opj
import numpy as np
import healpy as hp

from delensalot.utility.utils_hp import almxfl, alm_copy
import matplotlib.pyplot as plt

from delensalot.run import run

from delensalot.utility import plot_helper as ph

bpl, bpu = (10,1000)
def scale_and_bandpass(data):
    ll = np.arange(0,3001)
    return ph.bandpass_alms(hp.almxfl(data,np.sqrt(ll*(ll+1))), bpl, bpu)

fn = opj(os.getcwd(), 'conf_checkdc08_noOBD.py')
fn = opj(os.getcwd(), 'conf_checkdc08_oldtniti.py')
fn = opj(os.getcwd(), 'conf_checkdc08_oldtniti_QEtol4.py')

delensalot_runner = run(config_fn=fn, job_id='QE_lensrec', verbose=True)
delensalot_runner.run()
ana_mwe = delensalot_runner.init_job()

clear_output(wait=True)
print("Cell finished {}".format(datetime.now().strftime("%H:%M:%S")))