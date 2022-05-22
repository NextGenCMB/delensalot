import os
import numpy as np
import healpy as hp
from lerepi.survey_config import sc_08d as sc

THIS_CENTRALNLEV_UKAMIN = sc.THIS_CENTRALNLEV_UKAMIN
fg = '00'
TEMP = '/global/cscratch1/sd/sebibel/cmbs4/s08d/cILC_%s_test/'%fg
nside = sc.nside

# --- masks: here we test apodized at ratio 10 and weightmap
ivmap_path = os.path.join(TEMP, 'ipvmap.fits')
if not os.path.exists(ivmap_path):
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits')) #TODO this should come from survey_conf
    pixlev = THIS_CENTRALNLEV_UKAMIN / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    print("Pmap center pixel pol noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmap_path,  1./ pixlev ** 2 * rhits)  #TODO this should be provided to app level
ivmat_path = os.path.join(TEMP, 'itvmap.fits')
if not os.path.exists(ivmat_path):
    pixlev = 0.27 * np.sqrt(2) / (np.sqrt(hp.nside2pixarea(2048, degrees=True)) * 60.)
    rhits = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/expt_xx/08d/rhits/n2048.fits')) #TODO this should come from survey_conf
    rhits = np.where(rhits > 0., rhits, 0.)  # *(~np.isnan(rhits))
    print("Pmap center pixel T noise level: %.2f"%(pixlev * np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60.))
    hp.write_map(ivmat_path,  1./ pixlev ** 2 * rhits)  #TODO this should be provided to app level