from dlensalot import cachers, remapping
import numpy as np
from dlensalot import utils_scarf as sj
import healpy as hp
from dlensalot import utils_config
from plancklens.utils import camb_clfile


# PBOUNDS = (np.pi, 2* np.pi)
#j = sj.scarfjob()
#j.set_thingauss_geometry(3999, 2, zbounds=(0.9, 1.))
j, PBOUNDS = utils_config.cmbs4_08b_healpix()
print(PBOUNDS, np.min(j.geom.cth), np.max(j.geom.cth))


#lib_dir:str, h:str, lm_max_dlm:tuple, lm_max_elm:tuple,
#                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
#                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_ee_wl.alm_filter_ninv_wl,
# chain_descr