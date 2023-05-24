import os
from os.path import join as opj
if "SCRATCH" not in os.environ:
    os.environ["SCRATCH"] = "./SCRATCH"

import psutil
import healpy as hp
from delensalot.run import run
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Data, DLENSALOT_Qerec, DLENSALOT_Itrec, DLENSALOT_Computing, DLENSALOT_Noisemodel, DLENSALOT_Analysis
from delensalot.utils import camb_clfile

cls_len = camb_clfile(opj(os.path.dirname(__file__), 'data/cls/FFP10_wdipole_lensedCls.dat'))
cpp = camb_clfile(opj(os.path.dirname(__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'))['pp']

def map2map_del(maps, lmax_cmb, beam, itmax, noise, verbose=False):
    """Calculates a delensed B map on the full sky. Configuration is a faithful default. 

    Args:
        maps (array-like): input maps. If 2d, assumes 'p_p' estimator for the reconstruction, otherwise mv. TODO: implement 2d/3d checks
        lmax_cmb (int): delensed B map will be build up to this value.
        beam (float): beam (transfer functions) [arcmin] of the maps.
        itmax (int): number of iterations for the iterative reconstruction.
        noise (float): noise level [muK arcmin] of the maps (noise in map should be white and isotropic). 
        verbose (bool, optional): print log.info messages. Defaults to False.

    Returns:
        np.array: delensed B map
    """    
    dlensalot_model = DLENSALOT_Model(
        defaults_to = 'P_FS_CMBS4',
        data = DLENSALOT_Data(maps=maps),
        analysis = DLENSALOT_Analysis(
            beam = beam,
            lm_max_ivf = (lmax_cmb,lmax_cmb)
            # lm_max_blt=(lmax_blt,lmax_blt), lmin_teb=(2,2,lmax_blt)
        ),
        # qerec = DLENSALOT_Qerec(),
        itrec = DLENSALOT_Itrec(itmax=itmax, lm_max_unl=(lmax_cmb+200,lmax_cmb+200)),
        computing = DLENSALOT_Computing(OMP_NUM_THREADS=min([psutil.cpu_count()-1,8])),
        noisemodel = DLENSALOT_Noisemodel(nlev_p=noise))
    delensalot_runner = run(config_fn='', job_id='MAP_lensrec', config_model=dlensalot_model, verbose=verbose)
    delensalot_runner.run()
    delensalot_runner = run(config_fn='', job_id='delens', config_model=dlensalot_model, verbose=verbose)
    delensalot_runner.run()
    ana_mwe = delensalot_runner.init_job()

    return hp.alm2map(ana_mwe.get_residualblens(ana_mwe.simidxs[0], ana_mwe.its[-1]), nside=2048)