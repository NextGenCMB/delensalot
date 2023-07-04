import os, sys
from os.path import join as opj
from pathlib import Path
import hashlib, psutil, shutil
import numpy as np
import healpy as hp

if "SCRATCH" not in os.environ:
    if 'site-pack' not in os.path.dirname(__file__):
        os.environ["SCRATCH"] = opj(Path(__file__).parent.parent, "reconstruction/")
    else:
        # If delensalot is installed without dev mode, put SCRATCH to user folder.
        os.environ["SCRATCH"] = os.path.expanduser('~/reconstruction')
    if not os.path.exists(os.environ["SCRATCH"]):
        os.makedirs(os.environ["SCRATCH"])

from delensalot.run import run
from delensalot.utils import camb_clfile
from delensalot.config.metamodel.dlensalot_mm import DLENSALOT_Model, DLENSALOT_Qerec, DLENSALOT_Itrec, DLENSALOT_Computing, DLENSALOT_Noisemodel, DLENSALOT_Analysis, DLENSALOT_Mapdelensing, DLENSALOT_Simulation

cls_len = camb_clfile(opj(os.path.dirname(__file__), 'data/cls/FFP10_wdipole_lensedCls.dat'))
cpp = camb_clfile(opj(os.path.dirname(__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'))['pp']

__defaults_to = 'default_CMBS4_fullsky_polarization'

def anafast(maps, lmax_cmb, beam, itmax, nlev, use_approximateWF=False, verbose=False):
    delblm = map2delblm(maps, lmax_cmb, beam, itmax, nlev, use_approximateWF, verbose)
    return hp.alm2cl(delblm)


def map2delblm(maps, lmax_cmb, beam, itmax, nlev, use_approximateWF=False, verbose=False):
    """Calculates a delensed B map on the full sky. Configuration is a faithful default. 

    Args:
        maps (array-like): input maps. If 2d, assumes 'p_p' estimator for the reconstruction, otherwise mv.
        lmax_cmb (int): delensed B map will be build up to this value.
        beam (float): beam (transfer functions) [arcmin] of the maps.
        itmax (int): number of iterations for the iterative reconstruction.
        nlev (float): noise level [muK arcmin] of the maps (noise in map should be white and isotropic). 
        use_approximateWF (bool): If true, uses approximate Wiener-filtering in the conjugate gradient solver.
        verbose (bool, optional): print log.info messages. Defaults to False.

    Returns:
        np.array: delensed B map
    """

    assert lmax_cmb-1 < 3*hp.get_nside(maps), "lmax too large: {} < {}".format(lmax_cmb, 3*hp.get_nside(maps)-1)
    assert len(maps) < 3, 'only temperature (spin-0) and polarization (spin-2) currently supported'
    if len(maps) == 1:
        assert 'T' in nlev, "need to provide 'T'-key in nlev"
    if len(maps) == 2:
        assert 'P' in nlev, "need to provide 'P'-key in nlev"
    pm = np.round(np.sum([m[::100] for m in maps]),5)
    hlib = hashlib.sha256()
    hlib.update((str([pm,lmax_cmb,beam,nlev,use_approximateWF])).encode())
    suffix = hlib.hexdigest()[:4]
    len2TP = {1: 'temperature', 2: 'polarization', 3: 'combination'}
    len2field = {1: 'temperature', 2: 'polarization', 3: 'cross'}
    len2spin = {1: 0, 2: 2}
    approxWF2itt = {False: 'constmf', True: 'fastWF'}
    if use_approximateWF:
        Lmin = 10
    else:
        Lmin = 1
    dlensalot_model = DLENSALOT_Model(
        defaults_to = __defaults_to.replace('polarization', len2TP[len(maps)]),
        simulationdata = DLENSALOT_Simulation(
            maps = maps,
            space = 'map',
            flavour = 'obs',
            field = '{}'.format(len2field[len(maps)]),
            lmax = lmax_cmb,
            spin = len2spin[len(maps)],
            geominfo = ('healpix', {'nside': hp.get_nside(maps)})
        ),
        analysis = DLENSALOT_Analysis(
            TEMP_suffix = suffix,
            beam = beam,
            lm_max_ivf = (lmax_cmb,lmax_cmb),
            Lmin = Lmin,
        ),
        itrec = DLENSALOT_Itrec(
            itmax=itmax,
            lm_max_unl=(lmax_cmb+200,lmax_cmb+200),
            iterator_typ = approxWF2itt[use_approximateWF]
        ),
        computing = DLENSALOT_Computing(
            OMP_NUM_THREADS=min([psutil.cpu_count()-1,8])
        ),
        noisemodel = DLENSALOT_Noisemodel(
            nlev = nlev,
            geominfo = ('healpix', {'nside': hp.get_nside(maps)})
        ),
        madel = DLENSALOT_Mapdelensing(
            iterations = [itmax],
            basemap = 'obs'),
    )

    delensalot_runner = run(config_fn='', job_id='MAP_lensrec', config_model=dlensalot_model, verbose=verbose)
    delensalot_runner.run()
    delensalot_runner = run(config_fn='', job_id='delens', config_model=dlensalot_model, verbose=verbose)
    ana = delensalot_runner.init_job()

    return ana.get_residualblens(ana.simidxs[0], ana.its[-1])


def map2tempblm(maps, lmax_cmb, beam, itmax, nlev, use_approximateWF=False, verbose=False):
    """Calculates a B-lensing template on the full sky. Configuration is a faithful default. 

    Args:
        maps (array-like): input maps. If 2d, assumes 'p_p' estimator for the reconstruction, otherwise mv.
        lmax_cmb (int): delensed B map will be build up to this value.
        beam (float): beam (transfer functions) [arcmin] of the maps.
        itmax (int): number of iterations for the iterative reconstruction.
        nlev (float): noise level [muK arcmin] of the maps (noise in map should be white and isotropic). 
        use_approximateWF (bool): If true, uses approximate Wiener-filtering in the conjugate gradient solver.
        verbose (bool, optional): print log.info messages. Defaults to False.

    Returns:
        np.array: B-lensing template
    """
    assert lmax_cmb-1 < 3*hp.get_nside(maps), "lmax too large: {} < {}".format(lmax_cmb, 3*hp.get_nside(maps)-1)
    assert len(maps) < 3, 'only temperature (spin-0) and polarization (spin-2) currently supported'
    if len(maps) == 1:
        assert 'T' in nlev, "need to provide 'T'-key in nlev"
    if len(maps) == 2:
        assert 'P' in nlev, "need to provide 'P'-key in nlev"
    
    pm = np.round(np.sum([m[::100] for m in maps]),5)
    hlib = hashlib.sha256()
    hlib.update((str([pm,lmax_cmb,beam,nlev,use_approximateWF])).encode())
    suffix = hlib.hexdigest()[:4]

    len2TP = {1: 'temperature', 2: 'polarization', 3: 'combination'}
    len2field = {1: 'temperature', 2: 'polarization', 3: 'cross'}
    len2spin = {1: 0, 2: 2}
    approxWF2itt = {False: 'constmf', True: 'fastWF'}
    if use_approximateWF:
        Lmin = 10
    else:
        Lmin = 1
    dlensalot_model = DLENSALOT_Model(
        defaults_to = __defaults_to.replace('polarization', len2TP[len(maps)]),
        simulationdata = DLENSALOT_Simulation(
            maps = maps,
            space = 'map',
            flavour = 'obs',
            field = '{}'.format(len2field[len(maps)]),
            lmax = lmax_cmb,
            spin = len2spin[len(maps)],
            geominfo = ('healpix', {'nside': hp.get_nside(maps)})
        ),
        analysis = DLENSALOT_Analysis(
            TEMP_suffix = suffix,
            beam = beam,
            lm_max_ivf = (lmax_cmb,lmax_cmb),
            Lmin = Lmin,
        ),
        itrec = DLENSALOT_Itrec(
            itmax=itmax,
            lm_max_unl=(lmax_cmb+200,lmax_cmb+200),
            iterator_typ = approxWF2itt[use_approximateWF]
        ),
        computing = DLENSALOT_Computing(
            OMP_NUM_THREADS=min([psutil.cpu_count()-1,8])
        ),
        noisemodel = DLENSALOT_Noisemodel(
            nlev = nlev,
            geominfo = ('healpix', {'nside': hp.get_nside(maps)})
        ),
        madel = DLENSALOT_Mapdelensing(
            iterations = [itmax],
            basemap = 'obs'),
    )

    delensalot_runner = run(config_fn='', job_id='MAP_lensrec', config_model=dlensalot_model, verbose=verbose)
    delensalot_runner.run()
    ana_mwe = delensalot_runner.init_job()

    return ana_mwe.get_blt_it(ana_mwe.simidxs[0], ana_mwe.itmax)


def del_TEMP(path):
    if os.path.exists(path):
        shutil.rmtree(path)