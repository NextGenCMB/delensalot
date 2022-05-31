"""Iterative reconstruction for masked polarization CMB data

FIXME's :
    plancklens independent QEs ?
    degrade method of _wl_ filters
    check of invertibility at very first step
    mf_resp for EB-like ?
"""

import argparse
import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens

from plancklens import utils, qresp, qest, qecl
from plancklens.qcinv import cd_solve
from plancklens.sims import phas
from plancklens.filt import filt_cinv, filt_util
from plancklens.helpers import mpi

from lenscarf import remapping, utils_scarf, utils_sims
from lenscarf.iterators import cs_iterator as scarf_iterator, steps
from lenscarf.utils import cli, read_map
from lenscarf.iterators.statics import rec as Rec
from lenscarf.utils_hp import gauss_beam, almxfl, alm_copy
from lenscarf.opfilt import opfilt_ee_wl

from lenscarf.opfilt.bmodes_ninv import template_dense 


from lerepi.data.dc08 import data_08d as if_s
from lerepi.survey_config.dc08 import sc_08d as sc

def collect_jobs(self, libdir_iterators):
    jobs = []
    for idx in np.arange(self.run_config.imin, self.run_config.imax + 1):
        lib_dir_iterator = libdir_iterators(self.lensing_config.k, idx, self.run_config.v)
        if Rec.maxiterdone(lib_dir_iterator) < self.run_config.itmax:
            jobs.append(idx)
    self.jobs = jobs
    
    
def get_parser_paramfile():

    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')

    return parser.parse_args()

parser = get_parser_paramfile()
fg = '00'
mask_suffix = 2
isOBD = True
if parser.v == 'noMF':
    nsims_mf = 0
else:
    nsims_mf = 10
suffix = '08d_%s_r%s'%(fg,mask_suffix,)+'_isOBD'*isOBD
if nsims_mf > 0:
     suffix += '_MF%s'%(nsims_mf)
TEMP =  opj(os.environ['SCRATCH'], 'cmbs4', suffix)

lmax_ivf, mmax_ivf, beam, nlev_t, nlev_p = (3000, 3000, 2.3, 0.59/np.sqrt(2), 0.59)
lmin_tlm, lmin_elm, lmin_blm = (30, 30, 200) 
lmax_qlm, mmax_qlm = (4000, 4000)
lmax_unl, mmax_unl = (4000, 4000)

nside = 2048
zbounds = sc.get_zbounds()
ninvjob_geometry = utils_scarf.Geom.get_healpix_geometry(nside, zbounds=zbounds)

zbounds_len = sc.extend_zbounds(zbounds) # Outside of these bounds the reconstructed maps are assumed to be zero
pb_ctr, pb_extent = (0., 2 * np.pi) # Longitude cuts, if any, in the form (center of patch, patch extent)
lenjob_geometry = utils_scarf.Geom.get_thingauss_geometry(lmax_unl, 2, zbounds=zbounds_len)
lenjob_pbgeometry = utils_scarf.pbdGeometry(lenjob_geometry, utils_scarf.pbounds(pb_ctr, pb_extent))
lensres = 1.7  # Deflection operations will be performed at this resolution
Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted
# stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5) # handler of the size steps in the MAP BFGS iterative search
stepper = steps.harmonicbump(lmax_qlm, mmax_qlm, xa=400, xb=1500) #reduce the gradient by 0.5 for large scale and by 0.1 for small scales to improve convergence in regimes where the deflection field is not invertible

mc_sims_mf_it0 = np.arange(nsims_mf)
# mc_sims_mf_it0 = np.arange(320) # sims to use to build the very first iteration mean-field (QE mean-field)

# Multigrid chain descriptor
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
#chain_descrs = lambda lmax_sol, cg_tol :  \
#            [[2, ["split(dense(" + opj(TEMP, 'cinv_p', 'dense.pk') + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,cd_solve.cache_mem()],
#             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
#             [0, ["split(stage(1), 1024, diag_cl)"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
#------------------

cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))

# Fiducial model of the transfer function
transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)

# Isotropic approximation to the filtering (used eg for response calculations)
ftl =  cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel =  cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl =  cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)

# Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
ftl_unl =  cli(cls_unl['tt'][:lmax_ivf + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf_tlm ** 2)) * (transf_tlm > 0)
fel_unl =  cli(cls_unl['ee'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_elm ** 2)) * (transf_elm > 0)
fbl_unl =  cli(cls_unl['bb'][:lmax_ivf + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf_blm ** 2)) * (transf_blm > 0)


pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%nside), 3, (hp.nside2npix(nside),)) # T, Q, and U noise phases
#       actual data transfer function for the sim generation:
transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full FFP10 cmb's which are given to 4096)

sims = if_s.ILC_May2022(fg,mask_suffix=mask_suffix)
mask = sims.get_mask_path()
# Makes the simulation library consistent with the zbounds
sims_MAP  = utils_sims.ztrunc_sims(sims, nside, [zbounds])
# -------------------------

# Masking the poles 
masks = [mask]

# List of the inverse noise pixel variance maps, all will be multiplied together
ninv_t = [np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_t ** 2])] + masks
cinv_t = filt_cinv.cinv_t(opj(TEMP, 'cinv_t'), lmax_ivf,nside, cls_len, transf_tlm, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([hp.nside2pixarea(nside, degrees=True) * 60 ** 2 / nlev_p ** 2])] + masks]
cinv_p = filt_cinv.cinv_p(opj(TEMP, 'cinv_p'), lmax_ivf, nside, cls_len, transf_elm, ninv_p,
            chain_descr=chain_descrs(lmax_ivf, 1e-4), transf_blm=transf_blm, marge_qmaps=(), marge_umaps=())

ivfs_raw    = filt_cinv.library_cinv_sepTP(opj(TEMP, 'ivfs'), sims, cinv_t, cinv_p, cls_len)
ftl_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
fel_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_elm)
fbl_rs = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_blm)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl_rs, fel_rs, fbl_rs)

# ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)
qlms_dd = qest.library_sepTP(opj(TEMP, 'qlms_dd'), ivfs, ivfs,   cls_len['te'], nside, lmax_qlm=lmax_qlm)
qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)
# -------------------------
# This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
# This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.

# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                   np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], nside, lmax_qlm=lmax_qlm)

qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations
# -------------------------


#TODO export all these settings to a human readable file, and human readable

def get_itlib(k:str, simidx:int, version:str, cg_tol:float):
    """Return iterator instance for simulation idx and qe_key type k

        Args:
            k: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            simidx: simulation index to build iterative lensing estimate on
            version: string to use to test variants of the iterator with otherwise the same parfile
                     (here if 'noMF' is in version, will not use any mean-fied at the very first step)
            cg_tol: tolerance of conjugate-gradient filter

    """
    assert k in ['p_eb', 'p_p'], k
    libdir_iterator = libdir_iterators(k, simidx, version)
    print('Starting get itlib for {}'.format(libdir_iterator))

    if not os.path.exists(libdir_iterator):
        os.makedirs(libdir_iterator)
    tr = int(os.environ.get('OMP_NUM_THREADS', 8))
    cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
    cpp[:Lmin] *= 0.

    # QE mean-field fed in as constant piece in the iteration steps:
    mf_sims = np.unique(mc_sims_mf_it0 if not 'noMF' in version else np.array([]))
    mf0 = qlms_dd.get_sim_qlm_mf(k, mf_sims)  # Mean-field to subtract on the first iteration:
    if simidx in mf_sims:  # We dont want to include the sim we consider in the mean-field...
        Nmf = len(mf_sims)
        mf0 = (mf0 - qlms_dd.get_sim_qlm(k, int(simidx)) / Nmf) * (Nmf / (Nmf - 1))

    path_plm0 = opj(libdir_iterator, 'phi_plm_it000.npy')
    if not os.path.exists(path_plm0):
        # We now build the Wiener-filtered QE here since not done already
        plm0  = qlms_dd.get_sim_qlm(k, int(simidx))  #Unormalized quadratic estimate:
        plm0 -= mf0  # MF-subtracted unnormalized QE
        # Isotropic normalization of the QE
        R = qresp.get_response(k, lmax_ivf, 'p', cls_len, cls_len, {'e': fel, 'b': fbl, 't':ftl}, lmax_qlm=lmax_qlm)[0]
        # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
        WF = cpp * utils.cli(cpp + utils.cli(R))
        plm0 = alm_copy(plm0,  None, lmax_qlm, mmax_qlm) # Just in case the QE and MAP mmax'es were not consistent
        almxfl(plm0, utils.cli(R), mmax_qlm, True) # Normalized QE
        almxfl(plm0, WF, mmax_qlm, True)           # Wiener-filter QE
        almxfl(plm0, cpp > 0, mmax_qlm, True)
        np.save(path_plm0, plm0)

    plm0 = np.load(path_plm0)
    R_unl = qresp.get_response(k, lmax_ivf, 'p', cls_unl, cls_unl,  {'e': fel_unl, 'b': fbl_unl, 't':ftl_unl}, lmax_qlm=lmax_qlm)[0]
    if k in ['p_p'] and not 'noRespMF' in version :
        mf_resp = qresp.get_mf_resp(k, cls_unl, {'ee': fel_unl, 'bb': fbl_unl}, lmax_ivf, lmax_qlm)[0]
    else:
        print('*** mf_resp not implemented for key ' + k, ', setting it to zero')
        mf_resp = np.zeros(lmax_qlm + 1, dtype=float)
    # Lensing deflection field instance (initiated here with zero deflection)
    ffi = remapping.deflection(lenjob_pbgeometry, lensres, np.zeros_like(plm0), mmax_qlm, tr, tr)
    if k in ['p_p', 'p_eb']:
        if isOBD:
             tpl = template_dense(200, ninvjob_geometry, tr, _lib_dir=sc.BMARG_LIBDIR) # for template projection
        else:
            tpl = None # for template projection, here set to None
        wee = k == 'p_p' # keeps or not the EE-like terms in the generalized QEs
        ninv = [sims_MAP.ztruncify(read_map(ni)) for ni in ninv_p] # inverse pixel noise map on consistent geometry
        filtr = opfilt_ee_wl.alm_filter_ninv_wl(ninvjob_geometry, ninv, ffi, transf_elm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf), tr, tpl,
                                                wee=wee, lmin_dotop=min(lmin_elm, lmin_blm), transf_blm=transf_blm)
        datmaps = np.array(sims_MAP.get_sim_pmap(int(simidx)))
    else:
        assert 0
    k_geom = filtr.ffi.geom # Customizable Geometry for position-space operations in calculations of the iterated QEs etc
    # Sets to zero all L-modes below Lmin in the iterations:
    iterator = scarf_iterator.iterator_pertmf(libdir_iterator, 'p', (lmax_qlm, mmax_qlm), datmaps,
            plm0, mf_resp, R_unl, cpp, cls_unl, filtr, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
            ,mf0=mf0, wflm0=lambda : alm_copy(ivfs.get_sim_emliklm(simidx), None, lmax_unl, mmax_unl))
    return iterator


if __name__ == '__main__':
    """
    example logic 
    """
#     get_config_survey()
#     test_settings_config()
#     get_config_dlensalot()
#     test_settings_dlensalot()
#     
#     
#     get_qest()
#     other()
#     collect_jobs()
#     run()

    tol_iter   = lambda it : 10 ** (- parser.tol) # tolerance a fct of iterations ?
    soltn_cond = lambda it: True # Uses (or not) previous E-mode solution as input to search for current iteration one

    mpi.barrier = lambda : 1 # redefining the barrier (Why ? )
    jobs = collect_jobs(parser, libdir_iterators)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = libdir_iterators(parser.k, idx, parser.v)
        if parser.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < parser.itmax:
            itlib = get_itlib(parser.k, idx, parser.v, 1.)
            for i in range(parser.itmax + 1):
                # print("Rank {} with size {} is starting iteration {}".format(mpi.rank, mpi.size, i))
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))

                itlib.chain_descr  = chain_descrs(lmax_unl, tol_iter(i))
                itlib.soltn_cond   = soltn_cond(i)
                print("doing iter " + str(i))
                itlib.iterate(i, 'p')