"""
Scripts to compute the gradient MF from the MAP lensing field. 

Using or not the trick from Carron and Lewis 2017.

"""
import numpy as np
from delensalot.utility import utils_hp as uhp
from delensalot.utils import clhash
from delensalot.core import cachers
from plancklens.qcinv import multigrid
from plancklens.sims import phas
from delensalot.core.iterator import cs_iterator

import healpy as hp
from os.path import join as opj
import time

from delensalot.utility.utils_hp import almxfl as hp_almxfl

from lenspyx.utils_hp import alm_copy


def get_graddet_sim_mf_trick(itlib, itr, mcs, key='p', mf_phas:phas.lib_phas=None, zerolensing=False):   
    """Buid the gradient MF using the trick of Carron and Lewis 2017 Appendix B
    
        Args:
            itlib: iterator instance to compute the ds ans ss for
            itr: iteration index of MAP phi
            mcs: sim indices
            key: QE key
            mf_phase: phases of the alm for the simulations
            zerolensing: Set the lensing field to zero in the gradient
    """
    #FIXME: This script can easily be paralelized with MPI in the for loops

    fn_lik = lambda this_idx : f'{itlib.h}lm_grad{key.lower()}det_it{itr:03d}_sim{this_idx:04d}' + '_nolensing'*zerolensing
            
    cacher = itlib.cacher
    
    mf_key=1 # Uses the trikc of Carron and Lewis 2017

    # Setting up fitlering instance:
    dlm = itlib.get_hlm(itr - 1, key)
    if zerolensing:
        dlm *= 0 
    itlib.hlm2dlm(dlm, True)
    ffi = itlib.filter.ffi.change_dlm([dlm, None], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
    itlib.filter.set_ffi(ffi)
    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, itlib.filter)
    q_geom = itlib.filter.ffi.pbgeom

    _Gmfs = []
    for idx in np.unique(mcs):
        if not cacher.is_cached(fn_lik(idx)):
            print(f'Doing MF sim {idx}' + ' no lensing'*zerolensing)
            if mf_phas is not None:
                phas = mf_phas.get_sim(idx, idf=0)
                phas = alm_copy(phas, None, itlib.filter.lmax_len, itlib.filter.mmax_len) 
            else:
                phas = None

            t0 = time.time()
            G, C = itlib.filter.get_qlms_mf(mf_key, q_geom, mchain, cls_filt=itlib.cls_filt, phas=phas)
            hp_almxfl(G if key.lower() == 'p' else C, itlib._h2p(itlib.lmax_qlm), itlib.mmax_qlm, True)
            print('get_qlm_mf calculation done; (%.0f secs)' % (time.time() - t0))

            itlib.cacher.cache(fn_lik(idx), -G if key.lower() == 'p' else -C)
        
        _Gmfs.append(itlib.cacher.load(fn_lik(idx)))
    return _Gmfs



def get_graddet_sim_mf_true(qe_key:str, itr:int, mcs:np.ndarray, itlib:cs_iterator.qlm_iterator, 
                            itlib_phases:cs_iterator.qlm_iterator, noise_phase:phas.lib_phas, 
                            assert_phases_exist=False, zerolensing=False):
    """Builds grad MF from averaging sims with lensing field equal to MAP field

        Args:
            qe_key: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            itr: iteration index of MAP phi
            mcs: sim indices
            itlib: iterator instance to compute the gradient for
            itlib_phases: iterator instance that generates the unlensed CMB phases for the sims
            noise_phase: phase for the noise of the CMB
            assert_phases_exist: set this if you expect the phases to be already calculatex
            zerolensing: Set the lensing field to zero in the sims and in the gradient 
    """
    #FIXME: This script can easily be paralelized with MPI in the for loops
    
    assert qe_key == 'ptt', 'Phases not implemented for pol'
    assert hasattr(itlib.filter, 'synalm')
    
    # Setting up fitlering instance:
    dlm = itlib.get_hlm(itr - 1, 'p')
    
    if zerolensing:
        dlm *= 0 
    
    itlib.hlm2dlm(dlm, True)
    ffi = itlib.filter.ffi.change_dlm([dlm, None], itlib.mmax_qlm, cachers.cacher_mem())
    itlib.filter.set_ffi(ffi)
    chain_descr = itlib.chain_descr
    mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
 
    ivf_cacher = cachers.cacher_npy(opj(itlib.lib_dir, f'mf_sims_itr{itr:03d}'))
    ivf_phas_cacher = cachers.cacher_npy(opj(itlib_phases.lib_dir, f'mf_sims_itr{itr:03d}'))
    print(ivf_cacher.lib_dir)
    print(ivf_phas_cacher.lib_dir)
    
    fn_wf = lambda this_idx : 'dat_wf_filtersim_%04d'%this_idx + '_nolensing'*zerolensing # Wiener-filtered sim
    fn = lambda this_idx : 'dat_filtersim_%04d'%this_idx + '_nolensing'*zerolensing # full sims
    fn_unl = lambda this_idx : 'unllm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile
    fn_qlm = lambda this_idx : 'qlm_mf_sim_%04d'%this_idx + '_nolensing'*zerolensing # qlms sim 
    
    def _sim_unl(itlib, lmax_sol, mmax_sol):
        if qe_key == 'p_p':
            assert np.all(itlib_phases.cls_filt['ee'][:lmax_sol+1] == itlib.cls_filt['ee'][:lmax_sol+1]), 'inconsistent inputs'
            return uhp.synalm(itlib_phases.cls_filt['ee'][:lmax_sol+1], lmax_sol, mmax_sol)
        elif qe_key == 'ptt':
            assert np.all(itlib_phases.cls_filt['tt'][:lmax_sol+1] == itlib.cls_filt['tt'][:lmax_sol+1]), 'inconsistent inputs'
            return uhp.synalm(itlib_phases.cls_filt['tt'][:lmax_sol+1], lmax_sol, mmax_sol)
        elif qe_key == 'p':
            #FIXME should generate correlated TE
            return [uhp.synalm(itlib_phases.cls_filt[cl][:lmax_sol+1], lmax_sol, mmax_sol) for cl in ['tt', 'ee']]
        
    for i in np.unique(mcs):
        idx = int(i)
        if not ivf_cacher.is_cached(fn_wf(idx)) or not ivf_cacher.is_cached(fn(idx)):
            print(f'MF grad getting WF sim {idx}')
            
            # Generate unlensed CMB
            if not ivf_phas_cacher.is_cached(fn_unl(idx)):
                assert (not assert_phases_exist)
                lmax_sol, mmax_sol = itlib_phases.filter.lmax_sol, itlib_phases.filter.mmax_sol
                assert (lmax_sol, mmax_sol) == (itlib.filter.lmax_sol, itlib.filter.mmax_sol), 'inconsistent inputs'
                xlm_unl = _sim_unl(itlib, lmax_sol, mmax_sol)
                ivf_phas_cacher.cache(fn_unl(idx), xlm_unl)
            xlm_unl = ivf_phas_cacher.load(fn_unl(idx))
            
            # FIXME: get two fields for the EB case
            nltt = (itlib.filter.nlev_tlm / 180 / 60 * np.pi) ** 2 * (itlib.filter.transf > 0)
            noise_tlm = hp.almxfl(noise_phase.get_sim(idx, idf=0), nltt)
                                  
            # Generate CMB lensed by phi MAP, with fiducial beam and noise level 
            xlm_dat = itlib.filter.synalm(itlib.cls_filt, cmb_phas=xlm_unl, noise_phase=noise_tlm)
            ivf_cacher.cache(fn(idx), xlm_dat)
            # Get the WF CMB map
            soltn = np.zeros(uhp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, ivf_cacher.load(fn(idx)), dot_op=itlib.filter.dot_op())
            ivf_cacher.cache(fn_wf(idx), soltn)

    if qe_key == 'p_p':
        get_qlms = itlib.filter.get_qlms_old
    elif qe_key == 'ptt':
        get_qlms = itlib.filter.get_qlms

    q_geom = itlib.filter.ffi.pbgeom  
    #FIXME Seems like it is different from pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    # but should be the same, maybe only the memory adress is different so python see it as different in self.filtr._get_gpmap

    # Get the QEs gradients
    _qlms = []
    for idx in np.unique(mcs):
        if not ivf_cacher.is_cached(fn_qlm(idx)):
            wf_i = ivf_cacher.load(fn_wf(idx))
            qlm = get_qlms(ivf_cacher.load(fn(idx)), wf_i, q_geom)[0]
            ivf_cacher.cache(fn_qlm(idx), qlm)
        _qlms.append(ivf_cacher.load(fn_qlm(idx)))
    return np.array(_qlms)