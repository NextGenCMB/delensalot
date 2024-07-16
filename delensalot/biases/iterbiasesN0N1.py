# Adapted from Julien Carron lenspec script

import os
import numpy as np
import hashlib
import pickle as pk
from healpy import gauss_beam
from scipy.interpolate import UnivariateSpline as spl
from delensalot.core import cachers, mpi
from delensalot.utils import cls2dls, dls2cls
from plancklens import qresp, nhl, utils

#Uses lensitbiases to compute fast N1 
from lensitbiases import n1_fft

# Requires the full camb python package for the lensed spectra calc.
from camb.correlations import lensed_cls

def _dicthash(cl_dict:dict, lmax:int, keys:list or None=None):
    """Returns the hash key for the selected keys and maximum multipoles of the Cls in cl_dict"""
    h = hashlib.new('sha1')
    if keys is None:
        keys = list(cl_dict.keys())
    for k in keys:
        cl = cl_dict[k]
        h.update(np.copy(cl[:lmax+1].astype(float), order='C'))
    return h.hexdigest()

def _lmin_ivf(lmin_ivf:int or tuple):
    """From an int or a tuple, returns the lmin_ivf for the T, E and B fields"""
    if type(lmin_ivf) is tuple:
        lmin_tlm, lmin_elm, lmin_blm = lmin_ivf
    elif type(lmin_ivf) is int:
        lmin_tlm = lmin_ivf 
        lmin_elm = lmin_ivf 
        lmin_blm = lmin_ivf
    lmin_tlm =  max(lmin_tlm, 1) 
    lmin_elm = max(lmin_elm, 1)  
    lmin_blm = max(lmin_blm, 1)
    return lmin_tlm,lmin_elm,lmin_blm

def cls_lmin_filt(lmin_tlm:int, lmin_elm:int, lmin_blm:int, cl_dict:dict):
    """Remove the multipoles below lmin for all the Cls in cl_dict"""
    for key, cl in cl_dict.items():
        if key == 'tt':
            cl[:lmin_tlm] *= 0.
        elif key == 'ee':
            cl[:lmin_elm] *= 0.
        elif key == 'bb':
            cl[:lmin_blm] *= 0.
        elif key == 'te':
            cl[:max(lmin_tlm, lmin_elm)] *= 0.


class iterbiases:
    """"""
    def __init__(self, nlev_t:float, nlev_p:float, beam_fwhm:float, lmin_ivf:int or tuple, lmax_ivf:int, lmax_qlm:int, cls_unl_fid:dict, cls_noise_fid:dict or None=None, lib_dir:str or None=None, verbose:bool =False):
        """
        Computes the iterative N0 and N1 biases, given a set of fiducial unlensed Cls unlensed.

            Args:
                nlev_t: noise level in T (muK amin)
                nlev_p: noise level in P (muK amin)
                beam_fwhm: Beam full width at half maximum (amin)
                lmin_ivf: Minimum multipole of the CMB maps to use, if int same for T E and B, if 3-tuple for T, E and B respectively
                lmax_ivf (int): Maximum multipole of the CMB maps
                lmax_qlm: Maximum multipole to reconstruct the CMB lensing 
                cls_unl_fid: Fiducial unlensed CMB cls, with fiducial lensing power spectrum
                cls_noise_fid: Fiducial noise Cls, used for the filtering of the CMB maps
                cacher: to save teh computations (optional)
                verbose: print some stuff

        """

        self.config = (nlev_t, nlev_p, beam_fwhm, lmin_ivf, lmax_ivf, lmax_qlm) 

        self.fidcls_unl = cls_unl_fid
        lmin_tlm, lmin_elm, lmin_blm = _lmin_ivf(lmin_ivf)

        if verbose:
            print(f'lmin_tlm:{lmin_tlm}, lmin_elm:{lmin_elm}, lmin_blm:{lmin_blm}')

        if cls_noise_fid is None:
            if verbose:
                print('Filtering with gaussian beam and fiducial noise levels')
            transf_tlm   =  gauss_beam(beam_fwhm/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
            transf_elm   =  gauss_beam(beam_fwhm/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
            transf_blm   =  gauss_beam(beam_fwhm/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
            
            cls_noise_fid = {'tt': ( (nlev_t / 180 / 60 * np.pi) * utils.cli(transf_tlm) ) ** 2,
                                'ee': ( (nlev_p / 180 / 60 * np.pi) * utils.cli(transf_elm) ) ** 2,
                                'bb': ( (nlev_p / 180 / 60 * np.pi) * utils.cli(transf_blm) ) ** 2  }
        self.fidcls_noise = cls_noise_fid

        self.lmax_qlm = lmax_qlm

        if lib_dir is not None:
            self._cacher = cachers.cacher_pk(lib_dir)
            fn_hash = self._cacher._path('iterbias_hash')
            if mpi.rank == 0 and not os.path.exists(fn_hash) :
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            utils.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)
        else:
            self._cacher = cachers.cacher_mem()

    def hashdict(self):
        return {'cls_unl_fid':self.fidcls_unl, 'cls_noise_fid': self.fidcls_noise,
                'lmax_qlm':self.lmax_qlm}

    def get_n0n1(self, qe_key:str,  itrmax:int, cls_unl_true: dict or None, cls_noise_true:dict or None, fn:str or None=None, version:str = '', recache:bool = False):
        """Returns n0s, n1s and true and fid responses

            Args:
                qe_key: 'ptt', 'p_p', 'p' for Temparture, Polarization only (E+B) and joint estimators (T+E+B) respectively
                itrmax: maximum iteration of the N0 and N1 estimates, 0 will give you the QE
                cls_unl_true: dictionary of data map true cls_unl
                cls_noise_true: dictionary of data map true noise cls (beam deconvolved)
                fn: file name for caching. 
                    WARNING: If specified, the script will not check if the Cls in input are the same as the one that were used in the cached data
                    If not specified, the name of the file will contain a hash key of the Cls
                version: wN1 includes N1 in all iterations; wE includes imperfect knowledge of E in iterations
                recache: recomputes all instead of loading the cached arrays

            Returns:
                N0_biased: N0 with fiducial CMB Cls 
                N1_biased_spl: N1 with fiducial CMB Cls
                r_gg_fid: Fiducial response 
                r_gg_true: True response, with true CMB spectra

            Note: N0 and N1 output are normalized with fiducial response
        """
        assert qe_key in ['ptt', 'p_p', 'p'], "The qe_key should be in 'ptt', 'p_p' or 'p'"

        (nlev_t, nlev_p, beam, lmin_ivf, lmax_ivf, lmax_qlm) = self.config
        cls_unl_fid = self.fidcls_unl
        cls_noise_fid = self.fidcls_noise

        if cls_noise_true is None and cls_unl_true is None and fn is None:
            fn = f'n0n1_ref_{qe_key}' + '_it' + str(itrmax)

        if cls_noise_true is None: cls_noise_true = cls_noise_fid
        if cls_unl_true is None: cls_unl_true = cls_unl_fid


        if fn is None:
            fn = 'n0n1_' + str(qe_key) + '_it' + str(itrmax) + '_' + _dicthash(cls_noise_true, lmax_ivf, keys=['tt', 'ee', 'bb']) + _dicthash(cls_unl_true, 6000, keys = ['tt', 'te', 'ee', 'pp'])
            if version != '':
                fn = 'v' + version + fn
        if not self._cacher.is_cached(fn) or recache:
            fid_delcls, true_delcls = self.delcls(qe_key, itrmax, cls_unl_true, cls_noise_true, version=version)
            N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = cls2N0N1(qe_key, fid_delcls[-1], true_delcls[-1],
                                                                cls_noise_fid, cls_noise_true, lmin_ivf, lmax_ivf, lmax_qlm, doN1mat=False)
            self._cacher.cache(fn, np.array([N0_biased, N1_biased_spl, r_gg_fid, r_gg_true]))
            return np.array([N0_biased, N1_biased_spl, r_gg_fid, r_gg_true])
        return self._cacher.load(fn)

    def delcls(self, qe_key:str, itrmax:int, cls_unl_true: dict or None, cls_noise_true:dict or None, version:str = ''):
        """Returns fiducial and true partially delensed cls

        """
        (nlev_t, nlev_p, beam, lmin_ivf, lmax_ivf, lmax_qlm) = self.config
        cls_unl_fid = self.fidcls_unl
        cls_noise_fid = self.fidcls_noise

        if cls_noise_true is None: cls_noise_true = cls_noise_fid
        if cls_unl_true is None: cls_unl_true = cls_unl_fid
        fid_delcls, true_delcls = get_delcls(qe_key, itrmax, cls_unl_fid, cls_unl_true, cls_noise_fid,
                                                 cls_noise_true, lmin_ivf, lmax_ivf, lmax_qlm, version=version)
        return fid_delcls, true_delcls


def get_fals(qe_key:str, cls_cmb_filt:dict, cls_cmb_dat:dict, cls_noise_filt:dict, cls_noise_dat:dict, lmin_ivf:int or tuple, lmax_ivf:int):
    """
    Get the filtering Cls and from the fiducial CMB Cls and noise, as well as the IVF data Cls
    Returns as well the QE weights and CMB response functions 
    
    Args:
        qe_key: 'ptt', 'p_p', or 'p'
        cls_cmb_filt: Fiducial CMB Cls used for the filters
        cls_cmb_dat: Data CMB Cls to filter
        cls_noise_filt: Fiducial noise used in the filters   
        cls_noise_dat: Noise Cls 
        lmin_ivf: minimum scale(s) of the CMB maps
        lmax_ivf: maximum scale of the CMB maps
    
    Returns:
        fals: Filtering (inverse CMB + noise)  Cls 
        dat_cls: Data (CMB + noise) Cls
        cls_w: QE weights (depends only on the fiducials)
        cls_f: CMB response function (used to get the responses, depends on the data Cls)

    """
    assert qe_key in ['ptt', 'p_p', 'p'], "The qe_key should be in 'ptt', 'p_p' or 'p'"
    lmin_tlm, lmin_elm, lmin_blm = _lmin_ivf(lmin_ivf)  

    fals = {}
    dat_cls = {}

    if qe_key in ['ptt', 'p']:
        fals['tt'] = cls_cmb_filt['tt'][:lmax_ivf + 1] + cls_noise_filt['tt'][:lmax_ivf+1]
        dat_cls['tt'] = cls_cmb_dat['tt'][:lmax_ivf + 1] + cls_noise_dat['tt']
    if qe_key in ['p_p', 'p']:
        fals['ee'] = cls_cmb_filt['ee'][:lmax_ivf + 1] + cls_noise_filt['ee'][:lmax_ivf+1]
        fals['bb'] = cls_cmb_filt['bb'][:lmax_ivf + 1] + cls_noise_filt['bb'][:lmax_ivf+1]
        dat_cls['ee'] = cls_cmb_dat['ee'][:lmax_ivf + 1] + cls_noise_dat['ee']
        dat_cls['bb'] = cls_cmb_dat['bb'][:lmax_ivf + 1] + cls_noise_dat['bb']
    if qe_key in ['p']:
        fals['te'] = np.copy(cls_cmb_filt['te'][:lmax_ivf + 1])
        dat_cls['te'] = np.copy(cls_cmb_dat['te'][:lmax_ivf + 1])
    fals = utils.cl_inverse(fals)

    cls_lmin_filt(lmin_tlm, lmin_elm, lmin_blm, fals)
    cls_lmin_filt(lmin_tlm, lmin_elm, lmin_blm, dat_cls)

    cls_w = {q: np.copy(cls_cmb_filt[q][:lmax_ivf+1]) for q in ['tt', 'te', 'ee', 'bb']}
    cls_f = {q: np.copy(cls_cmb_dat[q]) for q in ['tt', 'te', 'ee', 'bb']}

    cls_lmin_filt(lmin_tlm, lmin_elm, lmin_blm, cls_w)
    cls_lmin_filt(lmin_tlm, lmin_elm, lmin_blm, cls_f)
    
    return fals, dat_cls, cls_w, cls_f


def get_delcls(qe_key: str, itermax:int, cls_unl_fid: dict, cls_unl_true:dict, cls_noise_fid:dict, cls_noise_true:dict, lmin_ivf:int or tuple, lmax_ivf:int, lmax_qlm:int, version:str = ''):
    """Iterative lensing-N0 estimate

        Calculates iteratively partially lensed spectra and lensing noise levels.
        This uses the python camb package to get the partially lensed spectra.

        This makes no assumption on response =  1 / noise hence is about twice as slow as it could be in standard cases.

        Args:
            qe_key: 'ptt', 'p_p', 'p' for Temparture, Polarization only (E+B) and joint estimators (T+E+B) respectively
            itermax: number of iterations to perform

            cls_unl_fid: fiducial unlensed spectra used in the iterator
            cls_unl_true: true unlensed spectra of the sky
            cls_noise_fid: fiducial noise spectra used in the iterator (beam deconvolved)
            cls_noise_true: true noise spectra of the sky  (beam deconvolved)
            lmin_ivf: minimal CMB multipole used in the QE
            lmax_ivf: maximal CMB multipole used in the QE
            lmax_qlm: maximum lensing multipole to consider.
            version(optional): chooses between different possible version

        Returns
            list of delensed spectra for each iteration, one for the 'fiducial' delensed spectra, the other for the actual delensed spectra

     """

    slic = slice(0, lmax_ivf + 1)

    lmin_tlm, lmin_elm, lmin_blm = _lmin_ivf(lmin_ivf)

    lmin_tlm =  max(lmin_tlm, 1) 
    lmin_elm = max(lmin_elm, 1)  
    lmin_blm = max(lmin_blm, 1)  

    llp2 = np.arange(lmax_qlm + 1, dtype=float) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2 / (2. * np.pi)
    
    delcls_fid = []
    delcls_true = []
    rho = []
    
    N0_unbiased = np.inf
    N1_unbiased = np.inf
    
    dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
    cls_len_fid = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
    if cls_unl_true is cls_unl_fid:
        cls_len_true = cls_len_fid
    else:
        dls_unl_true, cldd_true = cls2dls(cls_unl_true)
        cls_len_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))

    cls_plen_true = cls_len_true
    nls_plen_true = cls_noise_true
    nls_plen_fid = cls_noise_fid

    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_true)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = 0.
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            rho_sqd_phi[:lmax_qlm + 1] = cldd_true[:lmax_qlm + 1] * utils.cli(cldd_true[:lmax_qlm + 1] + llp2 * (N0_unbiased[:lmax_qlm + 1] + N1_unbiased[:lmax_qlm + 1]))
        if 'wE' in version:
            assert qe_key in ['p_p']
            if it == 0:
                print('including imperfect knowledge of E in iterations')
            slic = slice(lmin_elm, lmax_ivf + 1)
            rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
            rho_sqd_E[slic] = cls_len_true['ee'][slic] * utils.cli(cls_len_true['ee'][slic] + cls_noise_true['ee'][slic]) # Assuming that the difference between lensed and unlensed EE can be neglected
            dls_unl_fid[:, 1] *= rho_sqd_E
            dls_unl_true[:, 1] *= rho_sqd_E
            cldd_fid *= rho_sqd_phi
            cldd_true *= rho_sqd_phi

            cls_plen_fid_resolved = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true_resolved = dls2cls(lensed_cls(dls_unl_true, cldd_true))
            cls_plen_fid = {ck: cls_len_fid[ck] - (cls_plen_fid_resolved[ck] - cls_unl_fid[ck][:len(cls_len_fid[ck])])   for ck in cls_len_fid.keys()}
            cls_plen_true = {ck: cls_len_true[ck] - (cls_plen_true_resolved[ck] - cls_unl_true[ck][:len(cls_len_true[ck])]) for ck in  cls_len_true.keys()}

        else:
            cldd_true *= (1. - rho_sqd_phi)  # The true residual lensing spec.
            cldd_fid *= (1. - rho_sqd_phi)  # What I think the residual lensing spec is
            cls_plen_fid = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))
            
        if 'wNl' in version:
            nls_plen_fid =  dls2cls(lensed_cls(cls2dls(cls_noise_fid), cldd_fid))
            nls_plen_true =  dls2cls(lensed_cls(cls2dls(cls_noise_true), cldd_fid))

        fal, dat_delcls, cls_w, cls_f = get_fals(qe_key, cls_plen_fid, cls_plen_true, nls_plen_fid, nls_plen_true, lmin_ivf, lmax_ivf)

        cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
        cls_ivfs = dict()
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b'][i:]):
                if np.any(cls_ivfs_arr[i, j + i]):
                    cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]

        n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
        r_gg_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0]
        N0_unbiased = n_gg * utils.cli(r_gg_true ** 2)  # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased

        cls_plen_true['pp'] = cldd_true * utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 / (2. * np.pi))
        cls_plen_fid['pp'] = cldd_fid * utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 / (2. * np.pi))
        if 'wE' in version and it>0:
            # Need to convert the template of the lensing power spectrum: Cldd*rho, into the reidual lensing of the map: Cldd*(1-rho)
            cls_plen_true['pp'] =  cls_plen_true['pp'] *utils.cli( rho_sqd_phi) * (1. - rho_sqd_phi) 
            cls_plen_fid['pp'] =  cls_plen_fid['pp'] *utils.cli( rho_sqd_phi) * (1. - rho_sqd_phi) 
        elif 'wE' in version and it ==0:
            cls_plen_true['pp'] =  cls2dls(cls_unl_true)[1] * utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 /  (2. * np.pi))
            cls_plen_fid['pp'] =  cls2dls(cls_unl_fid)[1] * utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 /  (2. * np.pi))
        
        if 'wN1' in version:
            if it == 0: print('Adding n1 in iterations')
            from lensitbiases import n1_fft
            from scipy.interpolate import UnivariateSpline as spl
            lib = n1_fft.n1_fft(fal, cls_w, cls_f, np.copy(cls_plen_true['pp']), lminbox=50, lmaxbox=5000, k2l=None)
            n1_Ls = np.arange(50, lmax_qlm + 1, 50)
            if lmax_qlm not in n1_Ls:  n1_Ls = np.append(n1_Ls, lmax_qlm)
            n1 = np.array([lib.get_n1(qe_key, L, do_n1mat=False) for L in n1_Ls])
            N1_unbiased = spl(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_true[n1_Ls] ** 2, k=2, s=0, ext='zeros')(np.arange(len(N0_unbiased)))
            N1_unbiased *= utils.cli(np.arange(lmax_qlm + 1) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
        else:
            N1_unbiased = np.zeros(lmax_qlm + 1, dtype=float)

        rho.append(rho_sqd_phi)
        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)

    return delcls_fid, delcls_true



def cls2N0N1(qe_key:str, cls_cmb_filt:dict, cls_cmb_dat:dict, cls_noise_filt:dict, cls_noise_dat:dict, lmin_ivf:int, lmax_ivf:int, lmax_qlm:int, doN1mat:bool = False):
    """
        Returns QE N0 and N1 from input filtering and data cls
            Args:
                qe_key: 'ptt', 'p_p', 'p' for Temparture, Polarization only (E+B) and joint estimators (T+E+B) respectively
                cls_cmb_filt: Fiducial CMB Cls used for the filters
                cls_cmb_dat: Data CMB Cls to filter
                cls_noise_filt: Fiducial noise used in the filters   
                cls_noise_dat: Noise Cls 
                lmin_ivf: minimum scale(s) of the CMB maps
                lmax_ivf: maximum scale of the CMB maps
                lmax_qlm: maximum multipole for the output N0 and N1
                doN1mat: Returns the unnormalised N1 matrix, n^{1}_{L, L'}

            Returns:
                N0_biased: N0 with fiducial CMB Cls 
                N1_biased_spl: N1 with fiducial CMB Cls
                r_gg_fid: Fiducial response 
                r_gg_true: True response, with true CMB spectra
                n1_Ls: multipoles where the N1 matrix is evaluated
                n1_mat: N1 matrix, defined such that n^1_L = \sum_{L'} C^{\phi\phi}_{L'} n^{1}_{L, L'}

            Note: Can use this for iterative N0 and N1s using partially delensed Cls

            Note: These the N0 and N1 biases for the 'fiducial' normalized QE (biased wr.t. Rfid/Rtrue factor)

    """

    fals, dat_cls, cls_w, cls_f = get_fals(qe_key, cls_cmb_filt, cls_cmb_dat, cls_noise_filt, cls_noise_dat, lmin_ivf, lmax_ivf)

    lib = n1_fft.n1_fft(fals, cls_w, cls_f, np.copy(cls_cmb_dat['pp']), lminbox=50, lmaxbox=5000, k2l=None)
    n1_Ls = np.arange(50, (lmax_qlm // 50) * 50  + 50, 50)
    if not doN1mat:
        n1 = np.array([lib.get_n1(qe_key, L, do_n1mat=False)  for L in n1_Ls])
        n1mat = None
    else:
        n1_, n1m_ = lib.get_n1(qe_key, n1_Ls[0], do_n1mat=True)
        n1 = np.zeros(len(n1_Ls))
        n1mat = np.zeros( (len(n1_Ls), n1m_.size))
        n1[0] = n1_
        n1mat[0] = n1m_
        for iL, n1_L in enumerate(n1_Ls[1:]):
            n1_, n1m_ = lib.get_n1(qe_key, n1_L, do_n1mat=True)
            n1[iL + 1] = n1_
            n1mat[iL + 1] = n1m_

    cls_ivfs_arr = utils.cls_dot([fals, dat_cls, fals])
    cls_ivfs = dict()
    for i, a in enumerate(['t', 'e', 'b']):
        for j, b in enumerate(['t', 'e', 'b'][i:]):
            if np.any(cls_ivfs_arr[i, j + i]):
                cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]
    n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
    # The QE is normalized by the fiducial response:
    r_gg_fid = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_cmb_filt, fals, lmax_qlm=lmax_qlm)[0]
    if cls_cmb_dat is not cls_cmb_filt:
        r_gg_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_cmb_dat, fals, lmax_qlm=lmax_qlm)[0]
    else:
        r_gg_true = r_gg_fid
    N0_biased = n_gg * utils.cli(r_gg_fid ** 2)
    N1_biased_spl = spl(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_fid[n1_Ls] ** 2, k=2,s=0, ext='zeros') (np.arange(len(N0_biased)))
    N1_biased_spl *= utils.cli(np.arange(lmax_qlm + 1) ** 2  * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
    if not doN1mat:
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true
    else:
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true, (n1_Ls, n1mat)
