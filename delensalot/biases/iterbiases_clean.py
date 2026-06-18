"""
Iterative N0 and N1 bias computation for CMB lensing reconstruction.

This module computes iterative lensing noise biases (N0, N1) for quadratic estimators,
accounting for partial delensing at each iteration.

Adapted from Julien Carron's lenspec script by Louis Legrand
"""

import os
import hashlib
import pickle as pk
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from healpy import gauss_beam

import camb
from camb.correlations import lensed_cls
import plancklens
from plancklens import qresp, nhl, utils
from lensitbiases import n1_fft

from delensalot.core import cachers
from delensalot.utils import cls2dls, dls2cls, dls2cls_grad


# =============================================================================
# Type Aliases
# =============================================================================

ClsDict = Dict[str, np.ndarray]
LminType = Union[int, Tuple[int, int, int]]
LmaxType = Union[int, Tuple[int, int, int]]


# =============================================================================
# Utility Functions
# =============================================================================

def compute_hash(cl_dict: ClsDict, lmax: int, keys: Optional[List[str]] = None) -> str:
    """
    Compute SHA1 hash of power spectra for caching purposes.
    
    Args:
        cl_dict: Dictionary of power spectra.
        lmax: Maximum multipole to include in hash.
        keys: Specific keys to hash. If None, uses all keys.
    
    Returns:
        Hexadecimal hash string.
    """
    h = hashlib.sha1()
    if keys is None:
        keys = list(cl_dict.keys())
    for k in keys:
        h.update(np.copy(cl_dict[k][:lmax + 1].astype(float), order='C'))
    return h.hexdigest()


def parse_lmin(lmin: LminType) -> Tuple[int, int, int]:
    """
    Parse lmin parameter into separate T, E, B values.
    
    Args:
        lmin: Either a single int (same for T, E, B) or a 3-tuple.
    
    Returns:
        Tuple of (lmin_t, lmin_e, lmin_b), each at least 1.
    """
    if isinstance(lmin, tuple):
        lmin_t, lmin_e, lmin_b = lmin
    else:
        lmin_t = lmin_e = lmin_b = lmin
    return max(lmin_t, 1), max(lmin_e, 1), max(lmin_b, 1)


def parse_lmax(lmax: LmaxType) -> Tuple[int, int, int]:
    """
    Parse lmax parameter into separate T, E, B values.
    
    Args:
        lmax: Either a single int (same for T, E, B) or a 3-tuple.
    
    Returns:
        Tuple of (lmax_t, lmax_e, lmax_b).
    """
    if isinstance(lmax, tuple):
        return lmax
    return lmax, lmax, lmax


def apply_lmin_filter(cls_dict: ClsDict, lmin_t: int, lmin_e: int, lmin_b: int) -> None:
    """
    Zero out multipoles below lmin for each spectrum (in-place).
    
    Args:
        cls_dict: Dictionary of power spectra to filter.
        lmin_t, lmin_e, lmin_b: Minimum multipoles for T, E, B.
    """
    filter_map = {
        'tt': lmin_t,
        'ee': lmin_e,
        'bb': lmin_b,
        'te': max(lmin_t, lmin_e),
    }
    for key, lmin in filter_map.items():
        if key in cls_dict:
            cls_dict[key][:lmin] = 0.0


def apply_lmax_filter(cls_dict: ClsDict, lmax_t: int, lmax_e: int, lmax_b: int) -> None:
    """
    Zero out multipoles above lmax for each spectrum (in-place).
    
    Args:
        cls_dict: Dictionary of power spectra to filter.
        lmax_t, lmax_e, lmax_b: Maximum multipoles for T, E, B.
    """
    filter_map = {
        'tt': lmax_t,
        'ee': lmax_e,
        'bb': lmax_b,
        'te': max(lmax_t, lmax_e),
    }
    for key, lmax in filter_map.items():
        if key in cls_dict:
            cls_dict[key][lmax + 1:] = 0.0


def build_noise_cls(
    nlev_t: float, 
    nlev_p: float, 
    beam_fwhm: float, 
    lmax: int,
    lmin_t: int, 
    lmin_e: int, 
    lmin_b: int
) -> ClsDict:
    """
    Build noise power spectra from noise levels and beam.
    
    Args:
        nlev_t: Temperature noise level (µK·arcmin).
        nlev_p: Polarization noise level (µK·arcmin).
        beam_fwhm: Beam FWHM in arcminutes.
        lmax: Maximum multipole.
        lmin_t, lmin_e, lmin_b: Minimum multipoles for T, E, B.
    
    Returns:
        Dictionary with 'tt', 'ee', 'bb' noise spectra.
    """
    ells = np.arange(lmax + 1)
    beam_rad = beam_fwhm / 60.0 * np.pi / 180.0
    
    # Transfer functions with lmin cuts
    transf_t = gauss_beam(beam_rad, lmax=lmax) * (ells >= lmin_t)
    transf_e = gauss_beam(beam_rad, lmax=lmax) * (ells >= lmin_e)
    transf_b = gauss_beam(beam_rad, lmax=lmax) * (ells >= lmin_b)
    
    # Convert noise level to radians
    nlev_t_rad = nlev_t / 60.0 * np.pi / 180.0
    nlev_p_rad = nlev_p / 60.0 * np.pi / 180.0
    
    return {
        'tt': (nlev_t_rad * utils.cli(transf_t)) ** 2,
        'ee': (nlev_p_rad * utils.cli(transf_e)) ** 2,
        'bb': (nlev_p_rad * utils.cli(transf_b)) ** 2,
    }


def compute_lensed_cls(cls_unl: ClsDict) -> ClsDict:
    """
    Compute lensed CMB power spectra from unlensed spectra.
    
    Args:
        cls_unl: Unlensed power spectra including 'pp' for lensing.
    
    Returns:
        Lensed power spectra.
    """
    dls, cldd = cls2dls(cls_unl)
    return dls2cls(lensed_cls(dls, cldd))


def ell_prefactor(lmax: int) -> np.ndarray:
    """
    Compute L²(L+1)² / 2π prefactor for converting Clφφ to Clkk.
    
    Args:
        lmax: Maximum multipole.
    
    Returns:
        Array of prefactors.
    """
    ells = np.arange(lmax + 1, dtype=float)
    return ells ** 2 * (ells + 1) ** 2 / (2.0 * np.pi)


# =============================================================================
# Filter and Response Functions
# =============================================================================

def compute_filter_cls(
    qe_key: str,
    cls_cmb_filt: ClsDict,
    cls_cmb_dat: ClsDict,
    cls_noise_filt: ClsDict,
    cls_noise_dat: ClsDict,
    lmin_ivf: LminType,
    lmax_ivf: LmaxType,
) -> Tuple[ClsDict, ClsDict, ClsDict, ClsDict]:
    """
    Compute filtering spectra, data spectra, and QE weights.
    
    Args:
        qe_key: Estimator key ('ptt', 'p_p', or 'p').
        cls_cmb_filt: Fiducial CMB spectra for filtering.
        cls_cmb_dat: Data CMB spectra.
        cls_noise_filt: Fiducial noise spectra.
        cls_noise_dat: Data noise spectra.
        lmin_ivf: Minimum multipole(s).
        lmax_ivf: Maximum multipole(s).
    
    Returns:
        Tuple of (inverse_filter_cls, data_cls, qe_weights, response_cls).
            fals: Filtering (inverse CMB + noise)  Cls 
            dat_cls: Data (CMB + noise) Cls
            qe_weights: QE weights (depends only on the fiducials)
            response_cls: CMB response function (used to get the responses, depends on the data Cls)

    """
    assert qe_key in ['ptt', 'p_p', 'p'], f"Invalid qe_key: {qe_key}"
    
    lmin_t, lmin_e, lmin_b = parse_lmin(lmin_ivf)
    lmax_t, lmax_e, lmax_b = parse_lmax(lmax_ivf)
    lmax = max(lmax_t, lmax_e, lmax_b)
    
    # Build filter and data Cls based on estimator type
    fals = {}
    dat_cls = {}
    
    if qe_key in ['ptt', 'p']:
        fals['tt'] = cls_cmb_filt['tt'][:lmax + 1] + cls_noise_filt['tt'][:lmax + 1]
        dat_cls['tt'] = cls_cmb_dat['tt'][:lmax + 1] + cls_noise_dat['tt'][:lmax + 1]
    
    if qe_key in ['p_p', 'p']:
        fals['ee'] = cls_cmb_filt['ee'][:lmax + 1] + cls_noise_filt['ee'][:lmax + 1]
        fals['bb'] = cls_cmb_filt['bb'][:lmax + 1] + cls_noise_filt['bb'][:lmax + 1]
        dat_cls['ee'] = cls_cmb_dat['ee'][:lmax + 1] + cls_noise_dat['ee'][:lmax + 1]
        dat_cls['bb'] = cls_cmb_dat['bb'][:lmax + 1] + cls_noise_dat['bb'][:lmax + 1]
    
    if qe_key == 'p':
        fals['te'] = np.copy(cls_cmb_filt['te'][:lmax + 1])
        dat_cls['te'] = np.copy(cls_cmb_dat['te'][:lmax + 1])
    
    # Invert to get inverse-variance filter
    fals = utils.cl_inverse(fals)
    
    # Apply multipole cuts
    apply_lmin_filter(fals, lmin_t, lmin_e, lmin_b)
    apply_lmin_filter(dat_cls, lmin_t, lmin_e, lmin_b)
    apply_lmax_filter(fals, lmax_t, lmax_e, lmax_b)
    apply_lmax_filter(dat_cls, lmax_t, lmax_e, lmax_b)
    
    # QE weights and response functions
    qe_weights = {k: np.copy(cls_cmb_filt[k]) for k in ['tt', 'te', 'ee', 'bb']} # cls_w
    response_cls = {k: np.copy(cls_cmb_dat[k]) for k in ['tt', 'te', 'ee', 'bb']} # cls_f
    
    apply_lmin_filter(qe_weights, lmin_t, lmin_e, lmin_b)
    apply_lmin_filter(response_cls, lmin_t, lmin_e, lmin_b)
    apply_lmax_filter(qe_weights, lmax_t, lmax_e, lmax_b)
    apply_lmax_filter(response_cls, lmax_t, lmax_e, lmax_b)
    
    return fals, dat_cls, qe_weights, response_cls


def compute_n0(
    qe_key: str,
    cls_w: ClsDict,
    cls_ivfs: ClsDict,
    lmax: int,
    lmax_qlm: int,
) -> np.ndarray:
    """
    Compute N0 bias (Gaussian noise bias).
    
    Args:
        qe_key: Estimator key.
        cls_w: QE weight spectra.
        cls_ivfs: Inverse-variance filtered spectra.
        lmax: Maximum CMB multipole.
        lmax_qlm: Maximum lensing multipole.
    
    Returns:
        N0 array.
    """
    # The lmax is the same for T E and B here, but in the filters it is not the same  
    return nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax, lmax, lmax_out=lmax_qlm)[0]


def compute_n1(
    qe_key: str,
    fals: ClsDict,
    cls_w: ClsDict,
    cls_f: ClsDict,
    cl_pp: np.ndarray,
    lmax_qlm: int,
    response: np.ndarray,
    return_matrix: bool = False,
    lmin_box: int = 50,
    lmax_box: int = 5000,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute N1 bias using FFT method.
    
    Args:
        qe_key: Estimator key.
        fals: Inverse filter spectra.
        cls_w: QE weight spectra.
        cls_f: Response spectra.
        cl_pp: Lensing power spectrum (Clφφ).
        lmax_qlm: Maximum lensing multipole.
        response: QE response function.
        return_matrix: If True, also return the N1 matrix.
        lmin_box: Minimum multipole for FFT box.
        lmax_box: Maximum multipole for FFT box.
    Returns:
        N1 bias array, or tuple (N1, n1_Ls, n1_matrix) if return_matrix=True.
    """
    assert lmax_qlm <= lmax_box, "lmax_qlm must be <= lmax_box"
    lib = n1_fft.n1_fft(fals, cls_w, cls_f, np.copy(cl_pp), lminbox=lmin_box, lmaxbox=lmax_box)
    
    # Sample N1 at multipoles divisible by 50
    n1_Ls = np.arange(50, (lmax_qlm // 50) * 50 + 50, 50)
    
    if not return_matrix:
        n1_values = np.array([lib.get_n1(qe_key, L, do_n1mat=False) for L in n1_Ls])
        n1_matrix = None
    else:
        # Compute with matrix
        n1_values = np.zeros(len(n1_Ls))
        n1_0, n1m_0 = lib.get_n1(qe_key, n1_Ls[0], do_n1mat=True)
        n1_values[0] = n1_0
        n1_matrix = np.zeros((len(n1_Ls), n1m_0.size))
        n1_matrix[0] = n1m_0
        
        for i, L in enumerate(n1_Ls[1:], start=1):
            n1_values[i], n1_matrix[i] = lib.get_n1(qe_key, L, do_n1mat=True)
    
    # Interpolate to all multipoles
    prefactor = n1_Ls ** 2 * (n1_Ls + 1) ** 2
    n1_spline = UnivariateSpline(
        n1_Ls,
        prefactor * n1_values / response[n1_Ls] ** 2,
        k=2, s=0, ext='zeros'
    )
    
    n1_full = n1_spline(np.arange(lmax_qlm + 1))
    n1_full *= utils.cli(np.arange(lmax_qlm + 1) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
    
    if return_matrix:
        return n1_full, n1_Ls, n1_matrix
    return n1_full


def compute_response(
    qe_key: str,
    lmax: int,
    cls_w: ClsDict,
    cls_cmb: ClsDict,
    fals: ClsDict,
    lmax_qlm: int,
) -> np.ndarray:
    """
    Compute QE response function.
    
    Args:
        qe_key: Estimator key.
        lmax: Maximum CMB multipole.
        cls_w: QE weight spectra.
        cls_cmb: CMB spectra for response.
        fals: Inverse filter spectra.
        lmax_qlm: Maximum lensing multipole.
    
    Returns:
        Response function array.
    """
    return qresp.get_response(qe_key, lmax, 'p', cls_w, cls_cmb, fals, lmax_qlm=lmax_qlm)[0]


# =============================================================================
# Main N0/N1 Computation
# =============================================================================

def compute_n0_n1(
    qe_key: str,
    cls_cmb_filt: ClsDict,
    cls_cmb_dat: ClsDict,
    cls_noise_filt: ClsDict,
    cls_noise_dat: ClsDict,
    lmin_ivf: LminType,
    lmax_ivf: LmaxType,
    lmax_qlm: int,
    lmin_box: int = 50,
    lmax_box: int = 5000,
    return_n1_matrix: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute N0 and N1 biases for the quadratic estimator.
    
    Args:
        qe_key: Estimator key ('ptt', 'p_p', or 'p').
        cls_cmb_filt: Fiducial CMB spectra for filtering.
        cls_cmb_dat: Data CMB spectra.
        cls_noise_filt: Fiducial noise spectra.
        cls_noise_dat: Data noise spectra.
        lmin_ivf: Minimum multipole(s).
        lmax_ivf: Maximum multipole(s).
        lmax_qlm: Maximum lensing multipole.
        return_n1_matrix: If True, also return N1 matrix.
    
    Returns:
        Tuple of (N0, N1, response_fid, response_true).
        If return_n1_matrix=True, adds (n1_Ls, n1_matrix).
    
    Note:
        N0 and N1 are normalized by the fiducial response.
    """
    lmax = max(parse_lmax(lmax_ivf))
    
    # Get filter spectra
    fals, dat_cls, cls_w, cls_f = compute_filter_cls(
        qe_key, cls_cmb_filt, cls_cmb_dat,
        cls_noise_filt, cls_noise_dat,
        lmin_ivf, lmax_ivf
    )
    
    # Compute inverse-variance filtered spectra
    cls_ivfs_arr = utils.cls_dot([fals, dat_cls, fals])
    cls_ivfs = {}
    for i, a in enumerate(['t', 'e', 'b']):
        for j, b in enumerate(['t', 'e', 'b'][i:]):
            if np.any(cls_ivfs_arr[i, j + i]):
                cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]
    
    # Compute N0
    n_gg = compute_n0(qe_key, cls_w, cls_ivfs, lmax, lmax_qlm)
    
    # Compute responses
    r_gg_fid = compute_response(qe_key, lmax, cls_w, cls_cmb_filt, fals, lmax_qlm)
    if cls_cmb_dat is cls_cmb_filt:
        r_gg_true = r_gg_fid
    else:
        r_gg_true = compute_response(qe_key, lmax, cls_w, cls_cmb_dat, fals, lmax_qlm)
    
    # Normalize N0 by fiducial response
    N0 = n_gg * utils.cli(r_gg_fid ** 2)
    
    # Compute N1
    n1_result = compute_n1(
        qe_key, fals, cls_w, cls_f, cls_cmb_dat['pp'],
        lmax_qlm, r_gg_fid, return_matrix=return_n1_matrix, lmin_box=lmin_box, lmax_box=lmax_box
    )
    
    if return_n1_matrix:
        N1, n1_Ls, n1_matrix = n1_result
        return N0, N1, r_gg_fid, r_gg_true, n1_Ls, n1_matrix
    
    return N0, n1_result, r_gg_fid, r_gg_true


# =============================================================================
# Iterative Delensing
# =============================================================================

def compute_delensed_cls(
    qe_key: str,
    itermax: int,
    cls_unl_fid: ClsDict,
    cls_unl_true: ClsDict,
    cls_noise_fid: ClsDict,
    cls_noise_true: ClsDict,
    lmin_ivf: LminType,
    lmax_ivf: LmaxType,
    lmax_qlm: int,
    include_n1: bool = False,
    include_E_noise: bool = False,
    lmin_box: int = 50,
    lmax_box: int = 5000,
) -> Tuple[List[ClsDict], List[ClsDict]]:
    """
    Compute iteratively delensed power spectra.
    
    At each iteration, estimates the residual lensing after subtracting
    the reconstructed lensing field.
    
    Args:
        qe_key: Estimator key ('ptt', 'p_p', or 'p').
        itermax: Number of iterations (0 = QE only).
        cls_unl_fid: Fiducial unlensed spectra.
        cls_unl_true: True unlensed spectra.
        cls_noise_fid: Fiducial noise spectra.
        cls_noise_true: True noise spectra.
        lmin_ivf: Minimum multipole(s).
        lmax_ivf: Maximum multipole(s).
        lmax_qlm: Maximum lensing multipole.
        include_n1: Include N1 in iterations.
        include_E_noise: Include imperfect E-mode knowledge.
    
    Returns:
        Tuple of (fiducial_delensed_cls_list, true_delensed_cls_list).
    """
    lmin_t, lmin_e, lmin_b = parse_lmin(lmin_ivf)
    lmax_t, lmax_e, lmax_b = parse_lmax(lmax_ivf)
    lmax = max(lmax_t, lmax_e, lmax_b)
    
    llp2 = ell_prefactor(lmax_qlm)
    assert lmax_qlm >= lmax, "lmax_qlm must be >= lmax of CMB spectra, to make sure that we can delens all multipoles used in the CMB"
    # Compute fully lensed spectra
    cls_len_fid = compute_lensed_cls(cls_unl_fid)
    if cls_unl_true is cls_unl_fid:
        cls_len_true = cls_len_fid
    else:
        cls_len_true = compute_lensed_cls(cls_unl_true)
    
    # Initialize
    delcls_fid = []
    delcls_true = []
    N0_unbiased = np.inf
    N1_unbiased = np.inf
    
    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_true)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        
        # Compute delensing efficiency (cross-correlation coefficient)
        if it == 0:
            rho_sqd = 0.0
        else:
            rho_sqd = np.zeros(len(cldd_true))
            noise_total = llp2 * (N0_unbiased[:lmax_qlm + 1] + N1_unbiased[:lmax_qlm + 1])
            rho_sqd[:lmax_qlm + 1] = cldd_true[:lmax_qlm + 1] * utils.cli(
                cldd_true[:lmax_qlm + 1] + noise_total
            )
        
        if include_E_noise:
            # print("New module including imperfect E-mode knowledge in delensing.")
            assert qe_key == 'p_p', "include_E_noise only works with 'p_p'"
            cls_plen_fid, cls_plen_true = _compute_delensed_with_E_noise(
                dls_unl_fid, dls_unl_true, cldd_fid, cldd_true,
                cls_len_fid, cls_len_true, cls_unl_fid, cls_unl_true,
                cls_noise_true, rho_sqd, lmin_e, lmax_e
            )
        else:
            # Standard delensing: reduce lensing power by (1 - ρ²)
            cldd_fid_res = cldd_fid * (1.0 - rho_sqd)
            cldd_true_res = cldd_true * (1.0 - rho_sqd)
            cls_plen_fid = dls2cls(lensed_cls(dls_unl_fid, cldd_fid_res))
            cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true_res))
        
        # Compute N0 for this iteration
        fal, dat_delcls, cls_w, cls_f = compute_filter_cls(
            qe_key, cls_plen_fid, cls_plen_true,
            cls_noise_fid, cls_noise_true,
            lmin_ivf, lmax_ivf
        )
        
        cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
        cls_ivfs = {}
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b'][i:]):
                if np.any(cls_ivfs_arr[i, j + i]):
                    cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]
        
        n_gg = compute_n0(qe_key, cls_w, cls_ivfs, lmax, lmax_qlm)
        r_gg_true = compute_response(qe_key, lmax, cls_w, cls_f, fal, lmax_qlm)
        N0_unbiased = n_gg * utils.cli(r_gg_true ** 2)
        
        # Residual lensing power to output
        cldd_true_residual = cldd_true * (1.0 - rho_sqd) 
        cldd_fid_residual = cldd_fid * (1.0 - rho_sqd)
        
        cls_plen_true['pp'] = cldd_true_residual * utils.cli(ell_prefactor(len(cldd_true) - 1))
        cls_plen_fid['pp'] = cldd_fid_residual * utils.cli(ell_prefactor(len(cldd_fid) - 1))
        
        # Compute N1 if requested
        if include_n1:
            N1_unbiased = compute_n1(
                qe_key, fal, cls_w, cls_f, cls_plen_true['pp'],
                lmax_qlm, r_gg_true, lmin_box=lmin_t, lmax_box=lmax_box
            )
        else:
            N1_unbiased = np.zeros(lmax_qlm + 1)
        
        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)
    
    return delcls_fid, delcls_true


def _compute_delensed_with_E_noise(
    dls_unl_fid, dls_unl_true, cldd_fid, cldd_true,
    cls_len_fid, cls_len_true, cls_unl_fid, cls_unl_true,
    cls_noise_true, rho_sqd, lmin_e, lmax_e
):
    """Helper function for delensing with imperfect E-mode knowledge."""
    # E-mode correlation coefficient
    slic = slice(lmin_e, lmax_e + 1)
    rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
    rho_sqd_E[slic] = cls_len_true['ee'][slic] * utils.cli(
        cls_len_true['ee'][slic] + cls_noise_true['ee'][slic]
    )
    
    # Delensed spectra
    dls_unl_fid[:, 1] *= rho_sqd_E
    dls_unl_true[:, 1] *= rho_sqd_E
    cldd_fid_mod = cldd_fid * rho_sqd
    cldd_true_mod = cldd_true * rho_sqd
    
    cls_plen_fid_resolved = dls2cls(lensed_cls(dls_unl_fid, cldd_fid_mod))
    cls_plen_true_resolved = dls2cls(lensed_cls(dls_unl_true, cldd_true_mod))
    
    cls_plen_fid = {
        k: cls_len_fid[k] - (cls_plen_fid_resolved[k] - cls_unl_fid[k][:len(cls_len_fid[k])])
        for k in cls_len_fid.keys()
    }
    cls_plen_true = {
        k: cls_len_true[k] - (cls_plen_true_resolved[k] - cls_unl_true[k][:len(cls_len_true[k])])
        for k in cls_len_true.keys()
    }
    
    return cls_plen_fid, cls_plen_true


# =============================================================================
# Main Class
# =============================================================================

class IterativeBiases:
    """
    Compute iterative N0 and N1 lensing biases.
    
    This class manages the computation of noise biases for iterative
    lensing reconstruction, including caching of intermediate results.
    
    Args:
        nlev_t: Temperature noise level (µK·arcmin).
        nlev_p: Polarization noise level (µK·arcmin).
        beam_fwhm: Beam FWHM in arcminutes.
        lmin_ivf: Minimum CMB multipole (int or 3-tuple for T, E, B).
        lmax_ivf: Maximum CMB multipole (int or 3-tuple for T, E, B).
        lmax_qlm: Maximum lensing multipole.
        cls_unl_fid: Fiducial unlensed CMB spectra.
        cls_noise_fid: Fiducial noise spectra (optional).
        lib_dir: Directory for caching results (optional).
        verbose: Print progress information.
        use_grad_cls: Use gradient lensed spectra from CAMB.
    
    Example:
        >>> biases = IterativeBiases(
        ...     nlev_t=1.0, nlev_p=1.4, beam_fwhm=1.0,
        ...     lmin_ivf=100, lmax_ivf=3000, lmax_qlm=4000,
        ...     cls_unl_fid=cls_unl
        ... )
        >>> N0, N1, R_fid, R_true = biases.get_n0_n1('p_p', itrmax=3)
    """
    
    def __init__(
        self,
        nlev_t: float,
        nlev_p: float,
        beam_fwhm: float,
        lmin_ivf: LminType,
        lmax_ivf: LmaxType,
        lmax_qlm: int,
        cls_unl_fid: ClsDict,
        cls_noise_fid: Optional[ClsDict] = None,
        lib_dir: Optional[str] = None,
        verbose: bool = False,
        use_grad_cls: bool = False,
        lmin_box: int = 50,
        lmax_box: int = 5000,
    ):
        self.config = (nlev_t, nlev_p, beam_fwhm, lmin_ivf, lmax_ivf, lmax_qlm)
        self.cls_unl_fid = cls_unl_fid
        self.lmax_qlm = lmax_qlm
        self.verbose = verbose
        self.use_grad_cls = use_grad_cls
        self.lmin_box = lmin_box
        self.lmax_box = lmax_box

        # Parse multipole limits
        lmin_t, lmin_e, lmin_b = parse_lmin(lmin_ivf)
        lmax_t, lmax_e, lmax_b = parse_lmax(lmax_ivf)
        lmax = max(lmax_t, lmax_e, lmax_b)
        
        if verbose:
            print(f"lmin: T={lmin_t}, E={lmin_e}, B={lmin_b}")
            print(f"lmax: T={lmax_t}, E={lmax_e}, B={lmax_b}")
        
        # Build or use provided noise spectra
        if cls_noise_fid is None:
            if verbose:
                print("Building noise Cls from noise levels and beam")
            self.cls_noise_fid = build_noise_cls(
                nlev_t, nlev_p, beam_fwhm, lmax,
                lmin_t, lmin_e, lmin_b
            )
        else:
            self.cls_noise_fid = {
                'tt': cls_noise_fid['tt'][:lmax + 1] * (np.arange(lmax + 1) >= lmin_t),
                'ee': cls_noise_fid['ee'][:lmax + 1] * (np.arange(lmax + 1) >= lmin_e),
                'bb': cls_noise_fid['bb'][:lmax + 1] * (np.arange(lmax + 1) >= lmin_b),
            }
        
        # Set up caching
        self._setup_cacher(lib_dir)
        
        # Set up CAMB for gradient spectra if needed
        if use_grad_cls:
            self._setup_camb(verbose)
    
    def _setup_cacher(self, lib_dir: Optional[str]) -> None:
        """Initialize the caching system."""
        if lib_dir is not None:
            self._cacher = cachers.cacher_pk(lib_dir)
            hash_file = self._cacher._path('iterbias_hash')
            
            if not os.path.exists(hash_file):
                with open(hash_file, 'wb') as f:
                    pk.dump(self._get_hash_dict(), f, protocol=2)
            
            with open(hash_file, 'rb') as f:
                hash_dict = pk.load(f)
            utils.hash_check(
                self._get_hash_dict(),
                hash_dict,
                fn=hash_file
            )
        else:
            self._cacher = cachers.cacher_mem()
    
    def _setup_camb(self, verbose: bool) -> None:
        """Initialize CAMB for gradient spectrum computation."""
        cls_path = os.path.join(
            os.path.dirname(plancklens.__file__),
            'data', 'cls'
        )
        ini_file = os.path.join(cls_path, 'FFP10_wdipole_params.ini')
        
        if verbose:
            print(f"Loading CAMB parameters from: {ini_file}")
        
        pars = camb.read_ini(ini_file)
        self._camb_results = camb.get_results(pars)
    
    def _get_hash_dict(self) -> dict:
        """Get dictionary for hash verification."""
        return {
            'cls_unl_fid': self.cls_unl_fid,
            'cls_noise_fid': self.cls_noise_fid,
            'lmax_qlm': self.lmax_qlm,
        }
    
    def get_n0_n1(
        self,
        qe_key: str,
        itrmax: int,
        cls_unl_true: Optional[ClsDict] = None,
        cls_noise_true: Optional[ClsDict] = None,
        filename: Optional[str] = None,
        version: str = '',
        recache: bool = False,
    ) -> np.ndarray:
        """
        Compute iterative N0 and N1 biases.
        
        Args:
            qe_key: Estimator key ('ptt', 'p_p', or 'p').
            itrmax: Maximum iteration (0 = QE only).
            cls_unl_true: True unlensed spectra (default: fiducial).
            cls_noise_true: True noise spectra (default: fiducial).
            filename: Cache filename (auto-generated if None).
            version: Version string for different configurations.
            recache: Force recomputation.
        
        Returns:
            Array of [N0, N1, R_fid, R_true].
        """
        assert qe_key in ['ptt', 'p_p', 'p'], f"Invalid qe_key: {qe_key}"
        
        nlev_t, nlev_p, beam, lmin_ivf, lmax_ivf, lmax_qlm = self.config
        
        # Use fiducials if not specified
        if cls_unl_true is None:
            cls_unl_true = self.cls_unl_fid
        if cls_noise_true is None:
            cls_noise_true = self.cls_noise_fid
        
        # Generate cache filename
        if filename is None:
            filename = self._generate_cache_filename(
                qe_key, itrmax, cls_unl_true, cls_noise_true, version
            )
        
        # Check cache
        if self._cacher.is_cached(filename) and not recache:
            return self._cacher.load(filename)
        
        # Compute delensed spectra
        delcls_fid, delcls_true = self.get_delensed_cls(
            qe_key, itrmax, cls_unl_true, cls_noise_true, version=version
        )
        
        # Compute gradient spectra if needed
        grad_fid, grad_true = self._compute_grad_cls(delcls_fid[-1], delcls_true[-1])
        
        # Compute N0 and N1
        N0, N1, r_fid, r_true = compute_n0_n1(
            qe_key, delcls_fid[-1], delcls_true[-1],
            self.cls_noise_fid, cls_noise_true,
            lmin_ivf, lmax_ivf, lmax_qlm, 
            lmin_box=self.lmin_box, lmax_box=self.lmax_box,
        )
        
        # Cache and return
        result = np.array([N0, N1, r_fid, r_true])
        self._cacher.cache(filename, result)
        return result
    
    def get_delensed_cls(
        self,
        qe_key: str,
        itrmax: int,
        cls_unl_true: Optional[ClsDict] = None,
        cls_noise_true: Optional[ClsDict] = None,
        version: str = '',
        filename: Optional[str] = None,
        recache: bool = False,
    ) -> Tuple[List[ClsDict], List[ClsDict]]:
        """
        Compute iteratively delensed power spectra.
        
        Args:
            qe_key: Estimator key.
            itrmax: Maximum iteration.
            cls_unl_true: True unlensed spectra.
            cls_noise_true: True noise spectra.
            version: wN1 includes N1 in all iterations; wE includes imperfect knowledge of E in iterations
            filename: Cache filename.
            recache: Force recomputation.
        
        Returns:
            Tuple of (fiducial_delcls, true_delcls) lists.
        """
        _, _, _, lmin_ivf, lmax_ivf, lmax_qlm = self.config
        lmax = max(parse_lmax(lmax_ivf))
        
        if cls_unl_true is None:
            cls_unl_true = self.cls_unl_fid
        if cls_noise_true is None:
            cls_noise_true = self.cls_noise_fid
        
        # Generate filename
        if filename is None:
            hash_noise = compute_hash(cls_noise_true, lmax, ['tt', 'ee', 'bb'])
            hash_unl = compute_hash(cls_unl_true, 6000, ['tt', 'te', 'ee', 'pp'])
            filename = f"delcls_{qe_key}_it{itrmax}_{hash_noise}{hash_unl}"
            if version:
                filename = f"v{version}_{filename}"
        
        # Check cache
        fn_fid = f"{filename}_fid"
        fn_true = f"{filename}_true"
        
        if self._cacher.is_cached(fn_fid) and self._cacher.is_cached(fn_true) and not recache:
            return self._cacher.load(fn_fid), self._cacher.load(fn_true)
        
        # Compute
        include_n1 = 'wN1' in version
        include_E = 'wE' in version
        
        delcls_fid, delcls_true = compute_delensed_cls(
            qe_key, itrmax,
            self.cls_unl_fid, cls_unl_true,
            self.cls_noise_fid, cls_noise_true,
            lmin_ivf, lmax_ivf, lmax_qlm,
            include_n1=include_n1,
            include_E_noise=include_E,
            lmin_box=self.lmin_box,
            lmax_box=self.lmax_box,
        )
        
        # Cache
        self._cacher.cache(fn_fid, delcls_fid)
        self._cacher.cache(fn_true, delcls_true)
        
        return delcls_fid, delcls_true
    
    def _generate_cache_filename(
        self,
        qe_key: str,
        itrmax: int,
        cls_unl_true: ClsDict,
        cls_noise_true: ClsDict,
        version: str,
    ) -> str:
        """Generate cache filename based on configuration."""
        _, _, _, _, lmax_ivf, _ = self.config
        lmax = max(parse_lmax(lmax_ivf))
        
        # Check if using reference (fiducial) spectra
        if cls_noise_true is self.cls_noise_fid and cls_unl_true is self.cls_unl_fid:
            return f"n0n1_ref_{qe_key}_it{itrmax}"
        
        hash_noise = compute_hash(cls_noise_true, lmax, ['tt', 'ee', 'bb'])
        hash_unl = compute_hash(cls_unl_true, 6000, ['tt', 'te', 'ee', 'pp'])
        filename = f"n0n1_{qe_key}_it{itrmax}_{hash_noise}{hash_unl}"
        
        if version:
            filename = f"v{version}_{filename}"
        if self.use_grad_cls:
            filename = f"gradcls_{filename}"
        
        return filename
    
    def _compute_grad_cls(
        self,
        cls_fid: ClsDict,
        cls_true: ClsDict,
    ) -> Tuple[Optional[ClsDict], Optional[ClsDict]]:
        """Compute gradient lensed spectra if enabled."""
        if not self.use_grad_cls:
            return None, None
        
        dls_fid, cldd_fid = cls2dls(cls_fid)
        dls_true, cldd_true = cls2dls(cls_true)
        
        grad_fid = dls2cls_grad(
            self._camb_results.get_lensed_gradient_cls(CMB_unit='muK', clpp=cldd_fid)
        )
        grad_true = dls2cls_grad(
            self._camb_results.get_lensed_gradient_cls(CMB_unit='muK', clpp=cldd_true)
        )
        
        return grad_fid, grad_true
