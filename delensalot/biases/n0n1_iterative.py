import numpy as np
from plancklens import qresp, nhl, utils
from delensalot.utility.utils_hp import gauss_beam
from scipy.interpolate import UnivariateSpline as spl
from lensitbiases import n1_fft
# from lenspec.utils import dls2cls, cls2dls
import copy
from plancklens.helpers import cachers

# From Julien Carron lenspec

# TODO: Generalise to TT and MV estimators ?


def cls2dls(cls):
    """Turns cls dict. into camb cl array format"""
    keys = ['tt', 'ee', 'bb', 'te']
    lmax = np.max([len(cl) for cl in cls.values()]) - 1
    dls = np.zeros((lmax + 1, 4), dtype=float)
    refac = np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float) / (2. * np.pi)
    for i, k in enumerate(keys):
        cl = cls.get(k, np.zeros(lmax + 1, dtype=float))
        sli = slice(0, min(len(cl), lmax + 1))
        dls[sli, i] = cl[sli] * refac[sli]
    cldd = np.copy(cls.get('pp', None))
    if cldd is not None:
        cldd *= np.arange(len(cldd)) ** 2 * np.arange(1, len(cldd) + 1, dtype=float) ** 2 /  (2. * np.pi)
    return dls, cldd


def dls2cls(dls):
    """Inverse operation to cls2dls"""
    assert dls.shape[1] == 4
    lmax = dls.shape[0] - 1
    cls = {}
    refac = 2. * np.pi * utils.cli( np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k] = dls[:, i] * refac
    return cls


def _dictcopy(dict_in:dict): # deepcopy of dict of arrays
    return copy.deepcopy(dict_in)

def _dicthash(dict_in:dict, lmax:int, keys=None):
    h = ''
    if keys is None:
        keys = list(dict_in.keys())
    for k in keys:
        h += utils.clhash(dict_in[k][:lmax + 1], dtype=float)
    return h
    # NB: got into trouble with default float16 ?!

class polMAPbiases:
    def __init__(self, config, fidcls_unl, itrmax=6, cacher:cachers.cacher or None = None, verbose=None, qe_key='p_p'):

        (nlev_t, nlev_p, beam, lmin, lmax_ivf, lmax_qlm) = config

        self.fidcls_unl = fidcls_unl
        self.config = config
        if type(lmin) is tuple:
            lmin_tlm, lmin_elm, lmin_blm = lmin 
        elif type(lmin) is int:
            lmin_tlm = lmin 
            lmin_elm = lmin 
            lmin_blm = lmin 
        if verbose:
            print(f'lmin_tlm:{lmin_tlm}, lmin_elm:{lmin_elm}, lmin_blm:{lmin_blm}')
        transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
        transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
        transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
        
        self.fidcls_noise = {'tt': ( (nlev_t / 180 / 60 * np.pi) * utils.cli(transf_tlm) ) ** 2,
                             'ee': ( (nlev_p / 180 / 60 * np.pi) * utils.cli(transf_elm) ) ** 2,
                             'bb': ( (nlev_p / 180 / 60 * np.pi) * utils.cli(transf_blm) ) ** 2  }

        self.qe_key = qe_key
        self.itrmax= itrmax
        if cacher is None:
            cacher = cachers.cacher_mem()
        self._cacher = cacher



    def get_n0n1(self, cls_unl_true: dict or None, cls_noise_true:dict or None, fn=None, version='', recache=False):
        """Returns n0s, n1s and true and fid responses

                Args:
                    cls_unl_true: dictionary of data map true cls_unl
                    cls_noise_true: dictionary of data map true noise cls (beam deconvolved)

                Note: N0 and N1 output are the fid normalized

         """

        (nlev_t, nlev_p, beam, lmin, lmax, lmax_qlm) = self.config
        cls_unl_fid = self.fidcls_unl
        cls_noise_fid = self.fidcls_noise

        if cls_noise_true is None: cls_noise_true = cls_noise_fid
        if cls_unl_true is None: cls_unl_true = cls_unl_fid
        if fn is None:
            fn = 'n0n1_it' + str(self.itrmax) +'_' + _dicthash(cls_noise_true, lmax, keys=['ee', 'bb']) + _dicthash(cls_unl_true, 6000, keys = ['ee', 'pp'])
            if version != '':
                fn = 'v' + version + fn
        if not self._cacher.is_cached(fn) or recache:
            fid_delcls, true_delcls = self.delcls(cls_unl_true, cls_noise_true, version=version)
            N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = cls2N0N1(self.qe_key, fid_delcls[-1], true_delcls[-1],
                                                                cls_noise_fid, cls_noise_true, lmin, lmax, lmax_qlm, doN1mat=False)
            #self._cacher.cache(fn.replace('n0n1', 'n1Ls'), n1_Ls)
            #self._cacher.cache(fn.replace('n0n1', 'n1mats'), n1mat)
            self._cacher.cache(fn, np.array([N0_biased, N1_biased_spl, r_gg_fid, r_gg_true]))
            return np.array([N0_biased, N1_biased_spl, r_gg_fid, r_gg_true])
        return self._cacher.load(fn)

    def delcls(self, cls_unl_true: dict or None, cls_noise_true:dict or None, version=''):
        """Returns fiducial and true partially delensed cls


        """
        (nlev_t, nlev_p, beam, lmin, lmax, lmax_qlm) = self.config
        cls_unl_fid = self.fidcls_unl
        cls_noise_fid = self.fidcls_noise

        if cls_noise_true is None: cls_noise_true = cls_noise_fid
        if cls_unl_true is None: cls_unl_true = cls_unl_fid
        fid_delcls, true_delcls = get_delcls(self.qe_key, self.itrmax, cls_unl_fid, cls_unl_true, cls_noise_fid,
                                                 cls_noise_true, lmin, lmax, lmax_qlm, version=version)
        return fid_delcls, true_delcls

    def wf_pred(self, clunl_true:dict or None):
        """Calculate Wiener-filter prediction

                Args:
                    clunl_true: dictionary of data map true unlensed cls

                Note:
                    This is performed by using the fiducial lensing spectrum but true EE, and fiducial noise specs

        """
        (nlev_t, nlev_p, beam, lmin, lmax, lmax_qlm) = self.config
        cls_unl_fid = self.fidcls_unl
        if (clunl_true is None) or (clunl_true is cls_unl_fid):
            cls_unl_true = cls_unl_fid
        else:
            cls_unl_true = _dictcopy(clunl_true)
            cls_unl_true['pp'] = np.copy(cls_unl_fid['pp'])
        fn = 'wfpred' + _dicthash(cls_unl_true, lmax)
        if not self._cacher.is_cached(fn):
            cls_noise_fid = self.fidcls_noise
            cls_noise_true = cls_noise_fid
            fid_delcls, true_delcls = get_delcls(self.qe_key, self.itrmax, cls_unl_fid, cls_unl_true, cls_noise_fid, cls_noise_true, lmin, lmax, lmax_qlm)
            N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = cls2N0N1(self.qe_key, fid_delcls[-1], true_delcls[-1], cls_noise_fid, cls_noise_true, lmin, lmax, lmax_qlm)
            cpp = cls_unl_fid['pp'][:lmax_qlm +1 ]
            self._cacher.cache(fn, r_gg_true * utils.cli(utils.cli(cpp) + r_gg_true))
        return self._cacher.load(fn)

def cls2N0N1(k, cls_cmb_filt, cls_cmb_dat, cls_noise_filt, cls_noise_dat, lmin_ivf, lmax, lmax_qlm, doN1mat=False):
    """"
        Returns QE N0 and N1 from input filtering and data cls

            Note: Can use this for iterative N0 and N1s using partially delensed Cls

            Note: These the N0 and N1 biases for the 'fiducial' normalized QE (biased wr.t. Rfid/Rtrue factor)

    """
    assert k == 'p_p'

    if type(lmin_ivf) is tuple:
        lmin_tlm, lmin_elm, lmin_blm = lmin_ivf
    elif type(lmin_ivf) is int:
        lmin_tlm = lmin_ivf 
        lmin_elm = lmin_ivf 
        lmin_blm = lmin_ivf
    lmin_tlm =  max(lmin_tlm, 1) 
    lmin_elm = max(lmin_elm, 1)  
    lmin_blm = max(lmin_blm, 1)  

    fals = {'ee': utils.cli(cls_cmb_filt['ee'][:lmax + 1] + cls_noise_filt['ee'][:lmax+1]),
            'bb': utils.cli(cls_cmb_filt['bb'][:lmax + 1] + cls_noise_filt['bb'][:lmax+1])}

    for key, cl in fals.items():
        if key == 'tt':
            cl[:lmin_tlm] *= 0.
        elif key == 'ee':
            cl[:lmin_elm] *= 0.
        elif key == 'bb':
            cl[:lmin_blm] *= 0.
        elif key == 'te':
            cl[:max(lmin_tlm, lmin_elm)] *= 0.

    cls_w = {q: np.copy(cls_cmb_filt[q][:lmax+1]) for q in ['ee', 'bb']}
    for key, cl in cls_w.items():
        if key == 'tt':
            cl[:lmin_tlm] *= 0.
        elif key == 'ee':
            cl[:lmin_elm] *= 0.
        elif key == 'bb':
            cl[:lmin_blm] *= 0.
        elif key == 'te':
            cl[:max(lmin_tlm, lmin_elm)] *= 0.

    cls_f = {q: np.copy(cls_cmb_dat[q]) for q in ['ee', 'bb']}
    lib = n1_fft.n1_fft(fals, cls_w, cls_f, np.copy(cls_cmb_dat['pp']), lminbox=50, lmaxbox=5000, k2l=None)
    n1_Ls = np.arange(50, (lmax_qlm // 50) * 50  + 50, 50)
    if not doN1mat:
        n1 = np.array([lib.get_n1(k, L, do_n1mat=False)  for L in n1_Ls])
        n1mat = None
    else:
        n1_, n1m_ = lib.get_n1(k, n1_Ls[0], do_n1mat=True)
        n1 = np.zeros(len(n1_Ls))
        n1mat = np.zeros( (len(n1_Ls), n1m_.size))
        n1[0] = n1_
        n1mat[0] = n1m_
        for iL, n1_L in enumerate(n1_Ls[1:]):
            n1_, n1m_ = lib.get_n1(k, n1_L, do_n1mat=True)
            n1[iL + 1] = n1_
            n1mat[iL + 1] = n1m_

    dat_cls = {'ee': cls_cmb_dat['ee'][:lmax + 1] + cls_noise_dat['ee'][:lmax+1],
               'bb': cls_cmb_dat['bb'][:lmax + 1] + cls_noise_dat['bb'][:lmax+1]}
    for key, cl in dat_cls.items():
        if key == 'tt':
            cl[:lmin_tlm] *= 0.
        elif key == 'ee':
            cl[:lmin_elm] *= 0.
        elif key == 'bb':
            cl[:lmin_blm] *= 0.
        elif key == 'te':
            cl[:max(lmin_tlm, lmin_elm)] *= 0.

    cls_ivfs_arr = utils.cls_dot([fals, dat_cls, fals])
    cls_ivfs = dict()
    for i, a in enumerate(['t', 'e', 'b']):
        for j, b in enumerate(['t', 'e', 'b'][i:]):
            if np.any(cls_ivfs_arr[i, j + i]):
                cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]
    n_gg = nhl.get_nhl(k, k, cls_w, cls_ivfs, lmax, lmax, lmax_out=lmax_qlm)[0]
    # The QE is normalized by the fiducial response:
    r_gg_fid = qresp.get_response(k, lmax, 'p', cls_w, cls_cmb_filt, fals, lmax_qlm=lmax_qlm)[0]
    if cls_cmb_dat is not cls_cmb_filt:
        r_gg_true = qresp.get_response(k, lmax, 'p', cls_w, cls_cmb_dat, fals, lmax_qlm=lmax_qlm)[0]
    else:
        r_gg_true = r_gg_fid
    N0_biased = n_gg * utils.cli(r_gg_fid ** 2)
    N1_biased_spl = spl(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_fid[n1_Ls] ** 2, k=2,s=0, ext='zeros') (np.arange(len(N0_biased)))
    N1_biased_spl *= utils.cli(np.arange(lmax_qlm + 1) ** 2  * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
    if not doN1mat:
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true
    else:
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true, (n1_Ls, n1mat)


def get_delcls(qe_key: str, itermax, cls_unl_fid: dict, cls_unl_true:dict, cls_noise_fid:dict, cls_noise_true:dict, lmin_ivf, lmax_ivf, lmax_qlm:int, version=''):
    """Iterative lensing-N0 estimate

        Calculates iteratively partially lensed spectra and lensing noise levels.
        This uses the python camb package to get the partially lensed spectra.

        This makes no assumption on response =  1 / noise hence is about twice as slow as it could be in standard cases.

        Args:
            qe_key: QE estimator key
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


    #FIXME: this is requiring the full camb python package for the lensed spectra calc.

     """
    try:
        from camb.correlations import lensed_cls
    except ImportError:
        assert 0, "could not import camb.correlations.lensed_cls"


    slic = slice(0, lmax_ivf + 1)
    if type(lmin_ivf) is tuple:
        lmin_tlm, lmin_elm, lmin_blm = lmin_ivf
    elif type(lmin_ivf) is int:
        lmin_tlm = lmin_ivf 
        lmin_elm = lmin_ivf 
        lmin_blm = lmin_ivf
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
    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_true)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = 0.
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            rho_sqd_phi[:lmax_qlm + 1] = cldd_true[:lmax_qlm + 1] * utils.cli(cldd_true[:lmax_qlm + 1] + llp2 * (N0_unbiased[:lmax_qlm + 1] + N1_unbiased[:lmax_qlm + 1]))
            # print(rho_sqd_phi[10])
            # print(rho_sqd_phi[-1])
        if 'wE' in version:
            assert qe_key in ['p_p']
            if it == 0:
                print('including imperfect knowledge of E in iterations')
            slic = slice(lmin_elm, lmax_ivf + 1)
            rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
            # rho_sqd_E[slic] = cls_unl_true['ee'][slic] * utils.cli(cls_plen_true['ee'][slic] + cls_noise_true['ee'][slic])
            rho_sqd_E[slic] = cls_len_true['ee'][slic] * utils.cli(cls_len_true['ee'][slic] + cls_noise_true['ee'][slic]) # Assuming that the difference between lensed and unlensed EE can be neglected
            # rho_sqd_E[slic] = cls_unl_true['ee'][slic] * utils.cli(cls_unl_true['ee'][slic] + cls_noise_true['ee'][slic]) # Assuming that all E modes are delensed
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

        #TODO: could replace this here with cls2N0N1
        cls_filt = cls_plen_fid
        cls_w = cls_plen_fid
        cls_f = cls_plen_true
        fal = {}
        dat_delcls = {}
        if qe_key in ['ptt', 'p']:
            fal['tt'] = cls_filt['tt'][slic] + cls_noise_fid['tt'][slic]
            dat_delcls['tt'] = cls_plen_true['tt'][slic]+ cls_noise_true['tt'][slic]
        if qe_key in ['p_p', 'p']:
            fal['ee'] = cls_filt['ee'][slic] + cls_noise_fid['ee'][slic]
            fal['bb'] = cls_filt['bb'][slic] + cls_noise_true['bb'][slic]
            dat_delcls['ee'] = cls_plen_true['ee'][slic]+ cls_noise_true['ee'][slic]
            dat_delcls['bb'] = cls_plen_true['bb'][slic] + cls_noise_true['bb'][slic]
        if qe_key in ['p']:
            fal['te'] = np.copy(cls_filt['te'][slic])
            dat_delcls['te'] = np.copy(cls_plen_true['te'][slic])

        fal = utils.cl_inverse(fal)
        for k, cl in fal.items():
            if k == 'tt':
                cl[:lmin_tlm] *= 0.
            elif k == 'ee':
                cl[:lmin_elm] *= 0.
            elif k == 'bb':
                cl[:lmin_blm] *= 0.
            elif k == 'te':
                cl[:max(lmin_tlm, lmin_elm)] *= 0.
        for k, cl in dat_delcls.items():
            if k == 'tt':
                cl[:lmin_tlm] *= 0.
            elif k == 'ee':
                cl[:lmin_elm] *= 0.
            elif k == 'bb':
                cl[:lmin_blm] *= 0.
            elif k == 'te':
                cl[:max(lmin_tlm, lmin_elm)] *= 0.
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

def get_biases_iter(qe_key:str, nlev_t:float, nlev_p:float, beam_fwhm:float, cls_unl_fid:dict, lmin_ivf, lmax_ivf, itermax, cls_unl_dat=None,
                    lmax_qlm=None, datnoise_cls:dict or None=None, unlQE=False, version=''):
    # FIXME Takes this off eventually
    """Iterative lensing-N0 estimate

        Calculates iteratively partially lensed spectra and lensing noise levels.
        This uses the python camb package to get the partially lensed spectra.

        This makes no assumption on response =  1 / noise hence is about twice as slow as it could be in standard cases.

        Args:
            qe_key: QE estimator key
            nlev_t: temperature noise level (in :math:`\mu `K-arcmin)
            nlev_p: polarisation noise level (in :math:`\mu `K-arcmin)
            beam_fwhm: Gaussian beam full width half maximum in arcmin
            cls_unl_fid(dict): unlensed CMB power spectra
            lmin_ivf: minimal CMB multipole used in the QE
            lmax_ivf: maximal CMB multipole used in the QE
            itermax: number of iterations to perform
            lmax_qlm(optional): maximum lensing multipole to consider. Defaults to :math:`2 lmax_ivf`
            ret_delcls(optional): returns the partially delensed CMB cls as well if set
            datnoise_cls(optional): feeds in custom noise spectra to the data. The nlevs and beam only apply to the filtering in this case

        Returns
            Array of shape (itermax + 1, lmax_qlm + 1) with all iterated N0s. First entry is standard N0.


        Note: This assumes the unlensed spectra are known

    #FIXME: this is requiring the full camb python package for the lensed spectra calc.

     """
    try:
        from camb.correlations import lensed_cls
    except ImportError:
        assert 0, "could not import camb.correlations.lensed_cls"


    if lmax_qlm is None:
        lmax_qlm = 2 * lmax_ivf
    lmax_qlm = min(lmax_qlm, 2 * lmax_ivf)

    if type(lmin_ivf) is tuple:
        lmin_tlm, lmin_elm, lmin_blm = lmin_ivf
    elif type(lmin_ivf) is int:
        lmin_tlm = lmin_ivf 
        lmin_elm = lmin_ivf 
        lmin_blm = lmin_ivf
    lmin_tlm =  max(lmin_tlm, 1) 
    lmin_elm = max(lmin_elm, 1)  
    lmin_blm = max(lmin_blm, 1)  

    transfi2 = utils.cli(gauss_beam(beam_fwhm / 180. / 60. * np.pi, lmax_ivf)) ** 2
    llp2 = np.arange(lmax_qlm + 1, dtype=float) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2 / (2. * np.pi)
    if datnoise_cls is None:
        datnoise_cls = dict()
        if qe_key in ['ptt', 'p']:
            datnoise_cls['tt'] = (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
        if qe_key in ['p_p', 'p']:
            datnoise_cls['ee'] = (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            datnoise_cls['bb'] = (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
    N0s_biased = []
    N0s_unbiased = []
    N1s_biased = []
    N1s_unbiased = []

    R_fid = []
    R_true = []

    delcls_fid = []
    delcls_true = []

    N0_unbiased = np.inf
    N1_unbiased = np.inf
    dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
    cls_len_fid= dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
    if cls_unl_dat is None:
        cls_unl_dat = cls_unl_fid
        cls_len_true= cls_len_fid
    else:
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        cls_len_true= dls2cls(lensed_cls(dls_unl_true, cldd_true))
    cls_plen_true = cls_len_true
    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = 0.
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            rho_sqd_phi[:lmax_qlm +1] =   cldd_true[:lmax_qlm + 1] * utils.cli(cldd_true[:lmax_qlm + 1] + llp2 * (N0_unbiased[:lmax_qlm+1] + N1_unbiased[:lmax_qlm + 1]))

        if 'wE' in version:
            assert qe_key in ['p_p']
            if it == 0:
                print('including imperfect knowledge of E in iterations')
            slic = slice(lmin_elm, lmax_ivf + 1)
            rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
            #rho_sqd_E[slic] = cls_unl_dat['ee'][slic] * utils.cli(cls_plen_true['ee'][slic] + datnoise_cls['ee'][slic])
            # rho_sqd_E[slic] = cls_len_true['ee'][slic]* utils.cli(cls_len_true['ee'][slic] + datnoise_cls['ee'][slic]) 
            rho_sqd_E[slic] = cls_plen_true['ee'][slic]* utils.cli(cls_plen_true['ee'][slic] + datnoise_cls['ee'][slic]) 
            dls_unl_fid[:, 1] *= rho_sqd_E
            dls_unl_true[:, 1] *= rho_sqd_E
            cldd_fid *= rho_sqd_phi
            cldd_true *= rho_sqd_phi

            cls_plen_fid_resolved = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true_resolved = dls2cls(lensed_cls(dls_unl_true, cldd_true))
            cls_plen_fid =  {ck: cls_len_fid[ck] - (cls_plen_fid_resolved[ck] - cls_unl_fid[ck][:len(cls_len_fid[ck])]) for ck in cls_plen_fid_resolved.keys()}
            cls_plen_true = {ck: cls_len_true[ck] -(cls_plen_true_resolved[ck] - cls_unl_dat[ck][:len(cls_len_true[ck])]) for ck in cls_plen_true_resolved.keys()}

        else:
            cldd_true *= (1. - rho_sqd_phi)  # The true residual lensing spec.
            cldd_fid *= (1. - rho_sqd_phi)  # What I think the residual lensing spec is
            cls_plen_fid  = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))

        cls_filt = cls_plen_fid if not unlQE else cls_unl_fid
        cls_w = cls_plen_fid if not unlQE else cls_unl_fid
        cls_f = cls_plen_true
        fal = {}
        dat_delcls = {}
        if qe_key in ['ptt', 'p']:
            fal['tt'] = cls_filt['tt'][:lmax_ivf + 1] + (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['tt'] = cls_plen_true['tt'][:lmax_ivf + 1] + datnoise_cls['ee']
        if qe_key in ['p_p', 'p']:
            fal['ee'] = cls_filt['ee'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            fal['bb'] = cls_filt['bb'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['ee'] = cls_plen_true['ee'][:lmax_ivf + 1] + datnoise_cls['ee']
            dat_delcls['bb'] = cls_plen_true['bb'][:lmax_ivf + 1] + datnoise_cls['bb']
        if qe_key in ['p']:
            fal['te'] = np.copy(cls_filt['te'][:lmax_ivf + 1])
            dat_delcls['te'] = np.copy(cls_plen_true['te'][:lmax_ivf + 1])
        fal = utils.cl_inverse(fal)
        for k, cl in fal.items():
            if k == 'tt':
                cl[:lmin_tlm] *= 0.
            elif k == 'ee':
                cl[:lmin_elm] *= 0.
            elif k == 'bb':
                cl[:lmin_blm] *= 0.
            elif k == 'te':
                cl[:max(lmin_tlm, lmin_elm)] *= 0.
        for k, cl in dat_delcls.items():
            if k == 'tt':
                cl[:lmin_tlm] *= 0.
            elif k == 'ee':
                cl[:lmin_elm] *= 0.
            elif k == 'bb':
                cl[:lmin_blm] *= 0.
            elif k == 'te':
                cl[:max(lmin_tlm, lmin_elm)] *= 0.
        cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
        cls_ivfs = dict()
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b'][i:]):
                if np.any(cls_ivfs_arr[i, j + i]):
                    cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]

        n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
        r_gg_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0]
        r_gg_fid = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_w, fal, lmax_qlm=lmax_qlm)[0] if cls_f is not cls_w else r_gg_true
        N0_biased = n_gg * utils.cli(r_gg_fid ** 2) # N0 of possibly biased (by Rtrue / Rfid) QE estimator
        N0_unbiased = n_gg * utils.cli(r_gg_true ** 2) # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased
        N0s_biased.append(N0_biased)
        N0s_unbiased.append(N0_unbiased)
        cls_plen_true['pp'] =  cldd_true *utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 /  (2. * np.pi))
        cls_plen_fid['pp'] =  cldd_fid *utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 /  (2. * np.pi))

        if 'wN1' in version:
            if it == 0: print('Adding n1 in iterations')
            from lensitbiases import n1_fft
            lib = n1_fft.n1_fft(fal, cls_w, cls_f, np.copy(cls_plen_true['pp']), lminbox=50, lmaxbox=5000, k2l=None)
            n1_Ls = np.arange(50, lmax_qlm+1, 50)
            if lmax_qlm not in n1_Ls:  n1_Ls = np.append(n1_Ls, lmax_qlm)
            n1 = np.array([lib.get_n1(qe_key, L, do_n1mat=False) for L in n1_Ls])
            N1_biased  = spl(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_fid[n1_Ls] ** 2, k=2, s=0, ext='zeros')(np.arange(len(N0_unbiased)))
            N1_biased *= utils.cli(np.arange(lmax_qlm + 1) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
            N1_unbiased = N1_biased * (r_gg_fid * utils.cli(r_gg_true)) ** 2
        else:
            N1_biased = np.zeros(lmax_qlm + 1, dtype=float)
            N1_unbiased = np.zeros(lmax_qlm + 1, dtype=float)

        R_fid.append(r_gg_fid)
        R_true.append(r_gg_true)

        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)

        N1s_biased.append(N1_biased)
        N1s_unbiased.append(N1_unbiased)

    return np.array(N0s_biased), np.array(N0s_unbiased), np.array(N1s_biased), np.array(N1s_unbiased), delcls_fid, delcls_true, np.array(R_fid), np.array(R_true)
