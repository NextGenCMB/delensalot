import numpy as np
from lenscarf import utils
from lenscarf.iterators import statics
import plancklens
from plancklens.sims import planck2018_sims
from plancklens import qresp
from lenscarf import cachers
from lenscarf import utils_scarf, utils_sims
from plancklens.qcinv import multigrid
from plancklens import nhl 
from scipy.interpolate import UnivariateSpline as spline

from os.path import join as opj
import os
import importlib
from lenscarf.utils_hp import alm2cl, alm_copy
from lenscarf.utils import read_map
from lenscarf.utils_plot import pp2kk
from lenscarf import rdn0_cs
from lenscarf import utils_hp as uhp
from lenscarf import n0n1_iterative
from lensitbiases import n1_fft

import healpy as hp

class cpp_sims_lib:
    def __init__(self, k, itmax, tol, v='', param_file='cmbs4wide_planckmask', label=''):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.
        """
        
        # Load the parameters defined in the param_file
        self.param_file = param_file
        self.param = importlib.import_module('lenscarf.params.'+param_file)
        self.k = k
        self.itmax = itmax
        self.tol = tol
        self.tol_iter  = 10 ** (- self.tol) 
        # self.imin = imin
        # self.imax = imax
        self.version = v
        self.iters = np.arange(itmax+1)
        self.label = label
        self.TEMP =  self.param.TEMP
        self.lmax_qlm = self.param.lmax_qlm
        self.mmax_qlm = self.param.mmax_qlm

        # self.plms = [[None]*(itmax+1)]*(imax+1)
        self.cacher_param = cachers.cacher_npy(opj(self.TEMP, 'cpplib'))
        self.fsky = self.get_fsky() 

        self.cpp_fid = self.param.cls_unl['pp']


        if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            self.sims_unl = planck2018_sims.cmb_unl_ffp10() 
    
    def libdir_sim(self, simidx):
        return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)

    def get_itlib_sim(self, simidx):
        return self.param.get_itlib(self.k, simidx, self.version, self.tol_iter)

    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)

    def get_plm(self, simidx, itr):
        # if self.plms[simidx][itr] is None:
        #     self.plms[simidx][itr] = statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
        return statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
    
    def get_plm_input(self, simidx):
        return alm_copy(self.sims_unl.get_sim_plm(simidx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)
    
    def get_eblm_dat(self, simidx, lmaxout=1024):
        QU_maps = self.param.sims_MAP.get_sim_pmap(simidx)
        tr = int(os.environ.get('OMP_NUM_THREADS', 8))
        sht_job = utils_scarf.scarfjob()
        sht_job.set_geometry(self.param.ninvjob_geometry)
        sht_job.set_triangular_alm_info(self.param.lmax_ivf,self.param.mmax_ivf)
        sht_job.set_nthreads(tr)
        elm, blm = np.array(sht_job.map2alm_spin(QU_maps, 2))
        lmaxdat = hp.sphtfunc.Alm.getlmax(elm.size)
        elm = uhp.alm_copy(elm, mmaxin=lmaxdat, lmaxout=lmaxout, mmaxout=lmaxout)
        blm = uhp.alm_copy(blm, mmaxin=lmaxdat, lmaxout=lmaxout, mmaxout=lmaxout)
        return elm, blm



    def get_fsky(self):
        try:
            fn = 'fsky'
            if not self.cacher_param.is_cached(fn):
                mask = read_map(self.param.masks)
                fsky = np.sum(mask)/np.size(mask)
                print(fsky)
                self.cacher_param.cache(fn, fsky)
            return self.cacher_param.load(fn)
        except AttributeError:
            print('No masks defined in param file ' + self.param_file)
            return 1.


    def get_cpp_input(self, simidx):
        fn_cpp_in = 'cpp_input'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_in):
            plmin = self.get_plm_input(simidx)
            cpp_in = alm2cl(plmin, plmin, self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn_cpp_in, cpp_in)
        cpp_in = cacher.load(fn_cpp_in)
        return cpp_in


    def get_cpp_itXinput(self, simidx, itr):
        fn = 'cpp_in_x_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            plmin = self.get_plm_input(simidx)
            # plmit = self.plms[simidx][itr]
            plmit = self.get_plm(simidx, itr)
            cpp_itXin = alm2cl(plmit, plmin, self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin
    


    def get_cpp(self, simidx, itr):
        fn_cpp_it = 'cpp_it_{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it):
            cpp = alm2cl(self.get_plm(simidx, itr), self.get_plm(simidx, itr), self.lmax_qlm, self.mmax_qlm, None)
            cacher.cache(fn_cpp_it, cpp)
        cpp = cacher.load(fn_cpp_it)
        return cpp
        
    # def get_wf(self):

    
    def get_cpp_qe(self, simidx):
        fn_cpp_qe = 'cpp_qe'
        fn_resp_qe = 'resp_qe'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_resp_qe):
            R = qresp.get_response(self.k, self.param.lmax_ivf, 'p', self.param.cls_len, self.param.cls_len, {'e': self.param.fel, 'b': self.param.fbl, 't':self.param.ftl}, lmax_qlm=self.param.lmax_qlm)[0]
            cacher.cache(fn_resp_qe, R)
        R = cacher.load(fn_resp_qe)
        if not cacher.is_cached(fn_cpp_qe):
            cpp = self.param.cls_unl['pp'][:self.lmax_qlm + 1]
            WF = cpp * utils.cli(cpp + utils.cli(R))
            cpp_qe_wf = self.get_cpp(simidx, 0)
            cpp_qe = cpp_qe_wf * utils.cli(WF)**2
            cacher.cache(fn_cpp_qe, cpp_qe)
        cpp_qe = cacher.load(fn_cpp_qe)
        return cpp_qe, R

    def get_mf0(self, simidx):
        return np.load(opj(self.libdir_sim(simidx), 'mf.npy'))
    

    def get_mf(self, simidx, itr, ret_alm=False, verbose=False):
        cacher = self.cacher_sim(simidx)
        fn_mf1 = 'mf1_it{}'.format(itr)
        fn_mf2 = 'mf2_it{}'.format(itr)
        if not cacher.is_cached(fn_mf1) or not cacher.is_cached(fn_mf2):
            if verbose:
                print('Starting to estimate MF for it {} from sim {}'.format(itr, simidx))
            itlib = self.get_itlib_sim(simidx)
            filtr = itlib.filter
            filtr.set_ffi(itlib._get_ffi(itr)) # load here the phi map you want 
            chain_descr = self.param.chain_descrs(self.param.lmax_unl, self.tol_iter)
            mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
            MF1 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
            MF2 = filtr.get_qlms_mf(0, filtr.ffi.pbgeom, mchain, cls_filt=itlib.cls_filt)
            cacher.cache(fn_mf1, MF1)
            cacher.cache(fn_mf2, MF2)
        MF1 = cacher.load(fn_mf1)
        MF2 = cacher.load(fn_mf2)
        fn_mf = 'cpp_mf_it{}'.format(itr)
        fn_mf_mc = 'cpp_mf_mcnoise_it{}'.format(itr)
        if not cacher.is_cached(fn_mf) or not cacher.is_cached(fn_mf_mc):
            norm = self.param.qresp.get_response('p_p', self.param.lmax_ivf, 'p', self.param.cls_unl, self.param.cls_unl,  {'e': self.param.fel_unl, 'b': self.param.fbl_unl, 't':self.param.ftl_unl}, lmax_qlm=self.param.lmax_qlm)[0]
            cpp_mf = alm2cl(MF1[0], MF2[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
            cpp_mf_mc = alm2cl(MF1[0], MF1[0], self.lmax_qlm, self.mmax_qlm, None) / norm ** 2
            cacher.cache(fn_mf, cpp_mf)
            cacher.cache(fn_mf_mc, cpp_mf_mc)
        cpp_mf = cacher.load(fn_mf)
        cpp_mf_mc = cacher.load(fn_mf_mc)
        return (cpp_mf, cpp_mf_mc) if not ret_alm else (cpp_mf, cpp_mf_mc,MF1, MF2)

        # # plot MFs:
        # ls = np.arange(2, 2048)
        # w = ls ** 2 * (ls + 1.) ** 2 * 1e7 /2./np.pi
        # norm = self.param.qresp.get_response('p_p', self.param.lmax_ivf, 'p', self.param.cls_unl, self.param.cls_unl,  {'e': self.param.fel_unl, 'b': self.param.fbl_unl, 't':self.param.ftl_unl}, lmax_qlm=self.param.lmax_qlm)[0]
        # pl.loglog(ls, w * hp.alm2cl(MF1[0], MF2[0])[ls] / norm[ls] ** 2, label='MF spec')
        # pl.loglog(ls, w * hp.alm2cl(MF1[0], MF1[0])[ls] / norm[ls] ** 2, label='MF spec + MC noise')


    def get_N0_qe(self):
        fn_n0_qe = 'N0_qe'
        
        if not self.cacher_param.is_cached(fn_n0_qe):
            cls_cmb_dat = self.param.cls_len_fid
            
            # Simple white noise model. Can feed here something more fancy if desired
            transf = hp.gauss_beam(self.param.beam / 60. / 180. * np.pi, lmax=self.param.lmax_ivf)
            Noise_L_T = (self.param.nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2
            Noise_L_P = (self.param.nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2

            # Data power spectra
            cls_dat = {
                'tt': (self.param.cls_len['tt'][:self.param.lmax_ivf + 1] + Noise_L_T),
                'ee': (self.param.cls_len['ee'][:self.param.lmax_ivf + 1] + Noise_L_P),
                'bb': (self.param.cls_len['bb'][:self.param.lmax_ivf + 1] + Noise_L_P),
                'te': np.copy(self.param.cls_len['te'][:self.param.lmax_ivf + 1])}

            for s in cls_dat.keys():
                cls_dat[s][min(lmaxs_CMB[s[0]], lmaxs_CMB[s[1]]) + 1:] *= 0.

            # (C+N)^{-1} filter spectra
            # For independent T and P filtering, this is really just 1/ (C+ N), diagonal in T, E, B space
            fal_sepTP = {spec: utils.cli(cls_dat[spec]) for spec in ['tt', 'ee', 'bb']}
            # Spectra of the inverse-variance filtered maps
            # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
            cls_ivfs_sepTP = utils.cls_dot([fal_sepTP, cls_dat, fal_sepTP], ret_dict=True)

            # For joint TP filtering, fals is matrix inverse
            # fal_jtTP = utils.cl_inverse(cls_dat)
            # since cls_dat = fals, cls_ivfs = fals. If the data spectra do not match the filter, this must be changed:
            # cls_ivfs_jtTP = utils.cls_dot([fal_jtTP, cls_dat, fal_jtTP], ret_dict=True)
            # for cls in [fal_sepTP, fal_jtTP, cls_ivfs_sepTP, cls_ivfs_jtTP]:
            #     for cl in cls.values():
            #         cl[:max(1, lmin_ivf)] *= 0.


            N0 = nhl.get_nhl(self.k, self.k, self.param.cls_len, cls_ivfs_sepTP, self.param.lmax_ivf, self.param.lmax_ivf)
        N0 = self.cacher_param.load(fn_n0_qe)
        return N0


    def get_N0_N1_QE_fid(self, doN1mat=False):
        """
        Returns the fiducial QE N0 and N1 biases.
        """
        fn_n0_qe = 'N0_qe_fid'
        fn_n1_qe = 'N1_qe_fid'
        fn_resp_qe = 'Resp_qe_fid'
        assert self.k =='p_p', 'QE biases are currently not implemented fot MV and TT estimators (check nhl lib and n1_fft lib to get more QE biases)'

        if np.any([not self.cacher_param.is_cached(fn) for fn in [fn_n0_qe, fn_n1_qe, fn_resp_qe]]):
            
            
            if type(self.param.ivfs) == plancklens.filt.filt_util.library_ftl:
                """In the masked case"""
                cls_weights = self.param.ivfs.ivfs.cl
            elif type(self.param.ivfs) == plancklens.filt.filt_simple.library_fullsky_sepTP:
                """In the full sky case"""
                cls_weights = self.param.ivfs.cl

            # cls_cmb_dat = self.param.cls_len
            fidcls_noise = {'tt': (self.param.nlev_t / 180 / 60 * np.pi) **2 * utils.cli(self.param.transf_tlm ** 2) * (self.param.transf_tlm > 0),
                            'ee': (self.param.nlev_p / 180 / 60 * np.pi) **2 * utils.cli(self.param.transf_elm ** 2) * (self.param.transf_elm > 0),
                            'bb': (self.param.nlev_p / 180 / 60 * np.pi) **2 * utils.cli(self.param.transf_blm ** 2) * (self.param.transf_blm > 0) }
            # cls_noise_dat = fidcls_noise
            lmax =  self.param.lmax_ivf
            lmax_qlm =  self.param.lmax_qlm

            fals = {'tt':self.param.ftl, 
                        'ee':self.param.fel,
                        'bb':self.param.fbl}
            cls_ivfs = fals
            #FIXME: In principle this should be the filtered CMB data, but here we assume they are the fiducial ones
            # could compute it as in n0n1_iterative.py with the fiducial Cls for the filtering but the true Cls for the data
            
            n_gg = nhl.get_nhl(self.k, self.k, cls_weights, cls_ivfs, lmax, lmax, lmax_out=lmax_qlm)[0]
            # nhllib = nhl.nhl_lib_simple(opj(self.TEMP, 'cpplib'), self.param.ivfs, cls_weights, lmax_qlm)
            
            # FIXME : Assuming here that the Cls entering the response are the lensed Cls, but more optimal is to put the gradlensed Cls
            cls_f = self.param.cls_len
            r_gg_fid = qresp.get_response(self.k, lmax, 'p', cls_weights, cls_f, fals, lmax_qlm=lmax_qlm)[0]

            N0_fid = n_gg * utils.cli(r_gg_fid ** 2)

            n1lib = n1_fft.n1_fft(fals, cls_weights, cls_f, np.copy(self.param.cls_unl['pp']), lminbox=50, lmaxbox=5000, k2l=None)
            n1_Ls = np.arange(50, (lmax_qlm // 50) * 50  + 50, 50)
            if not doN1mat:
                n1 = np.array([n1lib.get_n1(self.k, L, do_n1mat=False)  for L in n1_Ls])
                n1mat = None
            else:
                n1_, n1m_ = n1lib.get_n1(self.k, n1_Ls[0], do_n1mat=True)
                n1 = np.zeros(len(n1_Ls))
                n1mat = np.zeros( (len(n1_Ls), n1m_.size))
                n1[0] = n1_
                n1mat[0] = n1m_
                for iL, n1_L in enumerate(n1_Ls[1:]):
                    n1_, n1m_ = n1lib.get_n1(self.k, n1_L, do_n1mat=True)
                    n1[iL + 1] = n1_
                    n1mat[iL + 1] = n1m_
            N1_fid_spl = spline(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_fid[n1_Ls] ** 2, k=2,s=0, ext='zeros') (np.arange(len(N0_fid)))

            self.cacher_param.cache(fn_n0_qe, N0_fid)
            self.cacher_param.cache(fn_n1_qe, N1_fid_spl)
            self.cacher_param.cache(fn_resp_qe, r_gg_fid)
        N0_fid = self.cacher_param.load(fn_n0_qe)
        N1_fid = self.cacher_param.load(fn_n1_qe)
        Resp_fid = self.cacher_param.load(fn_resp_qe)
        return N0_fid, N1_fid, Resp_fid

    def get_N0_N1_iter(self, itermax=15, version=''):
        assert self.k =='p_p', 'Iterative biases not implemented fot MV and TT estimators'

        config = (self.param.nlev_t, self.param.nlev_p, self.param.beam, self.param.lmin_elm, self.param.lmax_ivf, self.param.lmax_qlm)

        # TODO: the cached files do not depend on the itermax
        iterbiases = n0n1_iterative.polMAPbiases(config, fidcls_unl=self.param.cls_unl, itrmax = itermax, cacher=self.cacher_param)
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=version)
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true

    # def get_n0_iter(self, itermax=15, version=''):
    #     if self.k == 'p_p':
    #         lmin_ivf = self.param.lmin_elm  # TODO: what about lmin_blm ?
    #     elif self.k == 'ptt':
    #         lmin_ivf = self.param.lmin_tlm
    #     elif self.k == 'p':
    #         assert self.param.lmin_tlm == self.param.lmin_elm, "Dont know what lmin_ivf to take in this case, need to update nhl library"
    #         lmin_ivf = self.param.lmin_tlm

    #     cacher = self.cacher_param
    #     tail = '_lminivf_{}_itmax{}_{}'.format(lmin_ivf, itermax, version)
    #     fn_n0s = 'N0s_biased' + tail
    #     fn_n0s_unb = 'N0s_unbiased' + tail
    #     fn_resp_fid = 'resp_fid' + tail
    #     fn_resp_true = 'resp_true' + tail
    #     cached_files = [fn_n0s, fn_n0s_unb, fn_resp_fid, fn_resp_true]
    #     if 'wN1' in version:
    #         fn_N1s = 'N1s_biased' + tail
    #         fn_N1s_unb = 'N1s_unbiased' + tail
    #         cached_files += [fn_N1s, fn_N1s_unb]
    #     if any (not cacher.is_cached(fn) for fn in cached_files):
    #         ret = nhl.get_N0_iter(
    #             self.k, self.param.nlev_t, self.param.nlev_p, self.param.beam, self.param.cls_unl, lmin_ivf, self.param.lmax_ivf, itermax, ret_delcls=True, ret_resp=True, version=version)
    #         cacher.cache(fn_n0s, ret[0])
    #         cacher.cache(fn_n0s_unb, ret[1])
    #         cacher.cache(fn_resp_fid, ret[2])
    #         cacher.cache(fn_resp_true, ret[3])
    #         if 'wN1' in version:
    #             cacher.cache(fn_N1s, ret[4])
    #             cacher.cache(fn_N1s_unb, ret[5])           
    #     N0s_biased = cacher.load(fn_n0s)
    #     N0s_unbiased = cacher.load(fn_n0s_unb)
    #     rgg_fid  = cacher.load(fn_resp_fid)     
    #     rgg_true = cacher.load(fn_resp_true)     
    #     if 'wN1' in version:
    #         N1s_biased  = cacher.load(fn_N1s)     
    #         N1s_unbiased = cacher.load(fn_N1s_unb)   
    #     return N0s_biased, N0s_unbiased, rgg_fid, rgg_true if not 'wN1' in version else N0s_biased, N0s_unbiased, rgg_fid, rgg_true, N1s_biased, N1s_unbiased

    # def get_N1_qe(self):
    #     if resp_gradcls:
    #         fn_N1 = 'N1_QE_respgradcls'
    #     else:
    #         fn_N1 = 'N1_QE'
    #         if not self.cacher.is_cached(fn_N1):
    #             print('computing N1 QE')
    #             if resp_gradcls:
    #                 cls_f = self.cls_grad
    #             else:
    #                 cls_f = self.cls_len
    #             n1lib = n1_fft.n1_fft(fals = self.confi_crude.fals, cls_w=cls_len_fid, cls_f=cls_f, cpp=self.cpp_prior, lminbox=self.lminbox, lmaxbox=self.lmaxbox, k2l=self.k2l)
                
    #             tmp_ls, = np.where(n1lib.box.mode_counts()[:self.lmax_qlm + 1] > 0)
    #             idx = (np.linspace(1, len(tmp_ls) - 1, 50)).astype(int)
    #             Ls = tmp_ls[idx]

    #             n1 = np.array([n1lib.get_n1(self.k, L, do_n1mat=False) for L in Ls])
    #             resp = self.get_resp('QE', resp_gradcls=resp_gradcls)

    #             # Interpolate on the normed and L^2(L+1)^2 N1
    #             ells_= np.arange(self.lmax_qlm+1)
    #             N1 = spline(Ls, n1 * cli(resp[Ls])**2 * (Ls * (Ls+1))**2 , s=0, k=3, ext='zeros')(ells_)  / ( ells_ * (ells_ +1) )**2
    #             self.cacher.cache(fn_N1, N1)
    #         N1s_qe = self.cacher.load(fn_N1)
    #         return N1s_qe


    # def get_N1_iter(self, itermax=15, wN1f=False, resp_gradcls=True):

    #     if self.k == 'p_p':
    #         lmin_ivf = self.param.lmin_elm  # TODO: what about lmin_blm ?
    #     elif self.k == 'ptt':
    #         lmin_ivf = self.param.lmin_tlm
    #     elif self.k == 'p':
    #         assert self.param.lmin_tlm == self.param.lmin_elm, "Dont know what lmin_ivf to take in this case, need to update nhl library"
    #         lmin_ivf = self.param.lmin_tlm


    #     tail = '_lminivf_{}_itmax{}_{}'.format(lmin_ivf, itermax, version)
    #     fn_N1s = 'N1s_biased' + tail
    #     fn_N1s_unb = 'N1s_unbiased' + tail

    #     if any (not self.cacher.is_cached(fn) for fn in [fn_N1s]):

    #         nmax = 3

    #         N0 = self.get_N0('MAP')
    #         cpp = np.copy(self.cls_unl['pp'])
    #         clwf = cpp[:self.lmax_qlm + 1] * cli(cpp[:self.lmax_qlm + 1] + N0[:self.lmax_qlm + 1])
    #         cpp[:self.lmax_qlm + 1] *= (1. - clwf)
            
    #         lib_len = len_fft.len_fft(self.cls_unl, cpp, lminbox=self.lminbox, lmaxbox=self.lmaxbox, k2l=self.k2l)
    #         cls_plen_2d =  lib_len.lensed_cls_2d(nmax=nmax)

    #         cls_plen = {k: lib_len.box.sum_in_l(cls_plen_2d[k]) * cli(lib_len.box.mode_counts() * 1.) for k in cls_plen_2d.keys()}
    #         cls_filt = cls_plen
    #         ivfs_cls, fals = utils_n1.get_ivf_cls(cls_plen, cls_filt, self.lmin_ivf, self.lmax_ivf, self.nlev_t, self.nlev_p,  self.nlev_t, self.nlev_p, self.transf,
    #                                     jt_tp=self.jt_TP)

    #         cls_f_2d = lib_len.lensed_gradcls_2d(nmax=nmax) # response cls
    #         cls_f = {k: lib_len.box.sum_in_l(cls_f_2d[k]) * cli(lib_len.box.mode_counts() * 1.) for k in cls_f_2d.keys()}
            
    #         cls_w = cls_f 

    #         if self.k == 'ptt':
    #             fals['ee'] *= 0.
    #             fals['bb'] *= 0.
    #             ivfs_cls['ee'] *= 0.
    #             ivfs_cls['bb'] *= 0.
    #         if self.k == 'p_p':
    #             fals['tt'] *= 0.
    #             ivfs_cls['tt'] *= 0.
    #             ivfs_cls['te'] *= 0.
    #         if self.k in ['ptt', 'p_p']:
    #             cls_w['te'] *= 0.

    #         ls, = np.where(lib_len.box.mode_counts()[:self.lmax_ivf + 1] > 0)
    #         fals_spl  = {k: spline(ls, fals[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax_ivf + 1) * 1.) for k in fals.keys()}
    #         cls_w_spl = {k: spline(ls, cls_w[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax_ivf + 1) * 1.) for k in cls_w.keys()}
    #         cls_f_spl = {k: spline(ls, cls_f[k][ls], k=2, s=0, ext='zeros')(np.arange(self.lmax_ivf + 1) * 1.) for k in cls_f.keys()}
            
    #         #This one spline probably not needed
    #         cpp_spl = spline(ls, cpp[ls], k=2, s=0, ext='zeros')(np.arange(self.lmax_ivf + 1) * 1.)

    #         libn1 = n1_fft.n1_fft(fals_spl, cls_w_spl, cls_f_spl, cpp_spl, lminbox=self.lminbox,  lmaxbox=self.lmaxbox, k2l=self.k2l)
    #         # Choose Ls correspoding to the ls in the box to be able to normalize the n1
    #         tmp_ls, = np.where(lib_len.box.mode_counts()[:self.lmax_qlm + 1] > 0)
    #         idx = (np.linspace(1, len(tmp_ls) - 1, 50)).astype(int)
    #         Ls = tmp_ls[idx]
    #         n1 =  np.array([libn1.get_n1(self.k, L, do_n1mat=False) for L in Ls])
    #         resp_ = self.get_resp('MAP')
    #         # Interpolate on the normed and L^2(L+1)^2 N1
    #         ells_ = np.arange(len(N0)) * 1.
    #         N1 = spline(Ls, n1 * cli(resp_[Ls])**2 * (Ls * (Ls+1))**2 , s=0, k=3, ext='zeros')(ells_) / ( ells_ * (ells_ +1) )**2
            
    #         self.cacher.cache(fn_N1, N1)
    #     N1 = self.cacher.load(fn_N1)
    #     return N1




    def get_wf_fid(self, itermax=15, version=''):
        """Fiducial iterative Wiener filter.
        
        Normalisation of :math:`phi^{MAP}`
        :math:`\mathcal{W} = \frac{C_{\phi\phi, \mathrm{fid}}}{C_{\phi\phi, \mathrm{fid}} + 1/\mathcal{R}_L}`
 
        """
        _, _, resp_fid, _ = self.get_N0_N1_iter(itermax=itermax, version=version)
        return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]))

    def get_wf_sim(self, simidx, itr):
        """Get the Wiener from the simulations.

        :math:`\hat \mathcal{W} = \frac{C_L{\phi^{\rm MAP} \phi{\rm in}}}{C_L{\phi^{\rm in} \phi{\rm in}}}`
        
        """
        fn = 'wf_sim_it{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            wf = self.get_cpp_itXinput(simidx, itr) * utils.cli(self.get_cpp_input(simidx))
            cacher.cache(fn, wf)
        return cacher.load(fn)

    def get_wf_eff(self, imin = 0, imax = 4, itermax=15):
        """Effective Wiener filter averaged over several simulations
        We spline interpolate the ratio between the effective WF from simulations and the fiducial WF
        We take into account the sky fraction to get the simulated WFs
        
        Args:
            imin: Minimum index of simulations
            imax: Maximum index of simulations
            itermax: Iteration of the MAP estimator

        Returns:
            wf_eff: Effective Wiener filter
            wfcorr_spl: Splined interpolation of the ratio between the effective and the fiducial Wiener filter
        
        """
    #    ckk_in = {}
    #     ckk_cross = {}
        wf_fid = self.get_wf_fid(itermax)
        wfsims_bias = np.zeros([imax-imin, len(wf_fid)])
    #     for i, f in enumerate(dat_files):
    #         _, ckk_in[f], ckk_cross[f] = np.loadtxt(os.path.join(DIR, f, 'MAP_cls.dat')).transpose()
    #         wfcorr_full[i] =  ckk_cross[f] *cli(ckk_in[f] * wfpred) 
        for isim in range(imax-imin):
            wfsims_bias[isim] = self.get_wf_sim(isim, itermax) / self.fsky * utils.cli(wf_fid)
        wfcorr_mean = np.mean(wfsims_bias, axis=0)
        wfcorr_spl = np.zeros(len(wf_fid))
        ells = np.arange(self.lmax_qlm+1)
        wfcorr_spl[ells] = spline(ells, wfcorr_mean[ells])(ells)
        wf_eff = wf_fid * wfcorr_spl
        return wf_eff[:self.lmax_qlm+1], wfcorr_spl



    def load_rdn0_kk_map(self, idx):
        """Load previously computed RDN0 estimate.
    
        See the file rdn0_cs.py to compute the RDN0 for a given paramfile and simidx.

        Args:
            idx: index of simulation

        Returns:
            rdn0: unormalized realisation dependent bias, raw phi-based spectum obtained by 1/4 L^2 (L + 1)^7 * 1e7 times this
            kds: data x sim QE estimates 
            kss: sim x sim QE estimates 
        """
        outputdir = rdn0_cs.output_sim(self.param.suffix, idx)
        fn = opj(outputdir, 'cls_dsss.dat')
        if not os.path.exists(fn):
            itlibdir = self.param.libdir_iterators(self.k, idx, self.version)
            Nroll = 10
            Nsims = 100
            ss_dict =  Nroll * (np.arange(Nsims) // Nroll) + (np.arange(Nsims) + 1) % Nroll
            rdn0_cs.export_dsss(itlibdir, self.param.suffix, idx, ss_dict)
        kds, kss, _, _, _, _ = np.loadtxt(fn).transpose()
        rdn0 = 4 * kds - 2 * kss

        return rdn0, kds, kss

    def get_rdn0_map(self, idx):
        """Get the normalized iterative RDN0 estimate        
        
        Args:
            idx: index of simulation

        Returns:
            RDN0: Normalized realisation dependent bias of Cpp MAP
        """
        rdn0, kds, kss = self.load_rdn0_kk_map(idx)
        assert self.itmax == 15, "Need to check if the exported RDN0 correspond to the same iteration as the Cpp MAP" 
        # TODO  maybe not relevant if everything is converged ?
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=15)
        RDN0 = rdn0[:self.lmax_qlm+1] * pp2kk(np.arange(self.lmax_qlm+1)) * 1e7 * utils.cli(r_gg_fid[:self.lmax_qlm+1])**2
        return RDN0
        
    def get_nsims(self):
        """Return the number of simulations reconstructed up to itmax"""
        