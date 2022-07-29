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

        self.config = (self.param.nlev_t, self.param.nlev_p, self.param.beam, 
                       (self.param.lmin_tlm,  self.param.lmin_elm, self.param.lmin_blm), 
                       self.param.lmax_ivf, self.param.lmax_qlm)


        if type(self.param.sims.sims_cmb_len).__name__ == 'cmb_len_ffp10':
            self.sims_unl = planck2018_sims.cmb_unl_ffp10() 


    def libdir_sim(self, simidx):
        return opj(self.TEMP,'%s_sim%04d'%(self.k, simidx) + self.version)


    def get_itlib_sim(self, simidx):
        return self.param.get_itlib(self.k, simidx, self.version, self.tol_iter)


    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)


    def get_plm(self, simidx, itr):
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

    def get_N0_N1_QE(self, version=''):
        assert self.k =='p_p', 'Biases not implemented fot MV and TT estimators'

        iterbiases = n0n1_iterative.polMAPbiases(self.config, fidcls_unl=self.param.cls_unl, itrmax = 0, cacher=self.cacher_param)
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=version)
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true

    def get_N0_N1_iter(self, itermax=15, version=''):
        assert self.k =='p_p', 'Iterative biases not implemented fot MV and TT estimators'

        # config = (self.param.nlev_t, self.param.nlev_p, self.param.beam, self.param.lmin_elm, self.param.lmax_ivf, self.param.lmax_qlm)
       
        iterbiases = n0n1_iterative.polMAPbiases(self.config, fidcls_unl=self.param.cls_unl, itrmax = itermax, cacher=self.cacher_param)
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = iterbiases.get_n0n1(cls_unl_true=None, cls_noise_true=None, version=version)
        return N0_biased, N1_biased_spl, r_gg_fid, r_gg_true


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
            wf = self.get_cpp_itXinput(simidx, itr) * utils.cli(self.get_cpp_input(simidx)) / self.fsky
            cacher.cache(fn, wf)
        return cacher.load(fn)

    def get_wf_eff(self, itmax_sims=None, itmax_fid=15, version='', do_spline=True, lmin_interp=0, lmax_interp=None,  k=3, s=None, verbose=False):
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
        if itmax_sims is None: itmax_sims = self.itmax
        wf_fid = self.get_wf_fid(itmax_fid, version=version)
        nsims = self.get_nsims_itmax()
        # sims_idx = self.get_idx_sims_done(itmax=15)
        print(f'I use {nsims} sims to estimate the effective WF')
        wfsims_bias = np.zeros([nsims, len(wf_fid)])
    #     for i, f in enumerate(dat_files):
    #         _, ckk_in[f], ckk_cross[f] = np.loadtxt(os.path.join(DIR, f, 'MAP_cls.dat')).transpose()
    #         wfcorr_full[i] =  ckk_cross[f] *cli(ckk_in[f] * wfpred) 
        for isim in range(nsims):
            if verbose: print(f'wf eff {isim}/{nsims}')
            wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims) * utils.cli(wf_fid)
            # wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims) * utils.cli(wf_fid)
        wfcorr_mean = np.mean(wfsims_bias, axis=0)
        if do_spline:
            wfcorr_spl = np.zeros(len(wf_fid))
            if lmax_interp is None: lmax_interp=self.lmax_qlm
            ells = np.arange(lmin_interp, lmax_interp+1)
            wfcorr_spl[ells] = spline(ells, wfcorr_mean[ells], k=k, s=s)(ells)
        else:
            wfcorr_spl = wfcorr_mean
        wf_eff = wf_fid * wfcorr_spl
        return wf_eff[:self.lmax_qlm+1], wfcorr_spl


    def get_num_rdn0(self):
        "Retunr number of sims with RDN0 estimated"
        idx = 0 
        rdn0_computed = True
        while rdn0_computed:      
            outputdir = rdn0_cs.output_sim(self.param.suffix, idx)
            fn = opj(outputdir, 'cls_dsss.dat')
            try:
                kdsfid, kssfid, _, _, _, _ = np.loadtxt(fn).transpose()
                idx +=1
            except OSError:
                 rdn0_computed = False
        return idx
    
    def get_R_eff(self, lmin_interp=0, lmax_interp=None, k=3, s=None):
        """Correct the Response (normalisation) using the RDN0 estimates from several sims"""
        
        rdn0s = []
        idx = 0 
        rdn0_computed = True

        while rdn0_computed:      
            outputdir = rdn0_cs.output_sim(self.param.suffix, idx)
            fn = opj(outputdir, 'cls_dsss.dat')
            try:
                kdsfid, kssfid, _, _, _, _ = np.loadtxt(fn).transpose()
                rdn0s.append(4 * kdsfid - 2 * kssfid)
                idx +=1
            except OSError:
                 rdn0_computed = False
        print(f'I average {idx} sims with rdn0 to get effective response')
        rdn0s = np.array(rdn0s)
        rdn0_mean = np.mean(rdn0s, axis=0)
        
        rdn0 = rdn0_mean[:self.lmax_qlm+1] * pp2kk(np.arange(self.lmax_qlm+1)) * 1e7 
        
        N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=15)
        Reff_Spline = np.ones(self.lmax_qlm+1) * r_gg_fid
        if lmax_interp is None: lmax_interp=self.lmax_qlm
        ells = np.arange(lmin_interp, lmax_interp+1)
        # Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] / self.fsky * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] /self.fsky * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        # kR_eff = np.zeros(4001)
        # R_eff  = r_gg_fid * Reff_Spline
        return Reff_Spline


    def load_rdn0_map(self, idx):
        """Load previously computed RDN0 estimate.
    
        See the file rdn0_cs.py to compute the RDN0 for a given paramfile and simidx.

        Args:
            idx: index of simulation

        Returns:
            rdn0: unormalized realisation dependent bias (phi-based spectum)
            pds: data x sim QE estimates 
            pss: sim x sim QE estimates 
        """
        outputdir = rdn0_cs.output_sim(self.param.suffix, idx)
        fn = opj(outputdir, 'cls_dsss.dat')
        if not os.path.exists(fn):
            itlibdir = self.param.libdir_iterators(self.k, idx, self.version)
            Nroll = 10
            Nsims = 100
            ss_dict =  Nroll * (np.arange(Nsims) // Nroll) + (np.arange(Nsims) + 1) % Nroll
            rdn0_cs.export_dsss(itlibdir, self.param.suffix, idx, ss_dict)
        pds, pss, _, _, _, _ = np.loadtxt(fn).transpose()
        pds *= pp2kk(np.arange(len(pds))) * 1e7 
        pss *= pp2kk(np.arange(len(pss))) * 1e7 
        rdn0 = 4 * pds - 2 * pss
        return rdn0, pds, pss

    def get_rdn0_map(self, idx, useReff=True,  Reff=None, lmin_interp=0, lmax_interp=None, k=3, s=None):
        """Get the normalized iterative RDN0 estimate        
        
        Args:
            idx: index of simulation

        Returns:
            RDN0: Normalized realisation dependent bias of Cpp MAP
        """
        rdn0, kds, kss = self.load_rdn0_map(idx)
        assert self.itmax == 15, "Need to check if the exported RDN0 correspond to the same iteration as the Cpp MAP" 
        # Fixme  maybe not relevant if everything is converged ?
        RDN0 = rdn0[:self.lmax_qlm+1]
        
        if Reff is None and useReff:
            Reff = self.get_R_eff(lmin_interp, lmax_interp, k=k, s=s)
        if useReff:
            RDN0 *= utils.cli(Reff[:self.lmax_qlm+1])**2
        else:
            N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=15)
            RDN0 *= utils.cli(r_gg_fid[:self.lmax_qlm+1])**2
        return RDN0 / self.fsky
        

    def get_nsims_itmax(self):
        """Return the number of simulations reconstructed up to itmax"""
        nsim = 0
        while statics.rec.maxiterdone(self.libdir_sim(nsim)) >= self.itmax:
            nsim+=1
        return nsim

    def get_idx_sims_done(self, itmax=15):
        isdone = [False]*5000
        for i in range(5000):
            if self.maxiterdone(i) ==itmax:
                isdone[i] = True
        return isdone

    def maxiterdone(self, simidx):
        return statics.rec.maxiterdone(self.libdir_sim(simidx))