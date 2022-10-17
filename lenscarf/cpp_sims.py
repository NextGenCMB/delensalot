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
from plancklens.utils import mchash
import plancklens.utils as ut
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


_write_alm = lambda fn, alm : hp.write_alm(fn, alm, overwrite=True)

class cpp_sims_lib:
    def __init__(self, k, v='', param_file='cmbs4wide_planckmask', label=''):
        """Helper library to plot results from MAP estimation of simulations.
        
        This class loads the results of the runs done with the param_file and the options asked
        You should first make sure you have run the iterations on the simulations you want to load.
        """
        
        # Load the parameters defined in the param_file
        self.param_file = param_file
        self.param = importlib.import_module('lenscarf.params.'+param_file)
        self.k = k
        # self.itmax = itmax
        # self.tol = tol
        # self.tol_iter  = 10 ** (- self.tol) 
        # self.imin = imin
        # self.imax = imax
        self.version = v
        # self.iters = np.arange(itmax+1)
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


    def get_itlib_sim(self, simidx, tol):
        tol_iter  = 10 ** (- tol) 
        return self.param.get_itlib(self.k, simidx, self.version, tol_iter)


    def cacher_sim(self, simidx, verbose=False):
        return cachers.cacher_npy(opj(self.libdir_sim(simidx), 'cpplib'), verbose=verbose)


    def get_plm(self, simidx, itr, use_cache=True):
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_it{itr:03.0f}"
            if not cacher.is_cached(fn):
                plm = statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]
                cacher.cache(fn, plm)
            plm = cacher.load(fn)
            return plm
        else:
            return statics.rec.load_plms(self.libdir_sim(simidx), [itr])[0]

    def get_plm_input(self, simidx, use_cache=True):
        if use_cache:
            cacher = cachers.cacher_npy(self.libdir_sim(simidx))
            fn = f"phi_plm_input"
            if not cacher.is_cached(fn):
                plm_in = alm_copy(self.sims_unl.get_sim_plm(simidx), mmaxin=None, lmaxout=self.lmax_qlm, mmaxout=self.mmax_qlm)
                cacher.cache(fn, plm_in)
            plm_in = cacher.load(fn)
            return plm_in
        else:
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

    def get_cl(self, alm, blm=None):
        if blm is None: blm = alm 
        return uhp.alm2cl(alm, blm, self.param.lmax_qlm, self.param.mmax_qlm, None)

    def almxfl(self, alm, cl):
        return uhp.almxfl(alm, cl, mmax = self.param.mmax_qlm, inplace=False)


    def get_cpp_input(self, simidx):
        fn_cpp_in = 'cpp_input'
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_in):
            plmin = self.get_plm_input(simidx)
            cpp_in = self.get_cl(plmin)
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
            cpp_itXin = self.get_cl(plmit, plmin)
            cacher.cache(fn, cpp_itXin)
        cpp_itXin = cacher.load(fn)
        return cpp_itXin
    

    def get_cpp(self, simidx, itr):
        fn_cpp_it = 'cpp_it_{}'.format(itr)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn_cpp_it):
            plm = self.get_plm(simidx, itr)
            cpp = self.get_cl(plm)
            cacher.cache(fn_cpp_it, cpp)
        cpp = cacher.load(fn_cpp_it)
        return cpp


    # def get_cpp_submf(self, simidx, itr, mf=None):
    #     fn_cpp_it = 'cpp_it_{}_'.format(itr)
    #     cacher = self.cacher_sim(simidx)
    #     if not cacher.is_cached(fn_cpp_it):
    #         plm = self.get_plm(simidx, itr)
    #         cpp = alm2cl(plm, plm, self.lmax_qlm, self.mmax_qlm, None)
    #         cacher.cache(fn_cpp_it, cpp)
    #     cpp = cacher.load(fn_cpp_it)
    #     return cpp


    def get_cpp_itmax(self, simidx, itmax):
        """Get Cpp at all iterations up to itmax"""
        cacher = self.cacher_sim(simidx)
        fn_cpp_it = lambda itr:  'cpp_it_{}'.format(itr)
        if np.any([not cacher.is_cached(fn_cpp_it(itr)) for itr in np.arange(itmax+1)]):
            plms = statics.rec.load_plms(self.libdir_sim(simidx), np.arange(itmax+1))
            for itr in np.arange(itmax+1):
                if not cacher.is_cached(fn_cpp_it(itr)):
                    _cpp = self.get_cl(plms[itr])
                    cacher.cache(fn_cpp_it(itr), _cpp)
        cpp = []
        for itr in np.arange(itmax+1):
            cpp.append(cacher.load(fn_cpp_it(itr)))
        return cpp   


    def get_qe_resp(self):
        fn_resp_qe = 'resp_qe_{}'.format(self.k) + self.version
        cacher = self.cacher_param
        if not cacher.is_cached(fn_resp_qe):
            R = qresp.get_response(self.k, self.param.lmax_ivf, 'p', self.param.cls_len, self.param.cls_len, {'e': self.param.fel, 'b': self.param.fbl, 't':self.param.ftl}, lmax_qlm=self.param.lmax_qlm)[0]
            cacher.cache(fn_resp_qe, R)
        R = cacher.load(fn_resp_qe)
    
    def get_cpp_qe(self, simidx):
        fn_cpp_qe = 'cpp_qe'
        # fn_resp_qe = 'resp_qe'
        cacher = self.cacher_sim(simidx)
        # if not cacher.is_cached(fn_resp_qe):
        #     R = qresp.get_response(self.k, self.param.lmax_ivf, 'p', self.param.cls_len, self.param.cls_len, {'e': self.param.fel, 'b': self.param.fbl, 't':self.param.ftl}, lmax_qlm=self.param.lmax_qlm)[0]
        #     cacher.cache(fn_resp_qe, R)
        R = self.get_qe_resp()
        if not cacher.is_cached(fn_cpp_qe):
            cpp = self.param.cls_unl['pp'][:self.lmax_qlm + 1]
            WF = cpp * utils.cli(cpp + utils.cli(R))
            cpp_qe_wf = self.get_cpp(simidx, 0)
            cpp_qe = cpp_qe_wf * utils.cli(WF)**2
            cacher.cache(fn_cpp_qe, cpp_qe)
        cpp_qe = cacher.load(fn_cpp_qe)
        return cpp_qe, R

    def get_mf0(self, simidx):
        """Get the QE mean-field"""
        return np.load(opj(self.libdir_sim(simidx), 'mf.npy'))
    

    def get_mf_it(self, simidx, itr, tol, ret_alm=False, verbose=False):
        tol_iter  = 10 ** (- tol) 
        cacher = self.cacher_sim(simidx)
        fn_mf1 = 'mf1_it{}'.format(itr)
        fn_mf2 = 'mf2_it{}'.format(itr)
        if not cacher.is_cached(fn_mf1) or not cacher.is_cached(fn_mf2):
            if verbose:
                print('Starting to estimate MF for it {} from sim {}'.format(itr, simidx))
            itlib = self.get_itlib_sim(simidx)
            filtr = itlib.filter
            filtr.set_ffi(itlib._get_ffi(itr)) # load here the phi map you want 
            chain_descr = self.param.chain_descrs(self.param.lmax_unl, tol_iter)
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
        if version == 'wN1_end':
            _, N1, resp_fid, _ = self.get_N0_N1_iter(itermax=itermax, version='')
            return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]) + N1[:self.lmax_qlm+1])

        else:
            _, N1, resp_fid, _ = self.get_N0_N1_iter(itermax=itermax, version=version)
            return self.cpp_fid[:self.lmax_qlm+1] * utils.cli(self.cpp_fid[:self.lmax_qlm+1] + utils.cli(resp_fid[:self.lmax_qlm+1]))

    def get_wf_sim(self, simidx, itr, mf=False, mc_sims=None):
        """Get the Wiener from the simulations.

        :math:`\hat \mathcal{W} = \frac{C_L{\phi^{\rm MAP} \phi{\rm in}}}{C_L{\phi^{\rm in} \phi{\rm in}}}`
        
        """
        fn = 'wf_sim_it{}'.format(itr) if mf is False else 'wf_sim_it{}_mfsub'.format(itr)
        # print(fn)
        cacher = self.cacher_sim(simidx)
        if not cacher.is_cached(fn):
            if mf is False:
                wf = self.get_cpp_itXinput(simidx, itr) * utils.cli(self.get_cpp_input(simidx)) / self.fsky
            else:
                plmin = self.get_plm_input(simidx)
                # plmit = self.plms[simidx][itr]
                mf = self.get_mf(itr, mc_sims, simidx, use_cache=True)
                plmit = self.get_plm(simidx, itr, use_cache=True) - mf
                cpp_itXin = self.get_cl(plmit, plmin)
                
                wf = cpp_itXin * utils.cli(self.get_cpp_input(simidx)) / self.fsky
            cacher.cache(fn, wf)
        return cacher.load(fn)

    def get_wf_eff(self, itmax_sims=15, itmax_fid=15, mf=False, mc_sims=None, version='', do_spline=True, lmin_interp=0, lmax_interp=None,  k=3, s=None, verbose=False):
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
        # if itmax_sims is None: itmax_sims = self.itmax
        wf_fid = self.get_wf_fid(itmax_fid, version=version)
        nsims = self.get_nsims_itmax(itmax_sims)
        # sims_idx = self.get_idx_sims_done(itmax=15)
        print(f'I use {nsims} sims to estimate the effective WF')
        wfsims_bias = np.zeros([nsims, len(wf_fid)])
    #     for i, f in enumerate(dat_files):
    #         _, ckk_in[f], ckk_cross[f] = np.loadtxt(os.path.join(DIR, f, 'MAP_cls.dat')).transpose()
    #         wfcorr_full[i] =  ckk_cross[f] *cli(ckk_in[f] * wfpred) 
        for isim in range(nsims):
            if verbose: print(f'wf eff {isim}/{nsims}')
            wfsims_bias[isim] = self.get_wf_sim(isim, itmax_sims, mf=mf, mc_sims=mc_sims) * utils.cli(wf_fid)
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


    def get_num_rdn0(self, itr=50, ss_dict=None, mcs=None, Nroll=None):
        "Retunr number of sims with RDN0 estimated"
        idx = 0 
        rdn0_computed = True
        if ss_dict is None:
            Nroll = 10
            Nsims = 100
            mcs = np.arange(0, Nsims)
            ss_dict =  rdn0_cs._ss_dict(mcs, Nroll)
        while rdn0_computed:      
            outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
            fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
            # print(fn)
            # fn = opj(outputdir, 'cls_dsss.dat')
            try:
                kdsfid, kssfid, _, _, _, _ = np.loadtxt(fn).transpose()
                idx +=1
            except OSError:
                 rdn0_computed = False
        return idx
    
    def get_R_eff(self, lmin_interp=0, lmax_interp=None, k=3, s=None, itr=None, mcs=np.arange(0, 100), Nroll=10):
        """Correct the Response (normalisation) using the RDN0 estimates from several sims"""
        
        rdn0s = []
        idx = 0 
        rdn0_computed = True

        while rdn0_computed:      
            outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
            fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
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
        # Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        Reff_Spline[ells] = r_gg_fid[ells] * spline(ells, rdn0[ells] /self.fsky * utils.cli(r_gg_fid[ells]), k=k, s=s)(ells)
        # kR_eff = np.zeros(4001)
        # R_eff  = r_gg_fid * Reff_Spline
        return Reff_Spline


    def load_rdn0_map(self, idx, itr, mcs, Nroll):
        """Load previously computed RDN0 estimate.
    
        See the file rdn0_cs.py to compute the RDN0 for a given paramfile and simidx.

        Args:
            idx: index of simulation

        Returns:
            rdn0: unormalized realisation dependent bias (phi-based spectum)
            pds: data x sim QE estimates 
            pss: sim x sim QE estimates 
        """
        outputdir = rdn0_cs.output_sim(self.k, self.param.suffix, idx)
        fn = opj(outputdir, rdn0_cs.fn_cls_dsss(itr, mcs, Nroll))
        if not os.path.exists(fn):
            itlibdir = self.param.libdir_iterators(self.k, idx, self.version)
            ss_dict =  rdn0_cs._ss_dict(mcs, Nroll)
            rdn0_cs.export_dsss(itlibdir, self.k, self.param.suffix, idx, ss_dict, mcs, Nroll)
        pds, pss, _, _, _, _ = np.loadtxt(fn).transpose()
        pds *= pp2kk(np.arange(len(pds))) * 1e7 
        pss *= pp2kk(np.arange(len(pss))) * 1e7 
        rdn0 = 4 * pds - 2 * pss
        return rdn0, pds, pss

    def get_rdn0_map(self, idx, itr, mcs=np.arange(0, 96), Nroll=8, useReff=True,  Reff=None, lmin_interp=0, lmax_interp=None, k=3, s=None):
        """Get the normalized iterative RDN0 estimate        
        
        Args:
            idx: index of simulation

        Returns:
            RDN0: Normalized realisation dependent bias of Cpp MAP
        """
        rdn0, kds, kss = self.load_rdn0_map(idx, itr, mcs, Nroll)
        # assert self.itmax == 15, "Need to check if the exported RDN0 correspond to the same iteration as the Cpp MAP" 
        # Fixme  maybe not relevant if everything is converged ?
        RDN0 = rdn0[:self.lmax_qlm+1]
        
        if Reff is None and useReff:
            Reff = self.get_R_eff(lmin_interp, lmax_interp, k=k, s=s, itr=itr, mcs=mcs, Nroll=Nroll)
        if useReff:
            RDN0 *= utils.cli(Reff[:self.lmax_qlm+1])**2
        else:
            N0_biased, N1_biased_spl, r_gg_fid, r_gg_true = self.get_N0_N1_iter(itermax=15)
            RDN0 *= utils.cli(r_gg_fid[:self.lmax_qlm+1])**2
        return RDN0 / self.fsky
        
    # def get_rdn0_qe(self, datidx, resp=None):
    #     fn_dir = rdn0_cs.output_sim(self.k, self.param.suffix, datidx)
    #     fn = os.path.join(fn_dir, 'QE_knhl.dat')
    #     if not os.path.exists(fn):
    #         print(fn)
    #         rdn0_cs.export_nhl(self.libdir_sim(datidx), self.k, self.param, datidx)
    #     GG = np.loadtxt(fn)
    #     GG *=  pp2kk(np.arange(len(GG))) * 1e7 
    #     if resp is None:
    #         _, _, resp, _ = self.get_N0_N1_QE()
    #     GG *= utils.cli(resp[:self.lmax_qlm+1])**2
    #     return GG


    def get_rdn0_qe(self):
        """Returns realization-dependent N0 lensing bias RDN0.

        """
        ds = self.param.qcls_ds.get_sim_stats_qcl(self.k, self.param.mc_sims_var, k2=self.k).mean()
        ss = self.param.qcls_ss.get_sim_stats_qcl(self.k, self.param.mc_sims_var, k2=self.k).mean()
        qe_resp = self.get_qe_resp()
        # _, qc_resp = self.param.qresp_dd.get_response(self.k1, self.ksource) * self.par.qresp_dd.get_response(self.k2, self.ksource)
        return self.get_cl(utils.cli(qe_resp)**2 * (4 * ds - 2. * ss))



    def get_nsims_itmax(self, itmax):
        """Return the number of simulations reconstructed up to itmax"""
        nsim = 0
        while statics.rec.maxiterdone(self.libdir_sim(nsim)) >= itmax:
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

    def get_mf(self, itmax, mc_sims, simidx=None, use_cache=True, verbose=False):
        """Get the mean field of the MAP, by averaging MAP estimates from a set simulations (caches the result).
            Adapted from plancklens.qest.get_sim_qlm_mf
            Args:
                itmax: Iteration of the MAP estimator
                mc_sims: simulation indices to use for the estimate.
                simidx: idx of simulation considered, to avoid subtracting twice the plm

            Returns:
                plm_mf: Mean field plm 
        """
        this_mcs = np.unique(mc_sims)

        # fname = os.path.join(self.cacher_param.lib_dir, 'simMF_k_%s.fits' % (mchash(mc_sims)))

        # if not os.path.exists(fname):
            
        #     MF = np.zeros(hp.Alm.getsize(self.lmax_qlm), dtype=complex)
        #     if len(this_mcs) == 0: return MF
        #     for i, idx in utils.enumerate_progress(this_mcs, label='calculating MF'):
        #         MF += self.get_plm(idx, itmax, use_cache=use_cache)
        #     MF /= len(this_mcs)
        #     _write_alm(fname, MF)
        #     print("Cached ", fname)

        # # MF = ut.alm_copy(hp.read_alm(fname), lmax=self.lmax_qlm)
        # MF = hp.read_alm(fname)

        # if simidx is not None and simidx in this_mcs:  # We dont want to include the sim we consider in the mean-field...
        #     print(f"Removing sim {simidx} from MF estimate")
        #     Nmf = len(this_mcs)
        #     mc_sims_less = np.delete(mc_sims, np.where(mc_sims==simidx))
        #     fn =  'simMF_k_%s.fits' % (mchash(mc_sims_less)) 
        #     if not os.path.exists(fn):
        #         MF = (MF - self.get_plm(simidx, itmax, use_cache=use_cache) / Nmf) * (Nmf / (Nmf - 1))
        #         _write_alm(fn, MF)
        #     MF = hp.read_alm(fn)


        cacher = cachers.cacher_npy(self.cacher_param.lib_dir, verbose=verbose)
        fn =  'simMF_k_%s.fits' % (mchash(mc_sims))
        if not cacher.is_cached(fn):
            MF = np.zeros(hp.Alm.getsize(self.lmax_qlm), dtype=complex)
            if len(this_mcs) == 0: return MF
            for i, idx in utils.enumerate_progress(this_mcs, label='calculating MF'):
                MF += self.get_plm(idx, itmax, use_cache=use_cache)
            MF /= len(this_mcs)
            # _write_alm(fname, MF)
            cacher.cache(fn, MF)
        MF = cacher.load(fn)

        if simidx is not None and simidx in this_mcs:  # We dont want to include the sim we consider in the mean-field...
            print(f"Removing sim {simidx} from MF estimate")
            Nmf = len(this_mcs)
            mc_sims_less = np.delete(mc_sims, np.where(mc_sims==simidx))
            fn =  'simMF_k_%s.fits' % (mchash(mc_sims_less)) 
            if not cacher.is_cached(fn):
                MF = (MF - self.get_plm(simidx, itmax, use_cache=use_cache) / Nmf) * (Nmf / (Nmf - 1))
                cacher.cache(fn, MF)
            MF = cacher.load(fn)
        return MF