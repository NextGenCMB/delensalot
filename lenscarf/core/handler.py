#!/usr/bin/env python

"""handler.py: This module receives input from lerepi, handles D.lensalot jobs and runs them.
    
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import os, sys
from os.path import join as opj
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import datetime
import getpass

import numpy as np
import healpy as hp

from scipy.signal import savgol_filter as sf

from MSC import pospace as ps
from plancklens import utils, qresp
from plancklens.sims import planck2018_sims

from lenscarf.core import mpi
from lenscarf.lerepi.core.visitor import transform
from lenscarf.lerepi.config.config_helper import data_functions as df
from lenscarf.iterators import cs_iterator
from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators.statics import rec as rec
from lenscarf.iterators import iteration_handler
from lenscarf.opfilt.bmodes_ninv import template_bfilt


class Notebook_interactor():
    '''
    Interface for notebooks,
     * load per-freq fg/noise/ maps/alms,
        * QU -> EB and vice versa
        * map2alm and vice versa
        * masking if needed
     * combine per-freq alms,
        * including weights, beams, pixwindows
     * calculate power spectra,
        * per-freq / combined
        * masking if needed
        * binning if needed
     * load power spectra
        * see read_data() and read_data_v2()
    '''
    def __init__(self, Interactor_model):
        self.__dict__.update(Interactor_model.__dict__)

        
    @log_on_start(logging.INFO, "read_data() started")
    @log_on_end(logging.INFO, "read_data() finished")
    def read_data(self, dm, simids=None, edges=None, dlm_mod=False, dir_idx=0):
        bcl_cs = np.zeros(shape=(len(self.iterations)+2, len(self.nlevels), len(self.simids), len(self.edges)-1))
        bcl_L = np.zeros(shape=(len(self.iterations)+2, len(self.nlevels), len(self.simids), len(self.edges)-1))
        for simidx, simid in enumerate(self.simids):
            if dm.iteration.dlm_mod:
                data = np.load(self.TEMP + '/plotdata{}/{}'.format(self.vers_str,self.dirid) + '/ClBBwf_sim%04d_dlmmod_fg%2s_res2b3acm.npy'%(simid, dm.map_delensing.fg))
            else:
                data = np.load(self.TEMP + '/plotdata{}/{}'.format(self.vers_str,self.dirid) + '/ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(simid, dm.map_delensing.fg))
            # data =  np.load(dirroot_loc + '{}'.format(self.dirid) + '/Lenscarf_plotdata_ClBB_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg))
            bcl_L[0,:,simidx] = data[0][0]
            bcl_cs[0,:,simidx] = data[1][0]

            bcl_L[1,:,simidx] = data[0][1]
            bcl_cs[1,:,simidx] = data[1][1]

            for iti, it in enumerate(self.iterations):
                bcl_L[2+iti,:,simidx] = data[0][2+iti]
                bcl_cs[2+iti,:,simidx] = data[1][2+iti]

        print('dirid: {}'.format(self.dirid))

        return bcl_L, bcl_cs


    @log_on_start(logging.INFO, "read_data_v2() started")
    @log_on_end(logging.INFO, "read_data_v2() finished")
    def read_data_v2(self, edges_id=0):
        bcl_cs = np.zeros(shape=(len(self.its)+2, len(self.mask_ids), len(self.simidxs), len(self.edges[edges_id])-1))
        bcl_L = np.zeros(shape=(len(self.its)+2, len(self.mask_ids), len(self.simidxs), len(self.edges[edges_id])-1))
        
        print('Loading {} sims from {}'.format(len(self.simidxs),  '/'.join([f for f in self.file_op(0, self.fg, 0).split('/')[:-1]])))
        for simidxi, simidx in enumerate(self.simidxs):
            _file_op = self.file_op(simidx, self.fg, 0)
            data = np.load(_file_op)
            bcl_L[0,:,simidxi] = data[0][0]
            bcl_cs[0,:,simidxi] = data[1][0]

            bcl_L[1,:,simidxi] = data[0][1]
            bcl_cs[1,:,simidxi] = data[1][1]

            for iti, it in enumerate(self.its):
                bcl_L[2+iti,:,simidxi] = data[0][2+iti]
                bcl_cs[2+iti,:,simidxi] = data[1][2+iti]

        return bcl_L, bcl_cs

    
    @log_on_start(logging.INFO, "load_foreground_freqs_mapalm() started")
    @log_on_end(logging.INFO, "load_foreground_freqs_mapalm() finished")
    def load_foreground_freqs_mapalm(self, mask=1, rotate_alms=False):
        if self.fc.flavour == 'QU':
            for freq in self.ec.freqs:
                if self.fc.fns_syncdust:
                    buff = hp.read_map(self.fc.fns_syncdust.format(freq=freq, nside=self.nside), field=(1,2))*mask*self.fc.map_fac
                else:
                    buff = hp.read_map(self.fc.fns_sync.format(freq=freq, nside=self.nside, simidx=self.fc.simidx, fg_beamstring=self.fc.fg_beamstring[freq]), field=(1,2))*mask*self.fc.map_fac + hp.read_map(self.fc.fns_dust.format(freq=freq, nside=self.nside, simidx=self.fc.simidx, fg_beamstring=self.fc.fg_beamstring[freq]), field=(1,2))*mask*self.fc.map_fac
                self.data['fg']['fs']['map'][freq]['QU'] = buff
                self.data['fg']['fs']['alm'][freq]['EB'] = np.array(hp.map2alm_spin(buff, spin=2, lmax=self.lmax))
                if rotate_alms:
                    rotator = hp.rotator.Rotator(coord=rotate_alms)
                    self.data['fg']['fs']['alm'][freq]['EB'][0] = rotator.rotate_alm(self.data['fg']['fs']['alm'][freq]['EB'][0])
                    self.data['fg']['fs']['alm'][freq]['EB'][1] = rotator.rotate_alm(self.data['fg']['fs']['alm'][freq]['EB'][1])
        elif self.fc.flavour == 'EB':
            log.error('not yet implemented')


    @log_on_start(logging.INFO, "calc_powerspectrum_binned_fromEB() started")
    @log_on_end(logging.INFO, "calc_powerspectrum_binned_fromEB() finished")   
    def calc_powerspectrum_binned_fromEB(self, maps_mask, templates, freq='comb', component='cs', nlevel='fs', edges=None, lmax=None):
        ## Template

        if edges is None:
            edges_loc = np.array([self.edges[0],self.edges[0]])
        elif len(edges)>2 or len(edges)==1:
            edges_loc =  np.array([edges, edges])
        else:
            edges_loc = edges

        if lmax is None:
            lmax = self.lmax

        self.data[component]['fs']['cl_template'][freq]['E'] = templates[0]
        self.data[component]['fs']['cl_template'][freq]['B'] = templates[1]
        cl_templ_binned = np.array([[np.mean(cl[int(bl):int(bu+1)])
            for bl, bu in zip(edges_loc[cli], edges_loc[cli][1:])]
            for cli, cl in enumerate(templates)])
        self.data[component]['fs']['cl_template_binned'][freq]['E'] = cl_templ_binned[0]
        self.data[component]['fs']['cl_template_binned'][freq]['B'] = cl_templ_binned[1]

        pslibE = ps.map2cl_binned(maps_mask, self.data[component]['fs']['cl_template'][freq]['E'], edges_loc[0], lmax)
        pslibB = ps.map2cl_binned(maps_mask, self.data[component]['fs']['cl_template'][freq]['B'], edges_loc[1], lmax)

        # self.data[component][nlevel]['cl_patch'][freq]['E']  = np.array([
        #     pslibE.map2cl(self.data[component]['fs']['map'][freq]['EB'][0]),
        #     pslibB.map2cl(self.data[component]['fs']['map'][freq]['EB'][1])
        # ])
        self.data[component][nlevel]['cl_patch'][freq]['E']  = pslibE.map2cl(self.data[component]['fs']['map'][freq]['EB'][0])
        self.data[component][nlevel]['cl_patch'][freq]['B']  = pslibB.map2cl(self.data[component]['fs']['map'][freq]['EB'][1])

        
    @log_on_start(logging.INFO, "calc_powerspectrum_unbinned_fromEB() started")
    @log_on_end(logging.INFO, "calc_powerspectrum_unbinned_fromEB() finished")   
    def calc_powerspectrum_unbinned_fromEB(self, maps_mask, freq='comb', component='cs', nlevel='fs', lmax=None):
        if lmax is None:
            lmax = self.lmax

        self.data[component][nlevel]['cl_patch'][freq]['E'] = ps.map2cl(self.data[component]['fs']['map'][freq]['EB'][0], maps_mask, lmax, lmax_mask=2*lmax)
        self.data[component][nlevel]['cl_patch'][freq]['B'] = ps.map2cl(self.data[component]['fs']['map'][freq]['EB'][1], maps_mask, lmax, lmax_mask=2*lmax)
        

    @log_on_start(logging.INFO, "calc_powerspectrum_binned_fromQU() started")
    @log_on_end(logging.INFO, "calc_powerspectrum_binned_fromQU() finished")   
    def calc_powerspectrum_binned_fromQU(self, maps_mask, templates, freq='comb', component='cs', nlevel='fs', edges=None, lmax=None):
        ## Template

        if edges is None:
            edges = self.edges

        if lmax is None:
            lmax = self.lmax

        self.data[component]['fs']['cl_template'][freq]['EB'] = templates
        cl_templ_binned = np.array([[np.mean(cl[int(bl):int(bu+1)])
            for bl, bu in zip(edges, edges[1:])]
            for cl in templates])
        self.data[component]['fs']['cl_template_binned'][freq]['EB'] = cl_templ_binned
        print(edges)
        pslibQU = ps.map2cl_spin_binned(maps_mask, 2, self.data[component]['fs']['cl_template'][freq]['EB'][0], self.data[component]['fs']['cl_template'][freq]['EB'][1], edges, lmax)
        self.data[component][nlevel]['cl_patch'][freq]['EB'] = np.array(pslibQU.map2cl(self.data[component]['fs']['map'][freq]['QU']))


    # @log_on_start(logging.INFO, "getfn_blm_lensc() started")
    # @log_on_end(logging.INFO, "getfn_blm_lensc() finished")
    def getfn_blm_lensc(self, simidx, it):
        '''Lenscarf output using Catherinas E and B maps'''
        # TODO this needs cleaner implementation via lambda
        # _libdir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        # return _libdir_iterator+fn(**params)

        ## TODO add QE lensing templates to CFS dir?
        if self.data_from_CFS:
            rootstr = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/lt_recons/')
            if it==12:
                if self.fg == '00':
                    return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '07':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '09':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            elif it==0:
                return '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/BLT/QE/blm_%04d_fg%02d_it0.npy'%(simidx, int(self.fg))   
        else:
            if self.libdir_iterators == 'overwrite':
                if it==12:
                    rootstr = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/lt_recons/')
                    if self.fg == '00':
                        return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                    elif self.fg == '07':
                        return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                    elif self.fg == '09':
                        return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif it==0:
                    return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(self.fg, simidx, simidx)    
            else:
                # TODO this belongs via config to l2d
                # TODO fn needs to be defined in l2d
                # TODO only QE it 0 doesn't exists because no modification is done to it. catching this. Can this be done better?
                if False:
                    # TODO this needs to be chosne by hand. Old mfvar naming don't depend on Nmf
                    if it == 0:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(it, it)
                    if self.dlm_mod_bool:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod.npy'%(it, it)
                    else:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(it, it)
                else:
                    if it == 0:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%03d.npy'%(it, it, self.Nmf)
                    if self.dlm_mod_bool:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod%03d.npy'%(it, it, self.Nmf)
                    else:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%03d.npy'%(it, it, self.Nmf)

            
    # @log_on_start(logging.INFO, "getfn_qumap_cs() started")
    # @log_on_end(logging.INFO, "getfn_qumap_cs() finished")
    def getfn_qumap_cs(self, simidx):

        '''Component separated polarisation maps lm, i.e. lenscarf input'''

        return self.sims.get_sim_pmap(simidx)


    @log_on_start(logging.INFO, "combine() started")
    @log_on_end(logging.INFO, "combine() finished")
    def combine_alms(self, freq2beam_fwhm=None, weights=None, pixelwindow=True, component='fg'):
        if weights is None:
            weights = self.data['weight']
        if freq2beam_fwhm is None:
            freq2beam_fwhm = self.ec.freq2beam
        nalm = int((self.lmax+1)*(self.lmax+2)/2)
        comb_E = np.zeros((nalm), dtype=np.complex128)
        comb_B = np.zeros((nalm), dtype=np.complex128)
        for freqi, freq in enumerate(self.ec.freqs):
            comb_E += hp.almxfl(hp.almxfl(self.data[component]['fs']['alm'][freq]['EB'][0], hp.gauss_beam(df.a2r(freq2beam_fwhm[freq]), self.lmax, pol = True)[:,1]), np.squeeze(weights[0,freqi,:self.lmax]))
            comb_B += hp.almxfl(hp.almxfl(self.data[component]['fs']['alm'][freq]['EB'][1], hp.gauss_beam(df.a2r(freq2beam_fwhm[freq]), self.lmax, pol = True)[:,2]), np.squeeze(weights[1,freqi,:self.lmax]))
            if pixelwindow:
                comb_E = np.nan_to_num(hp.almxfl(comb_E, 1/hp.pixwin(self.nside, pol=True)[0][:self.lmax]))
                comb_B = np.nan_to_num(hp.almxfl(comb_B, 1/hp.pixwin(self.nside, pol=True)[1][:self.lmax]))

        self.data[component]['fs']['map']['comb']['EB'] = np.array([comb_E, comb_B])
        return self.data[component]['fs']['map']['comb']['EB']
            
    
    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished")
    def collect_jobs(self):
        # TODO fill if needed
        jobs = []
        for idx in self.simidxs:
            jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        # TODO fill if needed
        return None

        
class OBD_builder():
    def __init__(self, OBD_model):
        self.__dict__.update(OBD_model.__dict__)


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished")
    def collect_jobs(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        jobs = [1]
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        # This fakes the collect/run structure, as bpl takes care of MPI 
        for job in self.jobs:
            bpl = template_bfilt(self.BMARG_LCUT, self.geom, int(os.environ.get('OMP_NUM_THREADS', 4)), _lib_dir=self.TEMP)
            if not os.path.exists(self.TEMP + '/tnit.npy'):
                bpl._get_rows_mpi(self.ninv_p[0], prefix='')
            mpi.barrier()
            if mpi.rank == 0:
                int(os.environ.get('OMP_NUM_THREADS', 32)) # TODO not sure if this resets anything..
                tnit = bpl._build_tnit()
                np.save(self.TEMP + '/tnit.npy', tnit)
                tniti = np.linalg.inv(tnit + np.diag((1. / (self.nlev_dep / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
                np.save(self.TEMP + '/tniti.npy', tniti)
                readme = '{}: This tniti has been created from user {} using lerepi/D.lensalot with the following settings: {}'.format(getpass.getuser(), datetime.date.today(), self.__dict__)
                np.save(self.TEMP + '/README.txt', readme)
                int(os.environ.get('OMP_NUM_THREADS', 8))  # TODO not sure if this resets anything..
        mpi.barrier()


class QE_lr():
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        if 'overwrite_libdir' in dlensalot_model.__dict__:
            pass
        else:
            self.overwrite_libdir = None
        if self.overwrite_libdir is None:
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
            self.mf = lambda simidx: self.get_meanfield(simidx)
            self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
            self.mf_resp = lambda: self.get_response_meanfield()
            self.wflm = lambda simidx: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lmax_unl, self.mmax_unl)
            self.R_unl = lambda: qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_unl, self.cls_unl,  {'e': self.fel_unl, 'b': self.fbl_unl, 't':self.ftl_unl}, lmax_qlm=self.lmax_qlm)[0]
        else:
            # TODO hack. Only want to access old s08b sim result lib and generate B wf
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'zb_terator_%s_%04d_nofg_OBD_solcond_3apr20'%(qe_key, simidx) + version)
      

    @log_on_start(logging.INFO, "collect_jobs() started: id={id}, overwrite_libdir={self.overwrite_libdir}")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self, id=''):
        if self.overwrite_libdir is None:
            mf_fname = os.path.join(self.TEMP, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(self.simidxs)))
            log.info('{} - {}'.format(self.simidxs, id))
            if os.path.isfile(mf_fname):
                # can safely skip QE. MF exists, so we know QE ran before
                self.jobs = []
            elif id == "None":
                self.jobs = []
            elif id == 'All':
                jobs = []
                for idx in self.simidxs_mf:
                    jobs.append(idx)
                self.jobs = jobs
            else:
                # TODO if id='', skip finished simindices
                jobs = []
                for idx in self.simidxs_mf:
                    jobs.append(idx)
                self.jobs = jobs
        else:
            # TODO hack. Only want to access old s08b sim result lib and generate B wf
            jobs = []
            for idx in self.simidxs_mf:
                jobs.append(idx)
            self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        if self.overwrite_libdir is None:
            for idx in self.jobs[mpi.rank::mpi.size]:
                log.info('{}/{}, Starting job {}'.format(mpi.rank,mpi.size,idx))
                # TODO this triggers the creation of all files for the MAP input, defined by the job array. 
                # MAP later needs the corresponding values separately via getter. Can I think of something better?
                self.get_sim_qlm(int(idx))
                self.get_response_meanfield()
                self.get_wflm(idx)
                self.get_R_unl()
                # self.get_B_wf(idx)
                log.info('{}/{}, finished job {}'.format(mpi.rank,mpi.size,idx))
            if len(self.jobs)>0:
                log.info('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                mpi.barrier()
                # Tunneling the meanfield-calculation, so only rank 0 calculates it. Otherwise,
                # some processes will try accessing it too fast, or calculate themselves, which results in
                # an io error
                log.info("Done waiting. Rank 0 going to calculate meanfield-file.. everyone else waiting.")
                if mpi.rank == 0:
                    self.get_meanfield(idx)
                    log.info("rank finsihed calculating meanfield-file.. everyone else waiting.")
                mpi.barrier()
                log.info("Starting mf-calc task")
            for idx in self.jobs[mpi.rank::mpi.size]:
                self.get_meanfield(idx)
                self.get_plm(idx)
            if len(self.jobs)>0:
                log.info('{} finished qe mf-calc tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                mpi.barrier()
                log.info('All ranks finished qe mf-calc tasks.')


    @log_on_start(logging.INFO, "get_sim_qlm() started")
    @log_on_end(logging.INFO, "get_sim_qlm() finished")
    def get_sim_qlm(self, idx):

        return self.qlms_dd.get_sim_qlm(self.k, int(idx))


    @log_on_start(logging.INFO, "get_B_wf() started")
    @log_on_end(logging.INFO, "get_B_wf() finished")    
    def get_B_wf(self, simidx):
        fn = self.libdir_iterators(self.k, simidx, self.version)+'/bwf_qe_%04d.npy'%simidx
        if not os.path.isdir(self.libdir_iterators(self.k, simidx, self.version)):
            os.makedirs(self.libdir_iterators(self.k, simidx, self.version))
        if os.path.isfile(fn):
            bwf = self.ivfs.get_sim_bmliklm(simidx)
        else:
            bwf = self.ivfs.get_sim_bmliklm(simidx)
            np.save(fn, bwf)

        return bwf


    @log_on_start(logging.INFO, "get_wflm() started")
    @log_on_end(logging.INFO, "get_wflm() finished")    
    def get_wflm(self, simidx):

        return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lmax_unl, self.mmax_unl)


    @log_on_start(logging.INFO, "get_R_unl() started")
    @log_on_end(logging.INFO, "get_R_unl() finished")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_unl, self.cls_unl,  {'e': self.fel_unl, 'b': self.fbl_unl, 't':self.ftl_unl}, lmax_qlm=self.lmax_qlm)[0]


    @log_on_start(logging.INFO, "get_meanfield() started")
    @log_on_end(logging.INFO, "get_meanfield() finished")
    def get_meanfield(self, simidx):
        if self.mfvar == None:
            mf = self.qlms_dd.get_sim_qlm_mf(self.k, self.simidxs_mf)
            if simidx in self.simidxs_mf:    
                mf = (mf - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
        else:
            mf = hp.read_alm(self.mfvar)
            if simidx in self.simidxs_mf:    
                mf = (mf - self.qlms_dd_mfvar.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))

        return mf


    @log_on_start(logging.INFO, "get_plm() started")
    @log_on_end(logging.INFO, "get_plm() finished")
    def get_plm(self, simidx, subtract_meanfield=True):
        lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        if not os.path.exists(lib_dir_iterator):
            os.makedirs(lib_dir_iterator)
        path_plm = opj(lib_dir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            if subtract_meanfield:
                plm -= self.mf(simidx)  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lmax_ivf, 'p', self.cls_len, self.cls_len, {'e': self.fel, 'b': self.fbl, 't':self.ftl}, lmax_qlm=self.lmax_qlm)[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
            plm = alm_copy(plm,  None, self.lmax_qlm, self.mmax_qlm)
            almxfl(plm, utils.cli(R), self.mmax_qlm, True) # Normalized QE
            almxfl(plm, WF, self.mmax_qlm, True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.mmax_qlm, True)
            np.save(path_plm, plm)

        return np.load(path_plm)


    # TODO this could be done before, inside c2d()
    @log_on_start(logging.INFO, "get_response_meanfield() started")
    @log_on_end(logging.INFO, "get_response_meanfield() finished")
    def get_response_meanfield(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.fel_unl, 'bb': self.fbl_unl}, self.lmax_ivf, self.lmax_qlm)[0]
        else:
            log.info('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.lmax_qlm + 1, dtype=float)

        return mf_resp


class MAP_lr():
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        # TODO Only needed to hand over to ith(). in c2d(), prepare an ith model for it
        self.dlensalot_model = dlensalot_model
        # TODO not entirely happy how QE dependence is put into MAP_lr but cannot think of anything better at the moment.
        self.qe = QE_lr(dlensalot_model)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)

        if self.iterator_typ in ['pertmf', 'constmf']:
            # TODO this is the interface to the D.lensalot iterators and connects 
            # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
            # but directly with cs_iterator, e.g. by adding visitor pattern to cs_iterator
            self.ith = iteration_handler.transformer(self.iterator_typ)
        elif self.iterator_typ in ['pertmf_new']:
            self.ith = cs_iterator.transformer(self.iterator_typ)


    @log_on_start(logging.INFO, "collect_jobs() start")
    @log_on_end(logging.INFO, "collect_jobs() finished")
    def collect_jobs(self):
        jobs = list(range(len(self.tasks)))
        for taski, task in enumerate(self.tasks):
            _jobs = []

            # TODO order of task list matters, but shouldn't
            if task == 'calc_phi':
                self.qe.collect_jobs()
                for idx in self.simidxs:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        _jobs.append(idx)

            elif task == 'calc_meanfield':
                self.qe.collect_jobs()
                # TODO need to make sure that all iterator wflms are calculated
                # either mpi.barrier(), or check all simindices TD(1)
                log.info("Waiting for all ranks to finish their task")
                mpi.barrier()
                _jobs.append(0)
                # check = True
                # for idx in range(self.Nmf):
                #     lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)   
                #     if rec.maxiterdone(lib_dir_iterator) < self.itmax:
                #         check = False
                #         break
                # if check:
                #     _jobs.append(0)

            elif task == 'calc_btemplate':
                self.qe.collect_jobs()
                # TODO making sure that all meanfields are available, but the mpi.barrier() is likely a too strong statement.
                log.info("Waiting for all ranks to finish their task")
                mpi.barrier()
                for idx in self.simidxs:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if "calc_phi" in self.tasks:
                        # assume that this is a new analysis, so rec.maxiterdone won't work. Could collect task jobs after finishing previous task run to improve this.
                        _jobs.append(idx)
                    else:
                        if rec.maxiterdone(lib_dir_iterator) >= self.itmax:
                            _jobs.append(idx)

            jobs[taski] = _jobs
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        for taski, task in enumerate(self.tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_phi':
                self.qe.run()
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if self.itmax >= 0 and rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                        itlib_iterator = itlib.get_iterator()
                        for it in range(self.itmax + 1):
                            log.info("using cg-tol = %.4e"%self.cg_tol(it))
                            log.info("using soltn_cond = %s"%self.soltn_cond(it))
                            itlib_iterator.chain_descr = self.chain_descr(self.lmax_unl, self.cg_tol(it))
                            itlib_iterator.soltn_cond = self.soltn_cond(it)
                            itlib_iterator.iterate(it, 'p')
                            log.info('{}, simidx {} done with it {}'.format(mpi.rank, idx, it))

            elif task == 'calc_meanfield':
                self.qe.run()
                # Must use mpi.barrier() before get_meanfields_it(), otherwise running into fileNotExist errors, as job splitting changes.
                # TODO could assign it0 mf to whoever is first, but then would need to check if all files exist and either time.sleep() or skip and let the next rank try?
                # TODO if TD(1) solved, replace np.arange() accordingly
                mpi.barrier()
                self.get_meanfields_it(np.arange(self.itmax+1), calc=True)
                mpi.barrier()

            elif task == 'calc_btemplate':
                self.qe.run()
                log.info('{}, task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    log.info("{}: start sim {}".format(mpi.rank, idx))
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                    itlib_iterator = itlib.get_iterator()
                    if self.dlm_mod_bool:
                        dlm_mod = self.get_meanfields_it(np.arange(self.itmax+1), calc=False)
                        dlm_mod = (dlm_mod - np.array(rec.load_plms(lib_dir_iterator, np.arange(self.itmax+1)))/self.Nmf) * self.Nmf/(self.Nmf - 1)
                    for it in range(0, self.itmax + 1):
                        if it <= rec.maxiterdone(lib_dir_iterator):
                            _dlm_mod = None if (it == 0 or self.dlm_mod_bool == False) else dlm_mod[it]
                            itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1, dlm_mod=_dlm_mod, calc=True, Nmf=self.Nmf, perturbative=self.btemplate_perturbative_lensremap)
                    log.info("{}: finished sim {}".format(mpi.rank, idx))


    @log_on_start(logging.INFO, "get_plm_it() started")
    @log_on_end(logging.INFO, "get_plm_it() finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), its)

        return plms


    @log_on_start(logging.INFO, "get_meanfield_it() started: it={it}")
    @log_on_end(logging.INFO, "get_meanfield_it() finished: it={it}")
    def get_meanfield_it(self, it, calc=False):
        # for mfvar runs, this returns the correct meanfields, as mfvar runs go into distinct itlib dirs.
        fn = opj(self.mf_dirname, 'mf%03d_it%03d.npy'%(self.Nmf, it))
        if not calc:
            if os.path.isfile(fn):
                mf = np.load(fn)
            else:
                mf = self.get_meanfield_it(self, it, calc=True)
        else:
            plm = rec.load_plms(self.libdir_iterators(self.k, self.simidxs[0], self.version), [0])[-1]
            mf = np.zeros_like(plm)
            for simidx in self.simidxs_mf:
                log.info("it {:02d}: adding sim {:03d}/{}".format(it, simidx, self.Nmf))
                mf += rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), [it])[-1]
            np.save(fn, mf/self.Nmf)

        return mf


    @log_on_start(logging.INFO, "get_meanfields_it() started")
    @log_on_end(logging.INFO, "get_meanfields_it() finished")
    def get_meanfields_it(self, its, calc=False):
        plm = rec.load_plms(self.libdir_iterators(self.k, self.simidxs[0], self.version), [0])[-1]
        mfs = np.zeros(shape=(len(its),*plm.shape), dtype=np.complex128)
        if calc==True:
            for iti, it in enumerate(its[mpi.rank::mpi.size]):
                mfs[iti] = self.get_meanfield_it(it, calc=calc)
            mpi.barrier()
        for iti, it in enumerate(its[mpi.rank::mpi.size]):
            mfs[iti] = self.get_meanfield_it(it, calc=False)

        return mfs


class Map_delenser():
    """Script for calculating delensed ILC and Blens spectra using precaulculated Btemplates as input.
    This is a combination of,
     * loading the right files,
     * delensing with the right Btemplates (QE, MAP),
     * choosing the right power spectrum calculation as in binning, masking, and templating
     * running across all jobs
    """

    def __init__(self, bmd_model):
        self.__dict__.update(bmd_model.__dict__)
        # TODO hack. Remove and think of a better way of including old data without existing config file
        if 'libdir_iterators' in bmd_model.__dict__:
            pass
        else:
            self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.lib = dict()

        
    @log_on_start(logging.INFO, "read_data() started")
    @log_on_end(logging.INFO, "read_data() finished")
    def read_data(self, dm, simids=None, edges=None, dlm_mod=False, dir_idx=0):
        bcl_cs = np.zeros(shape=(len(self.iterations)+2, len(self.nlevels), len(self.simids), len(self.edges)-1))
        bcl_L = np.zeros(shape=(len(self.iterations)+2, len(self.nlevels), len(self.simids), len(self.edges)-1))
        for simidx, simid in enumerate(self.simids):
            if dm.iteration.dlm_mod:
                data = np.load(self.TEMP + '/plotdata{}/{}'.format(self.vers_str,self.dirid) + '/ClBBwf_sim%04d_dlmmod_fg%2s_res2b3acm.npy'%(simid, dm.map_delensing.fg))
            else:
                data = np.load(self.TEMP + '/plotdata{}/{}'.format(self.vers_str,self.dirid) + '/ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(simid, dm.map_delensing.fg))
            # data =  np.load(dirroot_loc + '{}'.format(self.dirid) + '/Lenscarf_plotdata_ClBB_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg))
            bcl_L[0,:,simidx] = data[0][0]
            bcl_cs[0,:,simidx] = data[1][0]

            bcl_L[1,:,simidx] = data[0][1]
            bcl_cs[1,:,simidx] = data[1][1]

            for iti, it in enumerate(self.iterations):
                bcl_L[2+iti,:,simidx] = data[0][2+iti]
                bcl_cs[2+iti,:,simidx] = data[1][2+iti]

        print('dirid: {}'.format(self.dirid))

        return bcl_L, bcl_cs


    @log_on_start(logging.INFO, "read_data_v2() started")
    @log_on_end(logging.INFO, "read_data_v2() finished")
    def read_data_v2(self, edges_id=0):
        bcl_cs = np.zeros(shape=(len(self.its)+2, len(self.mask_ids), len(self.simidxs), len(self.edges[edges_id])-1))
        bcl_L = np.zeros(shape=(len(self.its)+2, len(self.mask_ids), len(self.simidxs), len(self.edges[edges_id])-1))
        
        print('Loading {} sims from {}'.format(len(self.simidxs),  '/'.join([f for f in self.file_op(0, self.fg, 0).split('/')[:-1]])))
        for simidxi, simidx in enumerate(self.simidxs):
            _file_op = self.file_op(simidx, self.fg, 0)
            data = np.load(_file_op)
            bcl_L[0,:,simidxi] = data[0][0]
            bcl_cs[0,:,simidxi] = data[1][0]

            bcl_L[1,:,simidxi] = data[0][1]
            bcl_cs[1,:,simidxi] = data[1][1]

            for iti, it in enumerate(self.its):
                bcl_L[2+iti,:,simidxi] = data[0][2+iti]
                bcl_cs[2+iti,:,simidxi] = data[1][2+iti]

        return bcl_L, bcl_cs


    # @log_on_start(logging.INFO, "getfn_blm_lensc() started")
    # @log_on_end(logging.INFO, "getfn_blm_lensc() finished")
    def getfn_blm_lensc(self, simidx, it):
        '''Lenscarf output using Catherinas E and B maps'''
        # TODO this needs cleaner implementation via lambda
        # _libdir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        # return _libdir_iterator+fn(**params)

        ## TODO add QE lensing templates to CFS dir?
        if self.data_from_CFS:
            rootstr = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/lt_recons/')
            if it==12:
                if self.fg == '00':
                    return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '07':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif self.fg == '09':
                    return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
            elif it==0:
                return '/global/cfs/cdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/BLT/QE/blm_%04d_fg%02d_it0.npy'%(simidx, int(self.fg))   
        else:
            if self.libdir_iterators == 'overwrite':
                if it==12:
                    rootstr = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/lt_recons/')
                    if self.fg == '00':
                        return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                    elif self.fg == '07':
                        return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                    elif self.fg == '09':
                        return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(self.fg), it, simidx)
                elif it==0:
                    return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(self.fg, simidx, simidx)    
            else:
                # TODO this belongs via config to l2d
                # TODO fn needs to be defined in l2d
                # TODO only QE it 0 doesn't exists because no modification is done to it. catching this. Can this be done better?
                if False:
                    # TODO this needs to be chosne by hand. Old mfvar naming don't depend on Nmf
                    if self.dlm_mod_bool:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod.npy'%(it, it)
                    else:
                        return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(it, it)
                else:
                    fn = self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024'%(it, it)
                    if self.dlm_mod_bool:
                        fn += '_dlmmod%03d'%(self.Nmf)
                    else:
                        fn += '%03d'%(self.Nmf)
                    if self.btemplate_perturbative_lensremap:
                        fn += 'perturbative'
                    return fn+'.npy'

            
    # @log_on_start(logging.INFO, "getfn_qumap_cs() started")
    # @log_on_end(logging.INFO, "getfn_qumap_cs() finished")
    def getfn_qumap_cs(self, simidx):

        '''Component separated polarisation maps lm, i.e. lenscarf input'''

        return self.sims.get_sim_pmap(simidx)


    # @log_on_start(logging.INFO, "get_B_wf() started")
    # @log_on_end(logging.INFO, "get_B_wf() finished")
    def get_B_wf(self, simidx):
        '''Component separated polarisation maps lm, i.e. lenscarf input'''
        # TODO this is a quickfix and works only for already existing bwflm's for 08bb
        bw_fn = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/bwf_qe_%04d.npy'%(self.fg,simidx,simidx)
        if os.path.isfile(bw_fn):

            return np.load(bw_fn)
        else:
            assert 0, "File {} doesn't exist".format(bw_fn)


    # @log_on_start(logging.INFO, "get_teblm_ffp10() started")
    # @log_on_end(logging.INFO, "get_teblm_ffp10() finished")
    def get_teblm_ffp10(self, simidx):
        '''Pure BB-lensing from ffp10''' 
        ret = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(simidx), lmax=self.lmax), self.transf)

        return ret


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO perhaps trigger calc of B-templates here, if needed
        jobs = []
        for idx in self.simidxs:
            # Overwriting test
            if self.libdir_iterators == 'overwrite':
                jobs.append(idx)
            else:
                lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                if rec.maxiterdone(lib_dir_iterator) >= self.its[-1]:
                    jobs.append(idx)
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        
        @log_on_start(logging.INFO, "_prepare_job() started")
        @log_on_end(logging.INFO, "_prepare_job() finished")
        def _prepare_job(edges=[]):
            ## choose by hand: either binmasks, or masks
            masktype = list(self.binmasks.keys())[0]
            if self.binning == 'binned':
                outputdata = np.zeros(shape=(2, 2+len(self.its), len(self.mask_ids), len(edges)-1))
                for mask_id, mask in self.binmasks[masktype].items():
                    self.lib.update({mask_id: self.cl_calc.map2cl_binned(mask, self.clc_templ[:self.lmax_mask], edges, self.lmax_mask)})

            elif self.binning == 'unbinned':
                for mask_id, mask in self.binmasks[masktype].items():
                    a = overwrite_anafast() if self.cl_calc == hp else masked_lib(mask, self.cl_calc, self.lmax, self.lmax_mask)
                    outputdata = np.zeros(shape=(2, 2+len(self.its), len(self.mask_ids), self.lmax+1))
                    self.lib.update({mask_id: a})

            return outputdata


        # @log_on_start(logging.INFO, "_build_basemaps() started")
        # @log_on_end(logging.INFO, "_build_basemaps() finished")
        def _build_basemaps(idx):
            if self.data_type == 'map':
                if self.data_field == 'qu':
                    map_cs = self.getfn_qumap_cs(idx)
                    eblm_cs = hp.map2alm_spin(map_cs*self.base_mask, 2, self.lmax)
                    bmap_cs = hp.alm2map(eblm_cs[1], self.nside)
                elif self.data_field == 'eb':
                    map_cs = self.getfn_qumap_cs(idx)
                    teblm_cs = hp.map2alm([np.zeros_like(map_cs[0]), *map_cs], lmax=self.lmax, pol=False)
                    bmap_cs = hp.alm2map(teblm_cs[2], self.nside)
            elif self.data_type == 'alm':
                if self.data_field == 'eb':
                    eblm_cs = self.getfn_qumap_cs(idx)
                    bmap_cs = hp.alm2map(eblm_cs[1], self.nside)
                elif self.data_field == 'qu':
                    log.error("I don't think you have qlms,ulms")
                    sys.exit()

            # TODO fiducial choice should happen at transformer
            blm_L = self.get_teblm_ffp10(idx)
            bmap_L = hp.alm2map(blm_L, self.nside)

            btemplm_QE = np.load(self.getfn_blm_lensc(idx, 0))
            btempmap_QE = hp.alm2map(btemplm_QE, nside=self.nside)

            return bmap_L, bmap_cs, btempmap_QE


        # @log_on_start(logging.INFO, "_build_Btemplate_MAP() started")
        # @log_on_end(logging.INFO, "_build_Btemplate_MAP() finished")
        def _build_Btemplate_MAP(idx):
            fns = [self.getfn_blm_lensc(idx, it) for it in self.its]
            btemplm_MAP = np.zeros(shape=(len(fns), *np.load(self.getfn_blm_lensc(idx, 0)).shape), dtype=np.complex128)
            for fni, fn in enumerate(fns):
                if fn.endswith('.npy'):
                    btemplm_MAP[fni] = np.array(np.load(fn))
                else:
                    btemplm_MAP[fni] = np.array(hp.read_alm(fn))   
            btempmap_MAP = np.array([hp.alm2map(btemplm_MAP[iti], nside=self.nside) for iti in range(len(self.its))])

            return btempmap_MAP


        @log_on_start(logging.INFO, "_delens() started")
        @log_on_end(logging.INFO, "_delens() finished")
        def _delens(bmap_L, bmap_cs, btempmap_QE, btempmap_MAP):
            for mask_idi, mask_id in enumerate(self.mask_ids):
                log.info("starting mask {}".format(mask_id))
                bcl_cs = self.lib[mask_id].map2cl(bmap_cs)
                bcl_L = self.lib[mask_id].map2cl(bmap_L)

                outputdata[0][0][mask_idi] = bcl_L
                outputdata[1][0][mask_idi] = bcl_cs

                btempcl_L_QE = self.lib[mask_id].map2cl(bmap_L-btempmap_QE)
                btempcl_cs_QE = self.lib[mask_id].map2cl(bmap_cs-btempmap_QE)

                outputdata[0][1][mask_idi] = btempcl_L_QE
                outputdata[1][1][mask_idi] = btempcl_cs_QE

                for iti, it in enumerate(self.its):
                    log.info("starting MAP delensing for iteration {}".format(it))
                    btempcl_L_MAP = self.lib[mask_id].map2cl(bmap_L-btempmap_MAP[iti])    
                    btempcl_cs_MAP = self.lib[mask_id].map2cl(bmap_cs-btempmap_MAP[iti])

                    outputdata[0][2+iti][mask_idi] = btempcl_L_MAP
                    outputdata[1][2+iti][mask_idi] = btempcl_cs_MAP

            return outputdata

        if self.jobs != []:
            if self.binning == 'binned':
                for edgesi, edges in enumerate(self.edges):
                    print(self.binning)
                    outputdata = _prepare_job(edges)
                    for idx in self.jobs[mpi.rank::mpi.size]:
                        _file_op = self.file_op(idx, self.fg, edgesi)
                        log.info('will store file at: {}'.format(_file_op))
                        bmap_L, bmap_cs, btempmap_QE = _build_basemaps(idx)
                        btempmap_MAP = _build_Btemplate_MAP(idx)
                        outputdata = _delens(bmap_L, bmap_cs, btempmap_QE, btempmap_MAP)
                        np.save(_file_op, outputdata)
            else:
                outputdata = _prepare_job()
                for idx in self.jobs[mpi.rank::mpi.size]:
                    _file_op = self.file_op(idx, self.fg, 0)
                    log.info('will store file at: {}'.format(_file_op))
                    bmap_L, bmap_cs, btempmap_QE = _build_basemaps(idx)
                    btempmap_MAP = _build_Btemplate_MAP(idx)
                    outputdata = _delens(bmap_L, bmap_cs, btempmap_QE, btempmap_MAP)
                    np.save(_file_op, outputdata)


class Inspector():
    def __init__(self, qe, model):

        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):

        assert 0, "Implement if needed"


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished")
    def run(self):

        assert 0, "Implement if needed"


class overwrite_anafast():

    def map2cl(self, *args, **kwargs):
        return hp.anafast(*args, **kwargs)


class masked_lib:

    def __init__(self, mask, cl_calc, lmax, lmax_mask):
        self.mask = mask
        self.cl_calc = cl_calc
        self.lmax = lmax
        self.lmax_mask = lmax_mask

    def map2cl(self, map):
        return self.cl_calc.map2cl(map, self.mask, self.lmax, self.lmax_mask)
