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
import copy

import numpy as np
import healpy as hp

from MSC import pospace as ps
from plancklens import utils, qresp
from plancklens.sims import planck2018_sims

from lenscarf.core import mpi
from lenscarf.lerepi.config.config_helper import data_functions as df
from lenscarf.utils_hp import almxfl, alm_copy
from lenscarf.iterators.statics import rec as rec
from lenscarf.iterators import iteration_handler
from lenscarf.opfilt.bmodes_ninv import template_bfilt



class Basejob():
    """
    Base class for all jobs, i.e. convenience functions go in here as they should be accessible from anywhere
    """
    def __str__(self):
        ## overwrite print to summarize dlensalot model
        _str = ''
        for key, val in self.__dict__.items():
            keylen = len(str(key))
            if type(val) in [list, np.ndarray, np.array, dict]:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(type(val))
            else:
                _str += '{}:'.format(key)+(20-keylen)*' '+'\t{}'.format(val)
            _str += '\n'
        return _str

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



    def load_plm_it(self, simidx, it):

        return None


    def load_mf_it(self, simidx, it, normalized=True):

        return None


    def get_blt_it(self, simidx, it):

        return None


    def get_ivf(self, simidx, it, field):

        return None


    def get_wf(self, simidx, it, field):

        return None
    

    def get_fiducial_sim(self, simidx, field):

        return None


class Notebook_interactor(Basejob):
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
        * see read_data_v2()
    '''
    def __init__(self, Interactor_model):
        self.__dict__.update(Interactor_model.__dict__)



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
                    buff = hp.read_map(self.fc.fns_syncdust.format(freq=freq, nside=self._sims.nside), field=(1,2))*mask*self.fc.map_fac
                else:
                    buff = hp.read_map(self.fc.fns_sync.format(freq=freq, nside=self._sims.nside, simidx=self.fc.simidx, fg_beamstring=self.fc.fg_beamstring[freq]), field=(1,2))*mask*self.fc.map_fac + hp.read_map(self.fc.fns_dust.format(freq=freq, nside=self._sims.nside, simidx=self.fc.simidx, fg_beamstring=self.fc.fg_beamstring[freq]), field=(1,2))*mask*self.fc.map_fac
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
        pslibQU = ps.map2cl_spin_binned(maps_mask, 2, self.data[component]['fs']['cl_template'][freq]['EB'][0], self.data[component]['fs']['cl_template'][freq]['EB'][1], edges, lmax)
        self.data[component][nlevel]['cl_patch'][freq]['EB'] = np.array(pslibQU.map2cl(self.data[component]['fs']['map'][freq]['QU']))


    # @log_on_start(logging.INFO, "getfn_blm_lensc() started")
    # @log_on_end(logging.INFO, "getfn_blm_lensc() finished")
    def getfn_blm_lensc(self, simidx, it, fn_splitsetsuffix=''):
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
            if it == 0:
                return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%s%03d.npy'%(it, it, self.Nmf, fn_splitsetsuffix)
            if self.dlm_mod_bool:
                return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024_dlmmod%s%03d.npy'%(it, it, self.Nmf, fn_splitsetsuffix)
            else:
                return self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024%s%03d.npy'%(it, it, self.Nmf, fn_splitsetsuffix)

            
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
                comb_E = np.nan_to_num(hp.almxfl(comb_E, 1/hp.pixwin(self._sims.nside, pol=True)[0][:self.lmax]))
                comb_B = np.nan_to_num(hp.almxfl(comb_B, 1/hp.pixwin(self._sims.nside, pol=True)[1][:self.lmax]))

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

        
class OBD_builder(Basejob):
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
            bpl = template_bfilt(self.lmin_b, self.geom, self.tr, _lib_dir=self.libdir)
            if not os.path.exists(self.libdir + '/tnit.npy'):
                bpl._get_rows_mpi(self.ninv_p, prefix='')
            mpi.barrier()
            if mpi.rank == 0:
                if not os.path.exists(self.libdir + '/tnit.npy'):
                    tnit = bpl._build_tnit()
                    np.save(self.libdir + '/tnit.npy', tnit)
                else:
                    tnit = np.load(self.libdir + '/tnit.npy')
                if not os.path.exists(self.libdir + '/tniti.npy'):
                    log.info('inverting')
                    tniti = np.linalg.inv(tnit + np.diag((1. / (self.nlev_dep / 180. / 60. * np.pi) ** 2) * np.ones(tnit.shape[0])))
                    np.save(self.libdir + '/tniti.npy', tniti)
                    readme = '{}: tniti.npy. created from user {} using lerepi/D.lensalot with the following settings: {}'.format(getpass.getuser(), datetime.date.today(), self.__dict__)
                    with open(self.libdir + '/README.txt', 'w') as f:
                        f.write(readme)
                else:
                    log.info('Matrix already created')
        mpi.barrier()


class QE_lr(Basejob):
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        self.dlensalot_model = dlensalot_model

        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
        self.mf = lambda simidx: self.get_meanfield(int(simidx))
        self.plm = lambda simidx: self.get_plm(simidx, self.QE_subtract_meanfield)
        self.wflm = lambda simidx: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])
        self.R_unl = lambda: qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl,  self.ftebl_unl, lmax_qlm=self.qe_lm_max_qlm[0])[0]

        # TODO only needed for get_blt(), as this is done by cs_iterator.. move 
        self.ith = iteration_handler.transformer('constmf')
      

    @log_on_start(logging.INFO, "collect_jobs() started: qe_tasks={qe_tasks}")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self, qe_tasks=None, recalc=False):

        # qe_tasks overwrites task-list and is needed if MAP lensrec calls QE lensrec
        _qe_tasks = self.qe_tasks if qe_tasks == None else qe_tasks

        jobs = list(range(len(_qe_tasks)))
        for taski, task in enumerate(_qe_tasks):
            _jobs = []

            ## Calculate realization-independent mf and store in qlms_dd dir

            if task == 'calc_meanfield':
                ## appending all as I trust plancklens to skip existing ivfs
                mf_fname = os.path.join(self.TEMP, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(self.simidxs_mf)))
                ## if meanfield exists, no need to calculate any of the qlms_dd
                if not os.path.isfile(mf_fname) or recalc:
                    ## if meanfield not yet done, collect all qlms_dd which haven't yet been calculated
                    for idx in self.simidxs_mf:
                        fname = os.path.join(self.qlms_dd.lib_dir, 'sim_%s_%04d.fits'%(self.k, idx) if idx != -1 else 'dat_%s.fits'%self.k)
                        if not os.path.isfile(fname) or recalc:
                            _jobs.append(int(idx))

            ## Calculate realization dependent phi, i.e. plm_it000.
            if task == 'calc_phi':
                mf_fname = os.path.join(self.TEMP, 'qlms_dd/simMF_k1%s_%s.fits' % (self.k, utils.mchash(self.simidxs_mf)))
                ## Skip if meanfield already calculated
                if not os.path.isfile(mf_fname) or recalc:
                    for idx in self.simidxs:
                        ## TODO skip all plms already calculated
                        ## If plm i missing, this task in run() creates all prereqs
                        _jobs.append(idx)

            ## Calculate B-lensing template
            if task == 'calc_blt':
                for idx in self.simidxs:
                    # TODO remove hardcoded fname generation in the next line, perhaps use cacher
                    fname = os.path.join(self.TEMP, '{}_sim{:04d}'.format(self.k, idx), 'btempl_p000_e000_lmax1024{}'.format(len(self.simidxs_mf)))
                    if self.blt_pert:
                        fname += 'perturbative'
                    fname += '.npy'
                    if not os.path.isfile(fname) or recalc:
                        ## TODO if prereq missing, how do I catch this if only calc_blt chosen?
                        ## Well, in principle user would have to choose the other tasks then.. so should be fine
                        _jobs.append(idx)

            jobs[taski] = _jobs
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self, task=None):
        ## task may be set from MAP lensrec, as MAP lensrec has prereqs to QE lensrec
        ## if None, then this is a normal QE lensrec call
        _tasks = self.qe_tasks if task is None else [task]
        for taski, task in enumerate(_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_meanfield':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    # In principle it is enough to calculate qlms. 
                    # self.get_plm(idx, self.QE_subtract_meanfield)
                    self.get_sim_qlm(int(idx))
                    # self.get_response_meanfield()
                    # self.get_wflm(idx)
                    # self.get_R_unl()
                    log.info('{}/{}, finished job {}'.format(mpi.rank,mpi.size,idx))
                if len(self.jobs[taski])>0:
                    log.info('{} finished qe ivfs tasks. Waiting for all ranks to start mf calculation'.format(mpi.rank))
                    mpi.barrier()
                    # Tunneling the meanfield-calculation, so only rank 0 calculates it. Otherwise,
                    # some processes will try accessing it too fast, or calculate themselves, which results in
                    # an io error
                    log.info("Done waiting. Rank 0 going to calculate meanfield-file.. everyone else waiting.")
                    if mpi.rank == 0:
                        self.get_meanfield(int(idx))
                        log.info("rank finished calculating meanfield-file.. everyone else waiting.")
                    mpi.barrier()

            if task == 'calc_phi':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.get_plm(idx, self.QE_subtract_meanfield)

            if task == 'calc_blt':
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.get_blt(idx)


    @log_on_start(logging.INFO, "get_sim_qlm({simidx}) started")
    @log_on_end(logging.INFO, "get_sim_qlm({simidx}) finished")
    def get_sim_qlm(self, simidx):

        return self.qlms_dd.get_sim_qlm(self.k, int(simidx))


    @log_on_start(logging.INFO, "get_B_wf({simidx}) started")
    @log_on_end(logging.INFO, "get_B_wf({simidx}) finished")    
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


    @log_on_start(logging.INFO, "get_wflm({simidx}) started")
    @log_on_end(logging.INFO, "get_wflm({simidx}) finished")    
    def get_wflm(self, simidx):

        return lambda: alm_copy(self.ivfs.get_sim_emliklm(simidx), None, self.lm_max_unl[0], self.lm_max_unl[1])


    @log_on_start(logging.INFO, "get_R_unl() started")
    @log_on_end(logging.INFO, "get_R_unl() finished")    
    def get_R_unl(self):

        return qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_unl, self.cls_unl, self.fteb_unl, lmax_qlm=self.qe_lm_max_qlm[0])[0]


    @log_on_start(logging.INFO, "get_meanfield({simidx}) started")
    @log_on_end(logging.INFO, "get_meanfield({simidx}) finished")
    def get_meanfield(self, simidx):
        ret = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, 0))
        if self.Nmf > 0:
            if self.mfvar == None:
                # TODO hack: plancklens needs to be less restrictive with type for simidx. hack for now
                ret = self.qlms_dd.get_sim_qlm_mf(self.k, [int(simidx_mf) for simidx_mf in self.simidxs_mf])
                if simidx in self.simidxs_mf:    
                    ret = (ret - self.qlms_dd.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
            else:
                ret = hp.read_alm(self.mfvar)
                if simidx in self.simidxs_mf:    
                    ret = (ret - self.qlms_dd_mfvar.get_sim_qlm(self.k, int(simidx)) / self.Nmf) * (self.Nmf / (self.Nmf - 1))
            return ret
        return ret
        


    @log_on_start(logging.INFO, "get_plm({simidx}) started")
    @log_on_end(logging.INFO, "get_plm({simidx}) finished")
    def get_plm(self, simidx, subtract_meanfield=True):
        lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        if not os.path.exists(lib_dir_iterator):
            os.makedirs(lib_dir_iterator)
        path_plm = opj(lib_dir_iterator, 'phi_plm_it000.npy')
        if not os.path.exists(path_plm):
            plm  = self.qlms_dd.get_sim_qlm(self.k, int(simidx))  #Unormalized quadratic estimate:
            if subtract_meanfield and self.version != 'noMF':
                plm -= self.mf(int(simidx))  # MF-subtracted unnormalized QE
            R = qresp.get_response(self.k, self.lm_max_ivf[0], self.k[0], self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.qe_lm_max_qlm[0])[0]
            # Isotropic Wiener-filter (here assuming for simplicity N0 ~ 1/R)
            WF = self.cpp * utils.cli(self.cpp + utils.cli(R))
            plm = alm_copy(plm,  None, self.qe_lm_max_qlm[0], self.qe_lm_max_qlm[1])
            almxfl(plm, utils.cli(R), self.qe_lm_max_qlm[1], True) # Normalized QE
            almxfl(plm, WF, self.qe_lm_max_qlm[1], True) # Wiener-filter QE
            almxfl(plm, self.cpp > 0, self.qe_lm_max_qlm[1], True)
            np.save(path_plm, plm)
            
        return np.load(path_plm)


    # TODO this could be done before, inside c2d()
    @log_on_start(logging.INFO, "get_response_meanfield() started")
    @log_on_end(logging.INFO, "get_response_meanfield() finished")
    def get_response_meanfield(self):
        if self.k in ['p_p'] and not 'noRespMF' in self.version:
            mf_resp = qresp.get_mf_resp(self.k, self.cls_unl, {'ee': self.ftebl_len['e'], 'bb': self.ftebl_len['b']}, self.lm_max_ivf[0], self.qe_lm_max_qlm[0])[0]
        else:
            log.info('*** mf_resp not implemented for key ' + self.k, ', setting it to zero')
            mf_resp = np.zeros(self.qe_lm_max_qlm[0] + 1, dtype=float)

        return mf_resp


    def get_meanfield_normalized(self, simidx):

        mf_QE = copy.deepcopy(self.get_meanfield(simidx))
        cpp_loc = self.cpp
        R = qresp.get_response(self.k, self.lm_max_ivf[0], 'p', self.cls_len, self.cls_len, self.ftebl_len, lmax_qlm=self.qe_lm_max_qlm[0])[0]
        WF = cpp_loc * utils.cli(cpp_loc + utils.cli(R))
        almxfl(mf_QE, utils.cli(R), self.qe_lm_max_qlm[1], True) # Normalized QE
        almxfl(mf_QE, WF, self.qe_lm_max_qlm[1], True) # Wiener-filter QE
        almxfl(mf_QE, cpp_loc > 0, self.qe_lm_max_qlm[1], True)

        return mf_QE


    @log_on_start(logging.INFO, "get_blt({simidx}) started")
    @log_on_end(logging.INFO, "get_blt({simidx}) finished")
    def get_blt(self, simidx):
        itlib = self.ith(self, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)
        itlib_iterator = itlib.get_iterator()
        ## For QE, dlm_mod by construction doesn't do anything, because mean-field had already been subtracted from plm and we don't want to repeat that.
        ## But we are going to store a new file anyway.
        dlm_mod = np.zeros_like(self.qlms_dd.get_sim_qlm(self.k, int(simidx)))
        return itlib_iterator.get_template_blm(0, 0, lmaxb=1024, lmin_plm=1, dlm_mod=dlm_mod, perturbative=self.blt_pert)


class MAP_lr(Basejob):
    def __init__(self, dlensalot_model):
        self.__dict__.update(dlensalot_model.__dict__)
        # TODO Only needed to hand over to ith(). in c2d(), prepare an ith model for it
        self.dlensalot_model = dlensalot_model
        # TODO not entirely happy how QE dependence is put into MAP_lr but cannot think of anything better at the moment.
        self.qe = QE_lr(dlensalot_model)
        self.libdir_iterators = lambda qe_key, simidx, version: opj(self.TEMP,'%s_sim%04d'%(qe_key, simidx) + version)

        # if self.iterator_typ in ['pertmf', 'constmf', 'fastWF']:
        # TODO this is the interface to the D.lensalot iterators and connects 
        # to lerepi. Could be simplified, s.t. interfacing happens without the iteration_handler
        # but directly with cs_iterator, e.g. by adding visitor pattern
        self.ith = iteration_handler.transformer(self.iterator_typ)


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        jobs = list(range(len(self.it_tasks)))
        # TODO order of task list matters, but shouldn't
        for taski, task in enumerate(self.it_tasks):
            _jobs = []

            if task == 'calc_phi':
                ## Here I only want to calculate files not calculated before, and only for the it job tasks.
                ## i.e. if no blt task in iterator job, then no blt task in QE job 
                self.qe.collect_jobs(task, recalc=False)
                for idx in self.simidxs:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    ## Skip if itmax phi is calculated
                    if rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        _jobs.append(idx)

            ## Calculate realization independent meanfields up to iteration itmax
            ## prereq: plms exist for itmax. maxiterdone won't work if calc_phi in task list
            elif task == 'calc_meanfield':
                ## Don't necessarily depend on QE meanfield being calculated, remove next line
                self.qe.collect_jobs(task, recalc=False)
                # TODO need to make sure that all iterator wflms are calculated
                # either mpi.barrier(), or check all simindices TD(1)

                for idx in self.simidxs_mf:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if "calc_phi" in self.it_tasks:
                        _jobs.append(0)
                    elif rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        _jobs.append(0)
                log.info("Waiting for all ranks to finish their task")
                mpi.barrier()

            elif task == 'calc_blt':
                # TODO making sure that all meanfields are available, but the mpi.barrier() is likely a too strong statement.
                log.info("Waiting for all ranks to finish their task")
                mpi.barrier()
                for idx in self.simidxs:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if "calc_phi" in self.it_tasks:
                        # assume that this is a new analysis, so rec.maxiterdone won't work. Could collect task jobs after finishing previous task run to improve this.
                        _jobs.append(idx)
                    elif rec.maxiterdone(lib_dir_iterator) >= self.itmax:
                        _jobs.append(idx)
                    else:
                        log.info("Nothing to compute, as maxiterdone:{} < itermax:{}".format(rec.maxiterdone(lib_dir_iterator), self.itmax))

            jobs[taski] = _jobs
        self.jobs = jobs


    @log_on_start(logging.INFO, "run() started")
    @log_on_end(logging.INFO, "run() finished")
    def run(self):
        for taski, task in enumerate(self.it_tasks):
            log.info('{}, task {} started'.format(mpi.rank, task))

            if task == 'calc_phi':
                self.qe.run(task=task)
                for idx in self.jobs[taski][mpi.rank::mpi.size]:
                    lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
                    if self.itmax >= 0 and rec.maxiterdone(lib_dir_iterator) < self.itmax:
                        itlib = self.ith(self.qe, self.k, idx, self.version, self.libdir_iterators, self.dlensalot_model)
                        itlib_iterator = itlib.get_iterator()
                        for it in range(self.itmax + 1):
                            log.info("using cg-tol = %.4e"%self.it_cg_tol(it))
                            log.info("using soltn_cond = %s"%self.soltn_cond(it))
                            itlib_iterator.chain_descr = self.it_chain_descr(self.lm_max_unl[0], self.it_cg_tol(it))
                            itlib_iterator.soltn_cond = self.soltn_cond(it)
                            itlib_iterator.iterate(it, 'p')
                            log.info('{}, simidx {} done with it {}'.format(mpi.rank, idx, it))

            if task == 'calc_meanfield':
                self.qe.run(task=task)
                # Must use mpi.barrier() before get_meanfields_it(), otherwise running into fileNotExist errors, as job splitting changes.
                # TODO could assign it0 mf to whoever is first, but then would need to check if all files exist and either time.sleep() or skip and let the next rank try?
                # TODO if TD(1) solved, replace np.arange() accordingly
                mpi.barrier()
                self.get_meanfields_it(np.arange(self.itmax+1), calc=True)
                mpi.barrier()

            if task == 'calc_blt':
                log.info('{}, task {} started, jobs: {}'.format(mpi.rank, task, self.jobs[taski]))
                for simidx in self.jobs[taski][mpi.rank::mpi.size]:
                    self.lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
                    self.itlib = self.ith(self.qe, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)
                    self.itlib_iterator = self.itlib.get_iterator()
                    self.get_blt_it(simidx, self.itmax)


    @log_on_start(logging.INFO, "get_plm_it({simidx}, {its}) started")
    @log_on_end(logging.INFO, "get_plm_it({simidx}, {its}) finished")
    def get_plm_it(self, simidx, its):

        plms = rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), its)

        return plms


    @log_on_start(logging.INFO, "get_meanfield_it({it}) started")
    @log_on_end(logging.INFO, "get_meanfield_it({it}) finished")
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
                log.info("it {:02d}: adding sim {:03d}/{}".format(it, simidx, self.Nmf-1))
                mf += rec.load_plms(self.libdir_iterators(self.k, simidx, self.version), [it])[-1]
            np.save(fn, mf/self.Nmf)

        return mf


    @log_on_start(logging.INFO, "get_meanfields_it({its}) started")
    @log_on_end(logging.INFO, "get_meanfields_it({its}) finished")
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


    @log_on_start(logging.INFO, "get_blt_it({simidx}, {it}) started")
    @log_on_end(logging.INFO, "get_blt_it({simidx}, {it}) finished")
    def get_blt_it(self, simidx, it):
        if 'itlib' not in self.__dict__:
            self.itlib = self.ith(self.qe, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)
            self.itlib_iterator = self.itlib.get_iterator()
        if simidx != self.itlib.simidx:
            self.itlib = self.ith(self.qe, self.k, simidx, self.version, self.libdir_iterators, self.dlensalot_model)
            self.itlib_iterator = self.itlib.get_iterator()

        self.lib_dir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        dlm_mod = np.zeros_like(rec.load_plms(self.lib_dir_iterator, [0])[0])
        if self.dlm_mod_bool and it>0 and it<=rec.maxiterdone(self.lib_dir_iterator):
            dlm_mod = self.get_meanfields_it([it], calc=False)
            if simidx in self.simidxs_mf:
                dlm_mod = (dlm_mod - np.array(rec.load_plms(self.lib_dir_iterator, [it]))/self.Nmf) * self.Nmf/(self.Nmf - 1)
        if it>0 and it<=rec.maxiterdone(self.lib_dir_iterator):
            return self.itlib_iterator.get_template_blm(it, it, lmaxb=1024, lmin_plm=1, dlm_mod=dlm_mod, perturbative=False)
        elif it==0:
            return self.itlib_iterator.get_template_blm(0, 0, lmaxb=1024, lmin_plm=1, dlm_mod=dlm_mod, perturbative=self.blt_pert)


class Map_delenser(Basejob):
    """Script for calculating delensed ILC and Blens spectra using precaulculated Btemplates as input.
    This is a combination of,
     * loading the right files,
     * delensing with the right Btemplates (QE, MAP),
     * choosing the right power spectrum calculation as in binning, masking, and templating
     * running across all jobs
    """

    def __init__(self, bmd_model):
        self.__dict__.update(bmd_model.__dict__)
        self.lib = dict()
        if False:
            self.bcl_L, self.bcl_cs  = self.read_data_v2(edges_id=0)
        # self.bcl_L = np.array([b[0] for b in self.bcls])
        # self.bcl_cs = np.array([b[1] for b in self.bcls])

    def load_bcl(self):
        self.bcl_L, self.bcl_cs  = self.read_data_v2(edges_id=0)

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
    def getfn_blm_lensc(self, simidx, it, fn_splitsetsuffix=''):
        # TODO this needs cleaner implementation via lambda
        # _libdir_iterator = self.libdir_iterators(self.k, simidx, self.version)
        # return _libdir_iterator+fn(**params)

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
            # TODO this belongs via config to l2d
            # TODO fn needs to be defined in l2d
            fn = self.libdir_iterators(self.k, simidx, self.version)+'/wflms/btempl_p%03d_e%03d_lmax1024'%(it, it)
            if self.dlm_mod_bool:
                fn += '_dlmmod%s%03d'%(fn_splitsetsuffix, self.Nmf)
            else:
                fn += '%s%03d'%(fn_splitsetsuffix, self.Nmf)
            if self.blt_pert:
                fn += 'perturbative'
                
            return fn+'.npy'

            
    # @log_on_start(logging.INFO, "getfn_qumap_cs() started")
    # @log_on_end(logging.INFO, "getfn_qumap_cs() finished")
    def getfn_qumap_cs(self, simidx):

        '''Component separated polarisation maps lm, i.e. lenscarf input'''

        return self.sims.get_sim_pmap(simidx)



    # @log_on_start(logging.INFO, "get_teblm_ffp10() started")
    # @log_on_end(logging.INFO, "get_teblm_ffp10() finished")
    def get_teblm_ffp10(self, simidx):
        '''Pure BB-lensing from ffp10''' 
        ret = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(simidx), lmax=self.lmax), self.transf)

        return ret


    @log_on_start(logging.INFO, "collect_jobs() started")
    @log_on_end(logging.INFO, "collect_jobs() finished: jobs={self.jobs}")
    def collect_jobs(self):
        # TODO a valid job is any requested job, as BLTs may also be on CFS
        jobs = []
        for idx in self.simidxs:
            # lib_dir_iterator = self.libdir_iterators(self.k, idx, self.version)
            # if rec.maxiterdone(lib_dir_iterator) >= self.its[-1]:
            #     jobs.append(idx)
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
                    # eblm_cs = hp.map2alm_spin(map_cs*self.base_mask, 2, self.lmax)
                    # bmap_cs = hp.alm2map(eblm_cs[1], self.._sims.nside)
                elif self.data_field == 'eb':
                    map_cs = self.getfn_qumap_cs(idx)
                    # teblm_cs = hp.map2alm([np.zeros_like(map_cs[0]), *map_cs], lmax=self.lmax, pol=False)
                    # bmap_cs = hp.alm2map(teblm_cs[2], self.._sims.nside)
            elif self.data_type == 'alm':
                if self.data_field == 'eb':
                    eblm_cs = self.getfn_qumap_cs(idx)
                    # bmap_cs = hp.alm2map(eblm_cs[1], self.._sims.nside)
                elif self.data_field == 'qu':
                    log.error("I don't think you have qlms,ulms")
                    sys.exit()

            # TODO fiducial choice should happen at transformer
            blm_L = self.get_teblm_ffp10(idx)
            bmap_L = hp.alm2map(blm_L, self._sims.nside)

            if self.calc_via_MFsplitset:
                bltlm_QE1 = np.load(self.getfn_blm_lensc(idx, 0, 'set1'))
                blt_QE1 = hp.alm2map(bltlm_QE1, nside=self._sims.nside)
                bltlm_QE2 = np.load(self.getfn_blm_lensc(idx, 0, 'set2'))
                blt_QE2 = hp.alm2map(bltlm_QE2, nside=self._sims.nside)
            else:
                bltlm_QE1 = np.load(self.getfn_blm_lensc(idx, 0))
                blt_QE1 = hp.alm2map(bltlm_QE1, nside=self._sims.nside)
                blt_QE2 = np.copy(blt_QE1)
            # bmap_cs
            return bmap_L, np.zeros_like(bmap_L), blt_QE1, blt_QE2


        # @log_on_start(logging.INFO, "_build_Btemplate_MAP() started")
        # @log_on_end(logging.INFO, "_build_Btemplate_MAP() finished")
        def _build_Btemplate_MAP(idx):
            if self.calc_via_MFsplitset:
                fns = [self.getfn_blm_lensc(idx, it, 'set1') for it in self.its]
                bltlm_MAP = np.zeros(shape=(len(fns), *np.load(self.getfn_blm_lensc(idx, 0, 'set1')).shape), dtype=np.complex128)
                for fni, fn in enumerate(fns):
                    if fn.endswith('.npy'):
                        bltlm_MAP[fni] = np.array(np.load(fn))
                    else:
                        bltlm_MAP[fni] = np.array(hp.read_alm(fn))   
                blt_MAP1 = np.array([hp.alm2map(bltlm_MAP[iti], nside=self._sims.nside) for iti in range(len(self.its))])
                
                fns = [self.getfn_blm_lensc(idx, it, 'set2') for it in self.its]
                bltlm_MAP = np.zeros(shape=(len(fns), *np.load(self.getfn_blm_lensc(idx, 0, 'set2')).shape), dtype=np.complex128)
                for fni, fn in enumerate(fns):
                    if fn.endswith('.npy'):
                        bltlm_MAP[fni] = np.array(np.load(fn))
                    else:
                        bltlm_MAP[fni] = np.array(hp.read_alm(fn))   
                blt_MAP2 = np.array([hp.alm2map(bltlm_MAP[iti], nside=self._sims.nside) for iti in range(len(self.its))])
            else:
                fns = [self.getfn_blm_lensc(idx, it) for it in self.its]

                bltlm_MAP = np.zeros(shape=(len(fns), *np.load(self.getfn_blm_lensc(idx, 0)).shape), dtype=np.complex128)
                for fni, fn in enumerate(fns):
                    if fn.endswith('.npy'):
                        bltlm_MAP[fni] = np.array(np.load(fn))
                    else:
                        bltlm_MAP[fni] = np.array(hp.read_alm(fn))   
                blt_MAP1 = np.array([hp.alm2map(bltlm_MAP[iti], nside=self._sims.nside) for iti in range(len(self.its))])
                blt_MAP2 = np.copy(blt_MAP1)

            return blt_MAP1, blt_MAP2

                
        @log_on_start(logging.INFO, "_delens() started")
        @log_on_end(logging.INFO, "_delens() finished")
        def _delens(bmap_L, bmap_cs, blt_QE1, blt_MAP1, blt_QE2=None, blt_MAP2=None):
            if blt_QE2 is None:
                blt_QE2 = np.copy(blt_QE1)
            if blt_MAP2 is None:
                blt_MAP2 = np.copy(blt_MAP1)

            for mask_idi, mask_id in enumerate(self.mask_ids):
                log.info("starting mask {}".format(mask_id))
                
                bcl_L = self.lib[mask_id].map2cl(bmap_L)
                # bcl_cs = self.lib[mask_id].map2cl(bmap_cs)

                outputdata[0][0][mask_idi] = bcl_L
                # outputdata[1][0][mask_idi] = bcl_cs

                blt_L_QE = self.lib[mask_id].map2cl(bmap_L-blt_QE1, bmap_L-blt_QE2)
                # btempcl_cs_QE = self.lib[mask_id].map2cl(bmap_cs-btempmap_QE)

                outputdata[0][1][mask_idi] = blt_L_QE
                # outputdata[1][1][mask_idi] = btempcl_cs_QE

                for iti, it in enumerate(self.its):
                    log.info("starting MAP delensing for iteration {}".format(it))
                    blt_L_MAP = self.lib[mask_id].map2cl(bmap_L-blt_MAP1[iti], bmap_L-blt_MAP2[iti])    
                    # btempcl_cs_MAP = self.lib[mask_id].map2cl(bmap_cs-blt_MAP[iti])

                    outputdata[0][2+iti][mask_idi] = blt_L_MAP
                    # outputdata[1][2+iti][mask_idi] = btempcl_cs_MAP

            return outputdata

        if self.jobs != []:
            if self.binning == 'binned':
                for edgesi, edges in enumerate(self.edges):
                    outputdata = _prepare_job(edges)
                    for idx in self.jobs[mpi.rank::mpi.size]:
                        _file_op = self.file_op(idx, self.fg, edgesi)
                        log.info('will store file at: {}'.format(_file_op))
                        bmap_L, bmap_cs, blt_QE1, blt_QE2 = _build_basemaps(idx)
                        if self.subtract_mblt:
                            if self.calc_via_mbltsplitset:
                                simidxs_bltmean1 = np.arange(0,int(self.Nmblt/2))
                                simidxs_bltmean2 = np.arange(int(self.Nmblt/2), self.Nmblt)
                                if self.calc_via_MFsplitset:
                                    mblt_QE1 = hp.alm2map(np.mean([np.load(self.getfn_blm_lensc(simidx, 0, 'set1')) for simidx in simidxs_bltmean1 if simidx not in [idx]], axis=0), nside=self._sims.nside)
                                    mblt_QE2 = hp.alm2map(np.mean([np.load(self.getfn_blm_lensc(simidx, 0, 'set2')) for simidx in simidxs_bltmean2 if simidx not in [idx]], axis=0), nside=self._sims.nside)
                                else:
                                    mblt_QE1 = hp.alm2map(np.mean([np.load(self.getfn_blm_lensc(simidx, 0)) for simidx in simidxs_bltmean1 if simidx not in [idx]], axis=0), nside=self._sims.nside)
                                    mblt_QE2 = hp.alm2map(np.mean([np.load(self.getfn_blm_lensc(simidx, 0)) for simidx in simidxs_bltmean2 if simidx not in [idx]], axis=0), nside=self._sims.nside)
                            else:
                                mblt_QE1 = hp.alm2map(np.mean([np.load(self.getfn_blm_lensc(simidx, 0)) for simidx in np.arange(0,self.Nmblt) if simidx not in [idx]], axis=0), nside=self._sims.nside)
                                mblt_QE2 = np.copy(mblt_QE1)
                        else:
                            mblt_QE1 = 0
                            mblt_QE2 = 0
                        blt_MAP1, blt_MAP2 = _build_Btemplate_MAP(idx)
                        if self.subtract_mblt:
                            if self.calc_via_mbltsplitset:
                                simidxs_bltmean1 = np.arange(0,int(self.Nmblt/2))
                                simidxs_bltmean2 = np.arange(int(self.Nmblt/2), self.Nmblt)
                                if self.calc_via_MFsplitset:
                                    buff = np.mean([[np.load(self.getfn_blm_lensc(simidx, it, 'set1')) for simidx in simidxs_bltmean1 if simidx not in [idx]] for it in self.its], axis=1)
                                    mblt_MAP1 = np.array([hp.alm2map(buff[iti], nside=self._sims.nside) for iti, it in enumerate(self.its)])
                                    buff = np.mean([[np.load(self.getfn_blm_lensc(simidx, it, 'set2')) for simidx in simidxs_bltmean2 if simidx not in [idx]] for it in self.its], axis=1)
                                    mblt_MAP2 = np.array([hp.alm2map(buff[iti], nside=self._sims.nside) for iti, it in enumerate(self.its)])
                                else:
                                    buff = np.mean([[np.load(self.getfn_blm_lensc(simidx, it)) for simidx in simidxs_bltmean1 if simidx not in [idx]] for it in self.its], axis=1)
                                    mblt_MAP1 = np.array([hp.alm2map(buff[iti], nside=self._sims.nside) for iti, it in enumerate(self.its)])
                                    buff = np.mean([[np.load(self.getfn_blm_lensc(simidx, it)) for simidx in simidxs_bltmean2 if simidx not in [idx]] for it in self.its], axis=1)
                                    mblt_MAP2 = np.array([hp.alm2map(buff[iti], nside=self._sims.nside) for iti, it in enumerate(self.its)])
                            else:
                                buff = np.mean([[np.load(self.getfn_blm_lensc(simidx, it)) for simidx in np.arange(0,self.Nmf) if simidx not in [idx]] for it in self.its], axis=1)
                                mblt_MAP1 = np.array([hp.alm2map(buff[iti], nside=self._sims.nside) for iti, it in enumerate(self.its)])
                                mblt_MAP2 = np.copy(mblt_MAP1)
                        else:
                            mblt_MAP1 = 0
                            mblt_MAP2 = 0
                        outputdata = _delens(bmap_L, bmap_cs, blt_QE1-mblt_QE1, blt_MAP1-mblt_MAP1, blt_QE2-mblt_QE2, blt_MAP2-mblt_MAP2)
                        np.save(_file_op, outputdata)
            else:
                if self.subtract_mblt:
                    log.error("Implement if needed")
                    sys.exit()
                outputdata = _prepare_job()
                for idx in self.jobs[mpi.rank::mpi.size]:
                    _file_op = self.file_op(idx, self.fg, 0)
                    log.info('will store file at: {}'.format(_file_op))
                    bmap_L, bmap_cs, blt_QE, blt_QE2 = _build_basemaps(idx)
                    blt_MAP, blt_MAP2 = _build_Btemplate_MAP(idx)
                    outputdata = _delens(bmap_L, bmap_cs, blt_QE, blt_QE2, blt_MAP, blt_MAP2)
                    np.save(_file_op, outputdata)



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