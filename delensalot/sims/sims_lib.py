"""sims/sims_lib.py: library for collecting and handling simulations. Eventually, delensalot needs a get_sim_pmap(), which are the maps of the observed sky. But data may come from any of the above. So this module allows to start analysis with,
    cls,
    alms_unl
    alm_len + noise
    obs_sky
and is a mapper between them, so that get_sim_pmap()  

"""

import os
from os.path import join as opj
import numpy as np, healpy as hp

import logging
log = logging.getLogger(__name__)

import lenspyx
from plancklens.sims import phas
from delensalot.core import cachers



class iso_white_noise:

    def __init__(self, nlev_p, nside, lib_dir=None, fnsQ=None, fnsU=None, lib_dir_phas=None):
        self.lib_dir = lib_dir
        if lib_dir is None:        
            self.nlev_p = nlev_p
            self.nside = nside
            self.pix_lib_phas = phas.pix_lib_phas(lib_dir_phas, 3, (hp.nside2npix(nside),))
        else:
            self.fnsQ = fnsQ
            self.fnsU = fnsU  

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher
        # TODO same decisison-tree as in Xobs: if noise maps are stored somewhere, grab them. Otherwise, simulate noise


    def get_sim_qnoise(self, simidx):
        fnQ = 'qnoise_{}'.format(simidx)
        if not self.cacher.is_cached(fnQ):
            if self.lib_dir is None:
                vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
                return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=1)
            else:
                if self.fnsQ.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Qnoise = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                elif self.fnsQ.format(simidx).endswith('fits'):
                    Qnoise = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
            self.cacher.cache(fnQ, np.array(Qnoise))
        return self.cacher.load(fnQ)


    def get_sim_unoise(self, simidx):
        fnU = 'unoise_{}'.format(simidx)
        if not self.cacher.is_cached(fnU):
            if self.lib_dir is None:
                vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
                return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=2)
            else:
                if self.fnsU.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Unoise = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                elif self.fnsU.format(simidx).endswith('fits'):
                    Unoise = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fnU, np.array(Unoise))
        return self.cacher.load(fnU)


class Cls:
    def __init__(self, nside, lmax, lib_dir=None, fns=None, CAMB_file=None, simidxs=None):
        # TODO same decisison-tree as in Xobs: if unlensed Cls are stored somewhere, grab them. Otherwise, simulate lensed maps using lenspyx
        self.nside = nside
        self.fns = fns
        self.lmax = lmax
        self.lib_dir = lib_dir
        self.CAMB_file = CAMB_file
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_TEBunl(self, simidx):
        fn = 'cls_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                ClT, ClE, ClB = self.from_cambfile()
            else:
                ClT = hp.read_map(opj(self.lib_dir, self.fns['T'].format(simidx)))
                ClE = hp.read_map(opj(self.lib_dir, self.fns['E'].format(simidx)))
                ClB = hp.read_map(opj(self.lib_dir, self.fns['B'].format(simidx)))
            self.cacher.cache(fn, np.array([ClT, ClE, ClB]))
        return self.cacher.load(fn)   
        
    
    def get_sim_clphi(self, simidx):
        fn = 'clphi_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunl, Uunl = self.cl2alm(self.cls_lib.get_TEBunl())
            else:
                if self.fnsQ.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Uunl = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                elif self.fnsQ.format(simidx).endswith('fits'):
                    Uunl = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fn, np.array([Qunl, Uunl]))

        return self.cacher.load(fn)    
    
    def from_cambfile():
        return 'something'
    

    def cl2alm(self, cls):
        return 'something'


class Xunl:
    def __init__(self, nside, lmax, cls_lib=None, lib_dir=None, fnsQ=None, fnsU=None, fnsP=None, simidxs=None):
        # TODO same decisison-tree as in Xobs: if unlensed maps are stored somewhere, grab them. Otherwise, simulate lensed maps using lenspyx
        self.lib_dir = lib_dir
        if lib_dir is None: # need being generated
            self.lmax = lmax
            self.nside = nside
            self.simidxs = simidxs
            self.cls_lib = cls_lib
        else:
            self.fnsQ = fnsQ
            self.fnsU = fnsU
            self.fnsP = fnsP
            
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_unlmap(self, simidx):
        fn = 'unlmap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunl, Uunl = self.cl2alm(self.cls_lib.get_TEBunl())
            else:
                if self.fnsQ.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Qunl = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Uunl = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                elif self.fnsQ.format(simidx).endswith('fits'):
                    Qunl = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Uunl = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fn, np.array([Qunl, Uunl]))

        return self.cacher.load(fn)


    def get_sim_phi(self, simidx):
        fn = 'phi_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Phi = self.unl_lib.get_sim_clphi(simidx)
                print('here')
            else:
                if self.fnsP.format(simidx+1).endswith('npy'):
                    #TODO delensalotify this
                    Phi = np.load(opj(self.lib_dir, self.fnsP.format(simidx+1)))*1e-3
                elif self.fnsP.format(simidx).endswith('fits'):
                    Phi = np.load(opj(self.lib_dir, self.fnsP.format(simidx+1)))*1e-3
            self.cacher.cache(fn, Phi)
        return self.cacher.load(fn)
    


class Xsky:
    def __init__(self, nside, lmax, unl_lib=None, lib_dir=None, fnsQ=None, fnsU=None, simidxs=None, spin=None):
        self.lib_dir = lib_dir
        if lib_dir is None: # need being generated
            self.nside = nside
            self.lmax = lmax
            self.unl_lib = unl_lib
            self.spin = spin
            self.simidxs = simidxs
        else:
            self.fnsQ = fnsQ
            self.fnsU = fnsU

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_skymap(self, simidx):
        fn = 'skymap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunl, Uunl = self.unl_lib.get_sim_unlmap(simidx)
                phi = self.unl_lib.get_sim_phi(simidx)
                Qsky, Usky = self.unl2len(np.array([Qunl, Uunl]), phi, spin=self.spin)
            else:
                if self.fnsQ.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Qsky = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Usky = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                elif self.fnsQ.format(simidx).endswith('fits'):
                    Qsky = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Usky = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fn, np.array([Qsky, Usky]))
        return self.cacher.load(fn)


    def unl2len(self, Xunl, phi, **kwargs):
        alms = hp.map2alm_spin(Xunl, lmax=self.lmax, spin=self.spin)
        philm = hp.map2alm(phi, lmax=self.lmax)
        return lenspyx.alm2lenmap_spin(alms, philm, geometry=('healpix', {'nside': self.nside}), **kwargs)
    

class Xobs:

    def __init__(self, lmax, cl_transf_P, len_lib=None, noise_lib=None, lib_dir=None, fnsQ=None, fnsU=None, simidxs=None, beam=None, nside=None, nlev_p=None, lib_dir_noise=None, fnsQnoise=None, fnsUnoise=None, lib_dir_phas=None):
        self.simidxs = simidxs
        self.lib_dir = lib_dir
        if lib_dir is None:
            if len_lib is None:
                assert 0, "Either len_lib or lib_dir must be not None"
            else:
                self.len_lib = len_lib
            if noise_lib is None:
                if nside is None or nlev_p is None:
                    assert 0, "Need nside and nlev_p for generating noise"
                self.noise_lib = iso_white_noise(nside=nside, nlev_p=nlev_p, fnsQ=fnsQnoise, fnsU=fnsUnoise, lib_dir=lib_dir_noise, lib_dir_phas=lib_dir_phas)
            else:
                self.noise_lib = noise_lib
            self.cl_transf_P = cl_transf_P
            self.lmax = lmax
            self.beam = beam
            self.nside = nside
            self.nlev_p = nlev_p
        elif lib_dir is not None:
            self.fnsQ = fnsQ
            self.fnsU = fnsU

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_pmap(self, simidx, lmax=None):
        fn = 'pmap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            # Now, decide if data is on disk, or needs combining
            if self.lib_dir is None: # sky maps come from len_lib, and we add noise
                #TODO delensalotify this
                Qobs, Uobs = self.sky2obs(self.len_lib.get_sim_skymap(simidx), [self.noise_lib.get_sim_qnoise(simidx), self.noise_lib.get_sim_unoise(simidx)])
                self.cacher.cache(fn, np.array([Qobs, Uobs]))

            elif self.lib_dir is not None:  # observed maps are somewhere
                if self.fnsQ.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Qobs = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Uobs = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                elif self.fnsQ.format(simidx).endswith('fits'):
                    Qobs = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
                    Uobs = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
                self.cacher.cache(fn, np.array([Qobs, Uobs]))
   
        return self.cacher.load(fn)
    

    def sky2obs(self, QUsky, QUnoise):
        Qsky, Usky = QUsky
        Qnoise, Unoise = QUnoise
        elm, blm = hp.map2alm_spin([Qsky, Usky], 2, lmax=self.lmax)
        # delensalotify this
        hp.almxfl(elm, self.cl_transf_P, inplace=True)
        hp.almxfl(blm, self.cl_transf_P, inplace=True)
        beamedQ, beamedU = hp.alm2map_spin([elm,blm], self.nside, 2, hp.Alm.getlmax(elm.size))
        return np.array([beamedQ + Qnoise, beamedU + Unoise])
        

class Simhandler:

    def __init__(self, data, cls_lib=None, unl_lib=None, len_lib=None, obs_lib=None, noise_lib=None, lib_dir_noise=None, lib_dir=None, fnsQ=None, fnsU=None, fnsP=None, simidxs=None, beam=None, nside=None, lmax=None, cl_transf_P=None, nlev_p=None, fnsQnoise=None, fnsUnoise=None, lib_dir_phas=None, spin=None):
        """either provide,
                lib_dir (skyobs maps are on disk),
            or a library which,
                generates skyobs maps,
                generates lensed CMB maps, 
                synthesizes unlensed CMB maps
            Simhandler will take care of the rest
        """
        self.cls_lib = cls_lib
        if data == 'obs':
            if lib_dir is not None:
                self.lib_dir = lib_dir
                self.Qfns = fnsQ
                self.Ufns = fnsU
                self.simidxs = simidxs
                self.beam = beam
                self.nside = nside
                self.obs_lib = Xobs(cl_transf_P=cl_transf_P, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, simidxs=simidxs, beam=beam, nside=nside)
        if data == 'sky':
            self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, cl_transf_P=cl_transf_P, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, lib_dir_phas=lib_dir_phas)
            self.noise_lib = self.obs_lib.noise_lib
        if data == 'unl':
            self.unl_lib = Xunl(lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsP=fnsP, simidxs=simidxs, nside=nside)
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, cl_transf_P=cl_transf_P, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, lib_dir_phas=lib_dir_phas)
            self.noise_lib = self.obs_lib.noise_lib
        if data == 'cls':
            self.unl_lib = Xunl(cls_lib=self.cls_lib, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsP=fnsP, simidxs=simidxs, nside=nside)
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, cl_transf_P=cl_transf_P, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, lib_dir_phas=lib_dir_phas)
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_pmap(self, simidx):
        return self.obs_lib.get_sim_pmap(simidx)
    
    def get_sim_qnoise(self, simidx):
        return self.noise_lib.get_sim_qnoise(simidx)

    def get_sim_unoise(self, simidx):
        return self.noise_lib.get_sim_unoise(simidx)
    
    def get_sim_skymap(self, simidx):
        return self.len_lib.get_sim_skymap(simidx)

    def get_sim_unlmap(self, simidx):
        return self.unl_lib.get_sim_unlmap(simidx)