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

from delensalot.core import cachers

class Cls:
    def __init__(self, nside, lmax, lib_dir=None, fns=None, CAMB_file=None):
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
        

    def from_cambfile():
        return 'something'



class Xunl:
    def __init__(self, nside, lmax, cls_lib, lib_dir=None, fnsQ=None, fnsU=None):
        # TODO same decisison-tree as in Xobs: if unlensed maps are stored somewhere, grab them. Otherwise, simulate lensed maps using lenspyx
        self.nside = nside
        self.fnsQ = fnsQ
        self.fnsU = fnsU
        self.lmax = lmax
        self.lib_dir = lib_dir
        self.cls_lib = cls_lib
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_CMB_maps(self, simidx):
        fn = 'CMBmap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunl, Uunl = self.cl2alm(self.cls_lib.get_TEBunl())
            else:
                Qunl = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
                Uunl = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fn, np.array([Qunl, Uunl]))
            return self.cacher.load(fn)       


    def cl2alm(self, cls):
        return 'something'


class Xsky:
    def __init__(self, nside, lmax, phi, unl_lib=None, lib_dir=None, fnsQ=None, fnsU=None):
        self.nside = nside
        self.fnsQ = fnsQ
        self.fnsU = fnsU
        self.lmax = lmax
        self.lib_dir = lib_dir
        self.unl_lib = unl_lib
        self.phi = phi
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sky_maps(self, simidx):
        fn = 'skymap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qlen, Ulen = self.unl2len(self.unl_lib.get_CMB_maps(), self.phi)
            else:
                Qlen = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
                Ulen = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
            self.cacher.cache(fn, np.array([Qlen, Ulen]))
            return self.cacher.load(fn)
        
    def unl2len(self, Xunl, phi, kwargs):
        return lenspyx.alm2lenmap_spin(Xunl, phi, **kwargs)

    
    
class iso_white_noise:

    def __init__(self, nlev_p, nside, lib_dir=None, fnsQ=None, fnsU=None):
        self.nlev_p = nlev_p
        self.nside = nside
        self.fnsQ = fnsQ
        self.fnsU = fnsU
        self.lib_dir = lib_dir
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher
        # TODO same decisison-tree as in Xobs: if noise maps are stored somewhere, grab them. Otherwise, simulate noise

    def get_sim_qnoise(self, simidx):
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=1)

    def get_sim_unoise(self, simidx):
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=2)


class Xobs:

    def __init__(self, lmax, cl_transf_P, len_lib=None, noise_lib=None, lib_dir=None, fnsQ=None, fnsU=None):
        
        if len_lib is None:
            if lib_dir is None:
                assert 0, "Either len_lib or lib_dir must be not None"
        self.lib_dir = lib_dir
        self.len_lib = len_lib

        if noise_lib is None:
            self.noise_lib = iso_white_noise()
        else:
            self.noise_lib = noise_lib

        self.fnsQ = fnsQ
        self.fnsU = fnsU
        self.lmax = lmax
        self.cl_transf_P = cl_transf_P

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_pmap(self, simidx, lmax=None):
        fn = 'pmap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            # Now, decide if data is on disk, or needs combining
            if self.lib_dir is None: # sky maps come from len_lib, and we add noise
                #TODO delensalotify this
                Qobs, Uobs = self.lenplusnoise(self.len_lib.get_sky_maps(simidx), [self.noise_lib.get_sim_qnoise(simidx), self.noise_lib.get_sim_unoise(simidx)])
                self.cacher.cache(fn, )

            elif self.lib_dir is not None:  # observed maps are somewhere
                #TODO delensalotify this
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

    def __init__(self, cls_lib=None, unl_lib=None, len_lib=None, obs_lib=None):
        self.cls_lib = cls_lib
        self.unl_lib = unl_lib
        self.len_lib = len_lib
        self.obs_lib = obs_lib
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_pmap(self, simidx):
        return self.obs_lib.get_sim_pmap(simidx)
