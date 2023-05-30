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

import delensalot
from delensalot.utils import camb_clfile



class iso_white_noise:

    def __init__(self, nlev_p, nside, lmax=None, lib_dir=None, fnsQ=None, fnsU=None):
        self.lib_dir = lib_dir
        if lib_dir is None:        
            self.nlev_p = nlev_p
            self.nside = nside
            self.lmax = lmax
            lib_dir_phas = os.environ['SCRATCH']+'/sims/nside{}/phas/'.format(nside) # TODO phas should go to sims dir..
            self.pix_lib_phas = phas.pix_lib_phas(lib_dir_phas, 3, (hp.nside2npix(nside),))
        else:
            if fnsQ is None or fnsU is None:
                assert 0, "must provide filenames for Q and U noise"
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
    def __init__(self, nside, lmax, CAMB_fn=None, simidxs=None):
        self.nside = nside
        self.lmax = lmax
        if CAMB_fn is None:
            self.CAMB_file = camb_clfile(opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'))
        else:
            self.CAMB_fn = CAMB_fn
            self.CAMB_file = camb_clfile(CAMB_fn)
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_TEBunl(self, simidx):
        fn = 'cls_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClT, ClE, ClB = self.CAMB_file['tt'], self.CAMB_file['ee'], self.CAMB_file['bb']
            self.cacher.cache(fn, np.array([ClT, ClE, ClB]))
        return self.cacher.load(fn)   
        
    
    def get_sim_clphi(self, simidx):
        fn = 'clphi_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClP = self.CAMB_file['pp']
            self.cacher.cache(fn, np.array(ClP))
        return self.cacher.load(fn)   


class Xunl:
    def __init__(self, nside, lmax, cls_lib=None, lib_dir=None, fnsQ=None, fnsU=None, fnsP=None, simidxs=None, lib_dir_phi=None):
        self.lib_dir = lib_dir
        self.lib_dir_phi = lib_dir_phi
        if lib_dir is None or lib_dir_phi is None: # need being generated
            self.lmax = lmax
            self.nside = nside
            self.simidxs = simidxs
            if cls_lib is None:
                assert 0, 'must provide cls_lib in this case'
            self.cls_lib = cls_lib
        if lib_dir is not None:
            self.fnsQ = fnsQ
            self.fnsU = fnsU
        if lib_dir_phi is not None:
            self.fnsP = fnsP
            
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_unlmap(self, simidx):
        fn = 'unlmap_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Cls = self.cls_lib.get_TEBunl()
                Qunl, Uunl = self.cl2alm(Cls)
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
            if self.lib_dir_phi is None:
                ClP = self.cls_lib.get_sim_clphi(simidx)
                Phi = self.clp2alm(ClP, simidx)
            else:
                if self.fnsP.format(simidx+1).endswith('npy'):
                    #TODO delensalotify this
                    Phi = np.load(opj(self.lib_dir_phi, self.fnsP.format(simidx)))
                elif self.fnsP.format(simidx).endswith('fits'):
                    Phi = np.load(opj(self.lib_dir_phi, self.fnsP.format(simidx)))
            self.cacher.cache(fn, Phi)
        return self.cacher.load(fn)


    def cl2alm(self, cls):
        elm, blm = hp.synalm(cls[1], self.lmax), hp.synalm(cls[2], self.lmax)
        return hp.alm2map_spin([elm, blm], spin=2, nside=self.nside)
    

    def clp2alm(self, clp, simidx):
        plm = hp.synalm(clp, self.lmax)
        return hp.alm2map(plm, nside=self.nside)


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
        alms = np.array(hp.map2alm_spin(Xunl, lmax=self.lmax, spin=2))
        philm = hp.map2alm(phi, lmax=self.lmax)
        # TODO flexible geometry choice
        return lenspyx.alm2lenmap_spin(alms, philm, geometry=('healpix', {'nside': self.nside}), **kwargs)
    

class Xobs:

    def __init__(self, lmax, transfunction, len_lib=None, noise_lib=None, lib_dir=None, fnsQ=None, fnsU=None, simidxs=None, beam=None, nside=None, nlev_p=None, lib_dir_noise=None, fnsQnoise=None, fnsUnoise=None):
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
                self.noise_lib = iso_white_noise(nside=nside, nlev_p=nlev_p, fnsQ=fnsQnoise, fnsU=fnsUnoise, lib_dir=lib_dir_noise)
            else:
                self.noise_lib = noise_lib
            self.transfunction = transfunction
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
        hp.almxfl(elm, self.transfunction, inplace=True)
        hp.almxfl(blm, self.transfunction, inplace=True)
        beamedQ, beamedU = hp.alm2map_spin([elm,blm], self.nside, 2, hp.Alm.getlmax(elm.size))
        return np.array([beamedQ + Qnoise, beamedU + Unoise])
        

class Simhandler:

    def __init__(self, flavour, space='map', cls_lib=None, unl_lib=None, obs_lib=None, len_lib=None, noise_lib=None, lib_dir_noise=None, lib_dir=None, lib_dir_phi=None, fnsQ=None, fnsU=None, fnsP=None, simidxs=None, beam=None, nside=None, lmax=None, transfunction=None, nlev_p=None, fnsQnoise=None, fnsUnoise=None, spin=None, CAMB_fn=None):
        """either provide,
                lib_dir (skyobs maps are on disk),
            or a library which,
                generates skyobs maps,
                generates lensed CMB maps, 
                synthesizes unlensed CMB maps
            Simhandler will take care of the rest
        """
        spin = 2 # FIXME hardcoded for now
        if flavour == 'obs':
            if lib_dir is not None:
                self.lib_dir = lib_dir
                self.Qfns = fnsQ
                self.Ufns = fnsU
                self.simidxs = simidxs
                self.beam = beam
                self.nside = nside
                self.obs_lib = Xobs(transfunction=transfunction, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, simidxs=simidxs, beam=beam, nside=nside) if obs_lib is None else obs_lib
        if flavour == 'sky':
            self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, simidxs=simidxs, nside=nside, spin=spin) if len_lib is None else len_lib
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise)
            self.noise_lib = self.obs_lib.noise_lib
        if flavour == 'unl':
            if (lib_dir_phi is None or lib_dir is None) and cls_lib is None:
                cls_lib = Cls(nside=nside, lmax=lmax, CAMB_fn=CAMB_fn, simidxs=simidxs)
            self.cls_lib = cls_lib # just to be safe..
            self.unl_lib = Xunl(lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsP=fnsP, simidxs=simidxs, nside=nside, lib_dir_phi=lib_dir_phi, cls_lib=cls_lib) if unl_lib is None else unl_lib
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise)
            self.noise_lib = self.obs_lib.noise_lib
        if flavour == 'cls':
            self.unl_lib = Xunl(cls_lib=cls_lib, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsP=fnsP, simidxs=simidxs, nside=nside, lib_dir_phi=lib_dir_phi)
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise)
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