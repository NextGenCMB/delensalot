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

    def __init__(self, nlev_p, nside, lmax=None, lib_dir=None, fnsQ=None, fnsU=None, fnsE=None, fnsB=None, spin=None):
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
        if lib_dir is None:        
            self.nlev_p = nlev_p
            lib_dir_phas = os.environ['SCRATCH']+'/sims/nside{}/phas/'.format(nside) # TODO phas should go to sims dir..
            self.pix_lib_phas = phas.pix_lib_phas(lib_dir_phas, 3, (hp.nside2npix(nside),))
        else:
            if spin == 2:
                if fnsQ is None or fnsU is None:
                    assert 0, "must provide filenames for Q and U noise"
                self.fnsQ = fnsQ
                self.fnsU = fnsU
            elif spin == 0:
                if fnsE is None or fnsB is None:
                    assert 0, "must provide filenames for E and B noise"
                self.fnsE = fnsE
                self.fnsB = fnsB 

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher
        # TODO same decisison-tree as in Xobs: if noise maps are stored somewhere, grab them. Otherwise, simulate noise


    def get_sim_noise(self, simidx, spin=2):
        fn = 'noise_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
                Qnoise = self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=1)
                Unoise = self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=2)
                if spin == 0:
                    Enoise, Bnoise = hp.alm2map_spin(hp.map2alm_spin([Qnoise, Unoise], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
            else:
                if self.spin == 2:
                    if self.fnsQ.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Qnoise = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Unoise = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                    elif self.fnsQ.format(simidx).endswith('fits'):
                        Qnoise = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Unoise = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
                    if spin == 0:
                        Enoise, Bnoise = hp.alm2map_spin(hp.map2alm_spin([Qnoise, Unoise], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
                elif self.spin == 0:
                    if self.fnsE.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Enoise = np.load(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bnoise = np.load(opj(self.lib_dir, self.fnsB.format(simidx)))
                    elif self.fnsB.format(simidx).endswith('fits'):
                        Enoise = hp.read_map(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bnoise = hp.read_map(opj(self.lib_dir, self.fnsB.format(simidx)))
                    if spin == 2:
                        Qnoise, Unoise = hp.alm2map_spin(hp.map2alm_spin([Enoise, Bnoise], spin=0, lmax=self.lmax), lmax=self.lmax, spin=2, nside=self.nside)
            if spin == 2:
                self.cacher.cache(fn, np.array([Qnoise, Unoise]))
            elif spin == 0:
                self.cacher.cache(fn, np.array([Enoise, Bnoise]))  
        return self.cacher.load(fn)



class Cls:
    def __init__(self, lmax, CAMB_fn=None, simidxs=None):
        self.lmax = lmax
        self.simidxs = simidxs
        if CAMB_fn is None:
            self.CAMB_file = camb_clfile(opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'))
        else:
            self.CAMB_fn = CAMB_fn
            self.CAMB_file = camb_clfile(CAMB_fn)
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_TEBunl(self, simidx):
        fn = 'cls_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClT, ClE, ClB, ClTE = self.CAMB_file['tt'], self.CAMB_file['ee'], self.CAMB_file['bb'], self.CAMB_file['te']
            self.cacher.cache(fn, np.array([ClT, ClE, ClB, ClTE]))
        return self.cacher.load(fn)   
        
    
    def get_sim_clphi(self, simidx):
        fn = 'clphi_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClP = self.CAMB_file['pp']
            self.cacher.cache(fn, np.array(ClP))
        return self.cacher.load(fn)   


class Xunl:
    def __init__(self, lmax, cls_lib=None, lib_dir=None, fnsE=None, fnsB=None, fnsP=None, simidxs=None, lib_dir_phi=None):
        self.lib_dir = lib_dir
        self.lib_dir_phi = lib_dir_phi
        if lib_dir is None or lib_dir_phi is None: # need being generated
            self.lmax = lmax
            self.simidxs = simidxs
            if cls_lib is None:
                self.cls_lib = Cls(lmax=lmax)
            else:
                self.cls_lib = cls_lib
        if lib_dir is not None:
            self.fnsE = fnsE
            self.fnsB = fnsB
        if lib_dir_phi is not None:
            self.fnsP = fnsP
            
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_unllm(self, simidx):
        fn = 'unllm_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Cls = self.cls_lib.get_TEBunl(simidx)
                Eunl, Bunl = self.cl2alm(Cls)
            else:
                if self.fnsE.format(simidx).endswith('npy'):
                    #TODO delensalotify this
                    Eunl = np.load(opj(self.lib_dir, self.fnsE.format(simidx)))
                    Bunl = np.load(opj(self.lib_dir, self.fnsB.format(simidx)))
                elif self.fnsE.format(simidx).endswith('fits'):
                    Eunl = np.load(opj(self.lib_dir, self.fnsE.format(simidx)))
                    Bunl = np.load(opj(self.lib_dir, self.fnsB.format(simidx)))
            self.cacher.cache(fn, np.array([Eunl, Bunl]))

        return self.cacher.load(fn)


    def get_sim_philm(self, simidx):
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
        alms = hp.synalm(cls, self.lmax, new=True)
        return alms[1:]
    

    def clp2alm(self, clp, simidx):
        plm = hp.synalm(clp, self.lmax)
        return plm


class Xsky:
    def __init__(self, nside, lmax, unl_lib=None, lib_dir=None, fnsQ=None, fnsU=None, fnsE=None, fnsB=None, simidxs=None, spin=None):
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
        if lib_dir is None: # need being generated
            self.unl_lib = unl_lib
            self.simidxs = simidxs
        else:
            if spin == 0:
                if fnsE is None and fnsB is None:
                    assert 0, 'you need to provide fnsE and fnsB' 
            elif spin == 2:
                if fnsQ is None and fnsU is None:
                    assert 0, 'you need to provide fnsQ and fnsU' 
            else:
                assert 0, 'dont understand your spin'
            self.fnsE = fnsE
            self.fnsB = fnsB
            self.fnsQ = fnsQ
            self.fnsU = fnsU

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_skymap(self, simidx, spin=2):
        fn = 'sky_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunllm, Uunllm = self.unl_lib.get_sim_unllm(simidx)
                philm = self.unl_lib.get_sim_philm(simidx)
                Qsky, Usky = self.unl2len(np.array([Qunllm, Uunllm]), philm, spin=self.spin)
                if spin == 0:
                    Esky, Bsky = hp.alm2map_spin(hp.map2alm_spin([Qsky, Usky], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
            else:
                if self.spin == 2:
                    if self.fnsQ.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Qsky = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Usky = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                    elif self.fnsQ.format(simidx).endswith('fits'):
                        Qsky = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Usky = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                    if spin == 0:
                        Esky, Bsky = hp.alm2map_spin(hp.map2alm_spin([Qsky, Usky], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
                elif self.spin == 0:
                    if self.fnsE.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Esky = np.load(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bsky = np.load(opj(self.lib_dir, self.fnsB.format(simidx)))
                    elif self.fnsE.format(simidx).endswith('fits'):
                        Esky = hp.read_map(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bsky = hp.read_map(opj(self.lib_dir, self.fnsB.format(simidx)))
                    if spin == 2:
                        Qsky, Usky = hp.alm2map_spin(hp.map2alm_spin([Esky, Bsky], spin=0, lmax=self.lmax), lmax=self.lmax, spin=2, nside=self.nside)
            if spin == 2:
                self.cacher.cache(fn, np.array([Qsky, Usky]))
            elif spin == 0:
                self.cacher.cache(fn, np.array([Esky, Bsky]))
        return self.cacher.load(fn)


    def unl2len(self, Xlm, philm, **kwargs):
        # This is always polarization for now, therefore hardcoding spin # FIXME once we support temp
        kwargs['spin'] = 2
        ll = np.arange(0,self.lmax+1,1)
        return lenspyx.alm2lenmap_spin(Xlm, hp.almxfl(philm,  np.sqrt(ll*(ll+1))), geometry=('healpix', {'nside': self.nside}), **kwargs)


class Xobs:

    def __init__(self, lmax, transfunction=None, len_lib=None, noise_lib=None, lib_dir=None, fnsQ=None, fnsU=None, fnsE=None, fnsB=None, simidxs=None, nside=None, nlev_p=None, lib_dir_noise=None, fnsQnoise=None, fnsUnoise=None, fnsEnoise=None, fnsBnoise=None, spin=None):
        self.simidxs = simidxs
        self.lib_dir = lib_dir
        self.spin = spin
        self.lmax = lmax
        self.nside = nside
        if lib_dir is None:
            if len_lib is None:
                assert 0, "Either len_lib or lib_dir must be not None"
            else:
                self.len_lib = len_lib
            if noise_lib is None:
                if lib_dir_noise is None:
                    if nside is None or nlev_p is None:
                        assert 0, "Need nside and nlev_p for generating noise"
                self.noise_lib = iso_white_noise(nside=nside, nlev_p=nlev_p, lmax=lmax, fnsQ=fnsQnoise, fnsU=fnsUnoise, fnsE=fnsEnoise, fnsB=fnsBnoise, lib_dir=lib_dir_noise, spin=spin)
            else:
                self.noise_lib = noise_lib
            self.transfunction = transfunction
            
            self.nlev_p = nlev_p
        elif lib_dir is not None:
            if spin == 0:
                if fnsE is None and fnsB is None:
                    assert 0, 'you need to provide fnsE and fnsB' 
            elif spin == 2:
                if fnsQ is None and fnsU is None:
                    assert 0, 'you need to provide fnsQ and fnsU' 
            else:
                assert 0, 'dont understand your spin'
            self.fnsE = fnsE
            self.fnsB = fnsB
            self.fnsQ = fnsQ
            self.fnsU = fnsU

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_pmap(self, simidx, spin=2):  
        fn = 'pmap_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None: # sky maps come from len_lib, and we add noise
                #TODO delensalotify this
                QUEBobs = self.sky2obs(self.len_lib.get_sim_skymap(simidx, spin=spin), self.noise_lib.get_sim_noise(simidx, spin=spin), spin=spin)
                self.cacher.cache(fn, QUEBobs)
            elif self.lib_dir is not None:  # observed maps are somewhere
                if self.spin == 2:
                    if self.fnsQ.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Qobs = np.load(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Uobs = np.load(opj(self.lib_dir, self.fnsU.format(simidx)))
                    elif self.fnsQ.format(simidx).endswith('fits'):
                        Qobs = hp.read_map(opj(self.lib_dir, self.fnsQ.format(simidx)))
                        Uobs = hp.read_map(opj(self.lib_dir, self.fnsU.format(simidx)))
                    if spin == 0:
                        Eobs, Bobs = hp.alm2map_spin(hp.map2alm_spin([Qobs, Uobs], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
                elif self.spin == 0:
                    if self.fnsE.format(simidx).endswith('npy'):
                        #TODO delensalotify this
                        Eobs = np.load(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bobs = np.load(opj(self.lib_dir, self.fnsB.format(simidx)))
                    elif self.fnsE.format(simidx).endswith('fits'):
                        Eobs = hp.read_map(opj(self.lib_dir, self.fnsE.format(simidx)))
                        Bobs = hp.read_map(opj(self.lib_dir, self.fnsB.format(simidx)))
                    if spin == 2:
                        Qobs, Uobs = hp.alm2map_spin(hp.map2alm_spin([Eobs, Bobs], spin=0, lmax=self.lmax), lmax=self.lmax, spin=2, nside=self.nside)
                if spin == 2:
                    self.cacher.cache(fn, np.array([Qobs, Uobs]))
                elif spin == 0:
                    self.cacher.cache(fn, np.array([Eobs, Bobs]))
        return self.cacher.load(fn)
    

    def sky2obs(self, QUEBsky, QUEBnoise, spin):
        QEsky, UBsky = QUEBsky
        QEnoise, UBnoise = QUEBnoise
        elm, blm = hp.map2alm_spin([QEsky, UBsky], spin, lmax=self.lmax)
        # delensalotify this
        hp.almxfl(elm, self.transfunction, inplace=True)
        hp.almxfl(blm, self.transfunction, inplace=True)
        beamedQE, beamedUB = hp.alm2map_spin([elm,blm], self.nside, spin, hp.Alm.getlmax(elm.size))
        return np.array([beamedQE + QEnoise, beamedUB + UBnoise])
  
    def get_sim_noise(self, simidx, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin)
  

class Simhandler:

    def __init__(self, flavour, space='map', cls_lib=None, unl_lib=None, obs_lib=None, len_lib=None, noise_lib=None, lib_dir_noise=None, lib_dir=None, lib_dir_phi=None, fnsQ=None, fnsU=None, fnsE=None, fnsB=None, fnsP=None, simidxs=None, beam=None, nside=None, lmax=None, transfunction=None, nlev_p=None, fnsQnoise=None, fnsUnoise=None, fnsEnoise=None, fnsBnoise=None, spin=None, CAMB_fn=None):
        """either provide,
                lib_dir (skyobs maps are on disk),
            or a library which,
                generates skyobs maps,
                generates lensed CMB maps, 
                synthesizes unlensed CMB maps
            Simhandler will take care of the rest
        """
        self.spin = spin
        self.lmax = lmax
        if flavour == 'obs':
            if lib_dir is not None:
                self.lib_dir = lib_dir
                if spin == 0:
                    if fnsE is None and fnsB is None:
                        assert 0, 'you need to provide fnsE and fnsB' 
                elif spin == 2:
                    if fnsQ is None and fnsU is None:
                        assert 0, 'you need to provide fnsQ and fnsU' 
                else:
                    assert 0, 'dont understand your spin'
                self.fnsQ = fnsQ
                self.fnsU = fnsU
                self.fnsE = fnsE
                self.fnsB = fnsB
                self.simidxs = simidxs
                self.nside = nside
                self.obs_lib = Xobs(transfunction=transfunction, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsE=fnsE, fnsB=fnsB, simidxs=simidxs, nside=nside, spin=spin) if obs_lib is None else obs_lib
        if flavour == 'sky':
            self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, lib_dir=lib_dir, fnsQ=fnsQ, fnsU=fnsU, fnsE=fnsE, fnsB=fnsB, simidxs=simidxs, nside=nside, spin=spin) if len_lib is None else len_lib
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, fnsEnoise=fnsEnoise, fnsBnoise=fnsBnoise, spin=spin)
            self.noise_lib = self.obs_lib.noise_lib
        if flavour == 'unl':
            self.spin = 0 # there are genrally no qlms, ulms, therefore here we can safely assume that data is spin0
            if (lib_dir_phi is None or lib_dir is None) and cls_lib is None:
                cls_lib = Cls(lmax=lmax, CAMB_fn=CAMB_fn, simidxs=simidxs)
            self.cls_lib = cls_lib # just to be safe..
            self.unl_lib = Xunl(lmax=lmax, lib_dir=lib_dir, fnsE=fnsE, fnsB=fnsB, fnsP=fnsP, simidxs=simidxs, lib_dir_phi=lib_dir_phi, cls_lib=cls_lib) if unl_lib is None else unl_lib
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=self.spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, fnsEnoise=fnsEnoise, fnsBnoise=fnsBnoise, spin=self.spin)
            self.noise_lib = self.obs_lib.noise_lib
        if flavour == 'cls':
            self.unl_lib = Xunl(cls_lib=cls_lib, lmax=lmax, lib_dir=lib_dir, fnsE=fnsE, fnsB=fnsB, fnsP=fnsP, simidxs=simidxs, nside=nside, lib_dir_phi=lib_dir_phi)
            self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin)
            self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsQnoise=fnsQnoise, fnsUnoise=fnsUnoise, fnsEnoise=fnsEnoise, fnsBnoise=fnsBnoise, spin=spin)
            self.noise_lib = self.obs_lib.noise_lib
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_pmap(self, simidx, spin=2):
        return self.obs_lib.get_sim_pmap(simidx, spin=spin)
    
    def get_sim_noise(self, simidx, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin)

    def get_sim_skymap(self, simidx, spin=2):
        return self.len_lib.get_sim_skymap(simidx, spin=spin)

    def get_sim_unllm(self, simidx):
        return self.unl_lib.get_sim_unlmap(simidx)