"""sims/sims_lib.py: library for collecting and handling simulations. Eventually, delensalot needs a get_sim_pmap(), which are the maps of the observed sky. But data may come from any of the above. So this module allows to start analysis with,
    cls,
    alms_unl
    alm_len + noise
    obs_sky
and is a mapper between them, so that get_sim_pmap() always returns observed maps.
"""

import os
from os.path import join as opj
import numpy as np, healpy as hp

import logging
log = logging.getLogger(__name__)

import lenspyx
from lenspyx.lensing import get_geom
from plancklens.sims import phas
from delensalot.core import cachers
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV

import delensalot
from delensalot.utils import load_file, cli


def klm2plm(klm, lmax):
    assert 0, 'check factor'
    LL = np.arange(0,lmax+1,1)
    factor = LL*(LL+1)/2
    return hp.almxfl(klm, cli(factor))

def dlm2plm(dlm, lmax):
    assert 0, 'check factor'
    LL = np.arange(0,lmax+1,1)
    factor = np.sqrt(LL*(LL+1))
    return hp.almxfl(dlm, cli(factor))

def clk2clp(clk, lmax):
    assert 0, 'check factor'
    LL = np.arange(0,lmax+1,1)
    factor = (LL*(LL+1)/2)**2
    return hp.almxfl(clk, cli(factor))

def cld2clp(cld, lmax):
    assert 0, 'check factor'
    LL = np.arange(0,lmax+1,1)
    factor = LL*(LL+1)
    return hp.almxfl(cld, cli(factor))


class iso_white_noise:

    def __init__(self, nlev, lmax=DNaV, libdir=DNaV, fns=DNaV, spin=DNaV, space=DNaV, geometry=DNaV):
        self.geometry = geometry
        if geometry == DNaV:
            self.geometry = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(geometry)
        self.libdir = libdir
        self.spin = spin
        self.lmax = lmax
        self.space = space
        if libdir == DNaV:        
            self.nlev = nlev
            libdir_phas = os.environ['SCRATCH']+'/simulation/{}/{}/phas/'.format(str(geometry),str(nlev))
            self.pix_lib_phas = phas.pix_lib_phas(libdir_phas, 3, (self.geom_lib.npix(),))
        else:
            if fns == DNaV:
                assert 0, "must provide fns"
            self.fns = fns

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_noise(self, simidx, space, field, spin=2):
        if space == 'alm' and spin == 2:
            assert 0, "I don't think you want qlms ulms."
        if field == 'temperature' and spin == 2:
            assert 0, "I don't think you want spin-2 temperature."
        if field == 'temperature' and 'T' not in self.nlev:
            assert 0, "need to provide T key in nlev"
        if field == 'polarization' and 'P' not in self.nlev:
            assert 0, "need to provide T key in nlev"
        fn = 'noise_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            if self.libdir == DNaV:
                if self.geometry[0] == 'healpix':
                    vamin = np.sqrt(hp.nside2pixarea(self.geometry[1]['nside'], degrees=True)) * 60
                else:
                    ## TODO this is a rough estimate, based on total sky coverage / npix()
                    vamin =  np.sqrt(4*np.pi) * (180/np.pi) / self.geom_lib.npix() * 60
                if field == 'polarization':
                    noise1 = self.nlev['P'] / vamin * self.pix_lib_phas.get_sim(simidx, idf=1)
                    noise2 = self.nlev['P'] / vamin * self.pix_lib_phas.get_sim(simidx, idf=2) # TODO this always produces qu-noise in healpix geometry?
                    noise = np.array([noise1, noise2])
                    if space == 'map':
                        if spin == 0:
                            alm_buffer = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            noise = np.array([noise1, noise2])
                    elif space == 'alm':
                        noise = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                elif field == 'temperature':
                    noise = self.nlev['T'] / vamin * self.pix_lib_phas.get_sim(simidx, idf=0)
                    if space == 'alm':
                        noise = self.geom_lib.map2alm(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                if field == 'polarization':
                    noise1 = load_file(opj(self.libdir, self.fns[0].format(simidx)))
                    noise2 = load_file(opj(self.libdir, self.fns[1].format(simidx)))
                    noise = np.array([noise1, noise2])
                    if self.space == 'map':
                        if space == 'alm':
                            if self.spin == 0:
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif self.spin == 2:
                                noise = self.geom_lib.map2alm_spin(noise, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        elif space == 'map':
                            if self.spin != spin:
                                if self.spin == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(noise[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(noise[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    noise = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                elif self.spin == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(noise, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    noise = np.array([noise1, noise2])
                    elif self.space == 'alm':
                        if space == 'map':
                            if spin == 0:
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif spin == 2:
                                noise = self.geom_lib.alm2map_spin(noise, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)       
                elif field == 'temperature':
                    noise = np.array(load_file(opj(self.libdir, self.fns.format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            noise = self.geom_lib.map2alm(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            noise = self.geom_lib.alm2map(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, noise)  
        return self.cacher.load(fn)


class Cls:
    def __init__(self, lmax=DNaV, phi_lmax=DNaV, CMB_fn=DNaV, simidxs=DNaV, phi_fn=DNaV, phi_field='potential'):
        assert lmax != DNaV, "need to provide lmax"
        self.lmax = lmax
        self.phi_lmax = phi_lmax
        if phi_lmax == DNaV:
            self.phi_lmax = lmax + 1024
        self.simidxs = simidxs
        if CMB_fn == DNaV:
            self.CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat')
            self.CAMB_file = load_file(self.CMB_fn)
        else:
            self.CMB_fn = CMB_fn
            self.CAMB_file = load_file(CMB_fn)
        if phi_fn == 'None':
            self.phi_fn = None
        elif phi_fn == DNaV:
            self.phi_fn = self.CMB_fn
            self.phi_file = load_file(self.phi_fn)['pp']
            self.phi_field = phi_field # assuming that CAMB file is 'potential'
        else:
            self.phi_fn = phi_fn
            self.phi_file = load_file(self.phi_fn)['pp']
            self.phi_field = phi_field
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_TEBunl(self, simidx):
        fn = 'cls_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClT, ClE, ClB, ClTE = self.CAMB_file['tt'][:self.lmax+1], self.CAMB_file['ee'][:self.lmax+1], self.CAMB_file['bb'][:self.lmax+1], self.CAMB_file['te'][:self.lmax+1]
            self.cacher.cache(fn, np.array([ClT, ClE, ClB, ClTE]))
        return self.cacher.load(fn)   
        
    
    def get_sim_clphi(self, simidx):
        fn = 'clphi_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            ClP = self.phi_file[:self.phi_lmax+1]
            self.cacher.cache(fn, np.array(ClP))
        return self.cacher.load(fn)


class Xunl:
    def __init__(self, lmax, cls_lib=DNaV, libdir=DNaV, fns=DNaV, fnsP=DNaV, simidxs=DNaV, libdir_phi=DNaV, phi_field='potential', phi_space=DNaV, phi_lmax=DNaV, space=DNaV, geometry=DNaV, isfrozen=False, spin=DNaV):
        self.geometry = geometry
        if geometry == DNaV:
            self.geometry = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geometry)
        self.libdir = libdir
        self.space = space
        self.spin = spin
        self.lmax = lmax
        self.libdir_phi = libdir_phi
        self.phi_lmax = phi_lmax
        self.simidxs = simidxs
       
        if phi_field == DNaV:
            self.phi_field = 'potential'
        else:
            self.phi_field = phi_field
        self.phi_space = phi_space

        if libdir_phi == DNaV: # need being generated
            if cls_lib == DNaV:
                self.cls_lib = Cls(lmax=lmax, phi_field=self.phi_field, phi_lmax=self.phi_lmax)
            else:
                self.cls_lib = cls_lib
            self.phi_lmax = self.cls_lib.phi_lmax
        if libdir != DNaV:
            if self.space == DNaV:
                assert 0, 'need to give space (map or alm)'
            self.fns = fns
            if self.fns == DNaV:
                assert 0, 'need to give fns'
            if self.spin == DNaV:
                assert 0, 'need to give spin'
        else:
            if cls_lib == DNaV:
                self.cls_lib = Cls(lmax=lmax, phi_field=self.phi_field, phi_lmax=self.phi_lmax)
            else:
                self.cls_lib = cls_lib        
            self.phi_lmax = self.cls_lib.phi_lmax     
        if libdir_phi != DNaV:
            self.fnsP = fnsP
            if self.fnsP == DNaV:
                assert 0, 'need to give fnsP'
            if self.phi_space == DNaV:
                assert 0, 'need to give phi_space (map or alm)'
        else:
            if 'nside' in  self.geometry[1]:
                geom_lmax = 3*self.geometry[1]['nside']
            elif 'lmax' in  self.geometry[1]:
                geom_lmax = self.geometry[1]['lmax']
            else:
                geom_lmax = lmax + 1024
            self.phi_lmax = np.min([lmax + 1024, geom_lmax])
        self.isfrozen = isfrozen
            
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_unl(self, simidx, space, field, spin=2):
        """returns an unlensed simulation field (temp,pol,cross) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

        Args:
            simidx (_type_): _description_
            space (_type_): _description_
            spin (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        if space == 'alm' and spin == 2:
            assert 0, "I don't think you want qlms ulms."
        if field == 'temperature' and spin == 2:
            assert 0, "I don't think you want spin-2 temperature."
        fn = 'unl_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            if self.libdir == DNaV:
                Cls = self.cls_lib.get_TEBunl(simidx)
                unl = np.array(self.cl2alm(Cls, field=field, seed=simidx))
                if space == 'map':
                    if field == 'polarization':
                        if spin == 2:
                            unl = self.geom_lib.alm2map_spin(unl, lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                        elif spin == 0:
                            unl1 = self.geom_lib.alm2map(unl[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            unl2 = self.geom_lib.alm2map(unl[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            unl = np.array([unl1, unl2])
                    elif field == 'temperature':
                        unl = self.geom_lib.alm2map(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                if field  == 'polarization':
                    unl1 = load_file(opj(self.libdir, self.fns[0].format(simidx)))
                    unl2 = load_file(opj(self.libdir, self.fns[1].format(simidx)))
                    unl =  np.array([unl1, unl2])
                    if self.space == 'map':
                        if space == 'alm':
                            if self.spin == 2:
                                unl = self.geom_lib.map2alm_spin(unl, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            elif self.spin == 0:
                                unl1 = self.geom_lib.map2alm(unl[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                unl2 = self.geom_lib.map2alm(unl[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                unl = np.array([unl1, unl2])
                        elif space == 'map':
                            if self.spin != spin:
                                if self.spin == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(unl[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(unl[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    unl = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                elif self.spin == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(unl, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    unl1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    unl2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    unl = np.array([unl1, unl2])
                    elif self.space == 'alm':
                        if space == 'map':
                            if spin == 0:
                                unl = self.geom_lib.alm2map(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            elif spin == 2:
                                unl = self.geom_lib.alm2map_spin(unl, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                elif field == 'temperature':
                    unl = np.array(load_file(opj(self.libdir, self.fns.format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            unl = self.geom_lib.map2alm(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            unl = self.geom_lib.alm2map(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, unl)
        return self.cacher.load(fn)
    

    def get_sim_phi(self, simidx, space):
        """returns an unlensed simulation field (temp,pol,cross) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

        Args:
            simidx (_type_): _description_
            space (_type_): _description_
            spin (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """        
        fn = 'phi_space{}_{}'.format(space, simidx)
        if not self.cacher.is_cached(fn):
            if self.libdir_phi == DNaV:
                Clpf = self.cls_lib.get_sim_clphi(simidx)
                self.phi_field = self.cls_lib.phi_field
                Clp = self.clpf2clppot(Clpf)
                phi = self.clp2plm(Clp, simidx)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
            else:
                phi = load_file(opj(self.libdir_phi, self.fnsP.format(simidx)))
                if self.phi_space == 'map':
                    phi = self.geom_lib.map2alm(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
                phi = self.pflm2plm(phi)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
            self.cacher.cache(fn, phi)
        return self.cacher.load(fn)
    

    def pflm2plm(self, philm):
        if self.phi_field == 'kappa':
            return klm2plm(philm, self.phi_lmax)
        elif self.phi_field == 'deflection':
            return dlm2plm(philm, self.phi_lmax)
        elif self.phi_field == 'potential':
            return philm


    def clpf2clppot(self, cl):
        if self.phi_field == 'kappa':
            return clk2clp(cl, self.phi_lmax)
        elif self.phi_field == 'deflection':
            return cld2clp(cl, self.phi_lmax)
        elif self.phi_field == 'potential':
            return cl


    def cl2alm(self, cls, field, seed):
        np.random.seed(seed)
        if field == 'polarization':
            alms = hp.synalm(cls, self.lmax, new=True)
            return alms[1:]
        elif field == 'temperature':
            alm = hp.synalm(cls[0], self.lmax)
            return alm
    

    def clp2plm(self, clp, seed):
        np.random.seed(seed)
        plm = hp.synalm(clp, self.phi_lmax)
        return plm


class Xsky:
    def __init__(self, lmax, unl_lib=DNaV, libdir=DNaV, fns=DNaV, simidxs=DNaV, spin=DNaV, epsilon=1e-7, space=DNaV, geometry=DNaV, isfrozen=False):
        self.geometry = geometry
        if geometry == DNaV:
            self.geometry = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geometry)
        self.libdir = libdir
        self.fns = fns
        self.spin = spin
        self.lmax = lmax
        self.space = space
        if libdir == DNaV: # need being generated
            if unl_lib == DNaV:
                self.unl_lib = Xunl(lmax=lmax, simidxs=simidxs, geometry=self.geometry)
            else:
                self.unl_lib = unl_lib
            self.simidxs = simidxs
            if epsilon == DNaV:
                self.epsilon = 1e-7
            else:
                self.epsilon = epsilon
        else:
            if self.spin == DNaV:
                assert 0, 'need to give spin'
            if self.space == DNaV:
                assert 0, 'need to give space (map or alm)'
            if fns == DNaV:
                assert 0, 'you need to provide fns' 
        self.isfrozen = isfrozen

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_sky(self, simidx, space, field, spin=2):
        """returns a lensed simulation field (temp,pol,cross) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

        Args:
            simidx (_type_): _description_
            space (_type_): _description_
            field (_type_): _description_
            spin (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        if space == 'alm' and spin == 2:
            assert 0, "I don't think you want qlms ulms."
        if field == 'temperature' and spin == 2:
            assert 0, "I don't think you want spin-2 temperature."
        fn = 'sky_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        log.info('requesting "{}"'.format(fn))
        if not self.cacher.is_cached(fn):
            fn_other = 'sky_space{}_spin{}_field{}_{}'.format(space, self.spin, field, simidx)
            if not self.cacher.is_cached(fn_other):
                log.info('..nothing cached..')
                if self.libdir == DNaV:
                    log.info('.., generating.')
                    unl = self.unl_lib.get_sim_unl(simidx, space='alm', field=field, spin=0)
                    philm = self.unl_lib.get_sim_phi(simidx, space='alm')
                    if field == 'polarization':
                        sky = self.unl2len(unl, philm, spin=2, epsilon=self.epsilon)
                        if space == 'map':
                            if spin == 0:
                                alm_buffer = self.geom_lib.map2alm_spin(sky, spin=2, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky = np.array([sky1, sky2])
                        elif space == 'alm':
                            sky = self.geom_lib.map2alm_spin(sky, lmax=self.lmax, spin=2, mmax=self.lmax, nthreads=4)
                    elif field == 'temperature':
                        sky = self.unl2len(unl, philm, spin=0, epsilon=self.epsilon)
                        if space == 'alm':
                            sky = self.geom_lib.map2alm(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                else:
                    log.info('.., but stored on disk.')
                    if field == 'polarization':
                        sky1 = load_file(opj(self.libdir, self.fns[0].format(simidx)))
                        sky2 = load_file(opj(self.libdir, self.fns[1].format(simidx)))
                        sky = np.array([sky1, sky2])
                        if self.space == 'map':
                            if space == 'alm':
                                if self.spin == 0:
                                    sky1 = self.geom_lib.map2alm(sky[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    sky2 = self.geom_lib.map2alm(sky[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    sky = np.array([sky1, sky2])
                                else:
                                    sky = self.geom_lib.map2alm_spin(sky, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            elif space == 'map':
                                if self.spin != spin:
                                    if self.spin == 0:
                                        alm_buffer1 = self.geom_lib.map2alm(sky[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                        alm_buffer2 = self.geom_lib.map2alm(sky[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                        sky = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                    elif self.spin == 2:
                                        alm_buffer = self.geom_lib.map2alm_spin(sky, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                        sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                        sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                        sky = np.array([sky1, sky2])
                        elif self.space == 'alm':
                            if space == 'map':
                                if spin == 0:
                                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    sky = np.array([sky1, sky2])
                                else:
                                    sky = self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif field == 'temperature':
                        sky = np.array(load_file(opj(self.libdir, self.fns.format(simidx))))
                        if self.space == 'map':
                            if space == 'alm':
                                sky = self.geom_lib.map2alm(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        elif self.space == 'alm':
                            if space == 'map':
                                sky = self.geom_lib.alm2map(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                sky = self.cacher.load(fn_other)
                if space == 'map':
                    sky = self.geom_lib.alm2map_spin(self.geom_lib.map2alm_spin(sky, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4), lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)
    

    def unl2len(self, Xlm, philm, **kwargs):
        ll = np.arange(0,self.unl_lib.phi_lmax+1,1)
        return lenspyx.alm2lenmap_spin(Xlm, hp.almxfl(philm,  np.sqrt(ll*(ll+1))), geometry=self.geometry, **kwargs)


class Xobs:

    def __init__(self, lmax, maps=DNaV, transfunction=DNaV, len_lib=DNaV, unl_lib=DNaV, epsilon=DNaV, noise_lib=DNaV, libdir=DNaV, fns=DNaV, simidxs=DNaV, nlev=DNaV, libdir_noise=DNaV, fnsnoise=DNaV, spin=DNaV, space=DNaV, geometry=DNaV, field=DNaV):
        self.geometry = geometry
        if geometry == DNaV:
            self.geometry = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geometry)
        self.simidxs = simidxs
        self.libdir = libdir
        self.fns = fns
        self.spin = spin
        self.lmax = lmax
        self.space = space
        self.noise_lib = noise_lib
        self.fullsky = True #FIXME make it dependent on userdata: if Xobs is set via simhandler, then check if user data is full sky or not.
        
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher
        self.maps = maps
        if np.all(self.maps != DNaV):
            fn = 'obs_space{}_spin{}_field{}_{}'.format(space, spin, field, 0)
            self.cacher.cache(fn, np.array(self.maps))
        else:
            if libdir == DNaV:
                if len_lib == DNaV:
                    self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, libdir=libdir, fns=fns, space=space, simidxs=simidxs, epsilon=epsilon, geometry=geometry)
                else:
                    self.len_lib = len_lib
                if noise_lib == DNaV:
                    if libdir_noise == DNaV:
                        if nlev == DNaV:
                            assert 0, "Need nlev for generating noise"
                    self.noise_lib = iso_white_noise(nlev=nlev, lmax=lmax, fns=fnsnoise,libdir=libdir_noise, space=space, geometry=self.geometry)
                if np.all(transfunction == DNaV):
                    assert 0, 'need to give transfunction'
                self.transfunction = transfunction       
            elif libdir != DNaV:
                if self.space == DNaV:
                    assert 0, 'need to give space (map or alm)'
                self.fns = fns
                if fns == DNaV:
                    assert 0, 'you need to provide fns' 
                if self.spin == DNaV:
                    assert 0, 'need to give spin'


    def get_sim_obs(self, simidx, space, field, spin=2):
        # TODO this is missing field=cross, 
        """returns an observed simulation field (temp,pol,cross) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

        Args:
            simidx (_type_): _description_
            space (_type_): _description_
            field (_type_): _description_
            spin (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        if space == 'alm' and spin == 2:
            assert 0, "I don't think you want qlms ulms."
        if field == 'temperature' and spin == 2:
            assert 0, "I don't think you want spin-2 temperature."
        if not self.fullsky:
            assert self.spin == spin, "can only provide existing data"
            assert self.space == space, "can only provide existing data"
        fn = 'obs_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        log.info('requesting "{}"'.format(fn))
        fn_otherspin = 'obs_space{}_spin{}_field{}_{}'.format(space, self.spin, field, simidx)
        fn_otherspace = ''
        fn_otherspacespin = ''
        if self.space == 'alm':
            fn_otherspace = 'obs_space{}_spin{}_field{}_{}'.format('alm', 0, field, simidx)
        elif self.space == 'map':
            fn_otherspace = 'obs_space{}_spin{}_field{}_{}'.format('map', spin, field, simidx)
        if self.space == 'alm':
            fn_otherspacespin = 'obs_space{}_spin{}_field{}_{}'.format('alm', 0, field, simidx)
        elif self.space == 'map':
            fn_otherspacespin = 'obs_space{}_spin{}_field{}_{}'.format('map', self.spin, field, simidx)

        if not self.cacher.is_cached(fn) and not self.cacher.is_cached(fn_otherspin) and not self.cacher.is_cached(fn_otherspacespin) and not self.cacher.is_cached(fn_otherspace):
            log.info('..nothing cached..')
            if self.libdir == DNaV: # sky maps come from len_lib, and we add noise
                log.info('.., generating.')
                obs = self.sky2obs(
                    self.len_lib.get_sim_sky(simidx, spin=spin, space=space, field=field),
                    self.noise_lib.get_sim_noise(simidx, spin=spin, field=field, space=space),
                    spin=spin,
                    space=space,
                    field=field)
            elif self.libdir != DNaV:  # observed maps are somewhere
                log.info('.., but stored on disk.')
                if field == 'polarization':
                    obs1 = load_file(opj(self.libdir, self.fns[0].format(simidx)))
                    obs2 = load_file(opj(self.libdir, self.fns[1].format(simidx)))
                    obs = np.array([obs1, obs2])
                    if self.space == 'map':
                        if space == 'map':
                            if self.spin != spin:
                                if self.spin == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    obs = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                elif self.spin == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    obs1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    obs2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                    obs = np.array([obs1, obs2])
                        elif space == 'alm':
                            if self.spin == 0:
                                obs1 = self.geom_lib.map2alm(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                obs2 = self.geom_lib.map2alm(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            if spin == 0:
                                obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.alm2map_spin(obs, lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                elif field == 'temperature':
                    obs = np.array(load_file(opj(self.libdir, self.fns.format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            obs = self.geom_lib.map2alm(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            obs = self.geom_lib.alm2map(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                self.cacher.cache(fn, obs)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn):
            log.info('found "{}"'.format(fn))
            pass
        elif self.cacher.is_cached(fn_otherspin):
            log.info('found "{}"'.format(fn_otherspin))
            obs = np.array(self.cacher.load(fn_otherspin))
            if space == 'map':
                if self.spin == 2:
                    obs1 = self.geom_lib.map2alm(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs2 = self.geom_lib.map2alm(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs = np.array([obs1, obs2])
                    obs = self.geom_lib.alm2map_spin(obs, lmax=self.lmax, spin=self.spin, mmax=self.lmax, nthreads=4)
                else:
                    obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs = np.array([obs1, obs2])
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspace):
            log.info('found "{}"'.format(fn_otherspace))
            obs = np.array(self.cacher.load(fn_otherspace))
            if field == 'polarization':
                if self.space == 'alm':
                    if spin == 0:
                        obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        obs = np.array([obs1, obs2])
                    elif spin == 2:
                        obs = self.geom_lib.alm2map_spin(obs, lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                elif self.space == 'map':
                    if self.spin == 0:
                        alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        obs = np.array([alm_buffer1, alm_buffer2])
                    elif self.spin == 2:
                        obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            elif field == 'temperature':
                if self.space == 'alm': 
                    obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs = np.array([obs1, obs2])
                elif self.space == 'map':
                    alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    obs = np.array([alm_buffer1, alm_buffer2])
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspacespin):
            log.info('found "{}"'.format(fn_otherspacespin))
            obs = np.array(self.cacher.load(fn_otherspacespin))
            if self.space == 'alm':
                obs = self.geom_lib.alm2map_spin(obs, lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
            elif self.space == 'map':
                obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, obs)
        return self.cacher.load(fn)
    

    def sky2obs(self, sky, noise, spin, space, field):
        if field == 'polarization':
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.map2alm(sky[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    sky2 = self.geom_lib.map2alm(sky[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = self.geom_lib.map2alm_spin(sky, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            hp.almxfl(sky[0], self.transfunction, inplace=True)
            hp.almxfl(sky[1], self.transfunction, inplace=True)
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = np.array(self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4))
                return sky + noise
            else:
                return sky + noise
        elif field == 'temperature':
            if space == 'map':
                sky = self.geom_lib.map2alm(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            hp.almxfl(sky, self.transfunction, inplace=True)
            if space == 'map':
                return np.array(self.geom_lib.alm2map(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)) + noise
            else:
                return sky + noise


    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
  

class Simhandler:
    """Entry point for data handling and generating simulations. Data can be cl, unl, len, or obs, .. and alms or maps. Simhandler connects the individual libraries and decides what can be generated. E.g.: If obs data provided, len data cannot be generated. This structure makes sure we don't "hallucinate" data.

    """
    def __init__(self, flavour, space, maps=DNaV, cls_lib=DNaV, unl_lib=DNaV, obs_lib=DNaV, len_lib=DNaV, noise_lib=DNaV, libdir_noise=DNaV, libdir=DNaV, libdir_phi=DNaV, fns=DNaV, fnsP=DNaV, simidxs=DNaV, lmax=DNaV, transfunction=DNaV, nlev=DNaV, fnsnoise=DNaV, spin=0, CMB_fn=DNaV, phi_fn=DNaV, phi_field=DNaV, phi_space=DNaV, epsilon=1e-7, geometry=DNaV, phi_lmax=DNaV, field=DNaV):
        """_summary_

        Args:
            flavour (_type_): _description_
            space (str, optional): _description_. Defaults to 'map'.
            maps (_type_, optional): _description_. Defaults to None.
            cls_lib (_type_, optional): _description_. Defaults to None.
            unl_lib (_type_, optional): _description_. Defaults to None.
            obs_lib (_type_, optional): _description_. Defaults to None.
            len_lib (_type_, optional): _description_. Defaults to None.
            noise_lib (_type_, optional): _description_. Defaults to None.
            libdir_noise (_type_, optional): _description_. Defaults to None.
            libdir (_type_, optional): _description_. Defaults to None.
            libdir_phi (_type_, optional): _description_. Defaults to None.
            fns (_type_, optional): _description_. Defaults to None.
            fnsP (_type_, optional): _description_. Defaults to None.
            simidxs (_type_, optional): _description_. Defaults to None.
            lmax (_type_, optional): _description_. Defaults to None.
            transfunction (_type_, optional): _description_. Defaults to None.
            nlev (_type_, optional): _description_. Defaults to None.
            fnsnoise (_type_, optional): _description_. Defaults to None.
            spin (_type_, optional): _description_. Defaults to None.
            CMB_fn (_type_, optional): _description_. Defaults to None.
            epsilon (_type_, optional): _description_. Defaults to 1e-7.
        """
        self.spin = spin
        self.lmax = lmax
        self.phi_lmax = phi_lmax
        self.flavour = flavour
        self.space = space
        self.nlev = nlev
        if space == 'map':
            if flavour == 'obs':
                self.simidxs = simidxs
                if np.all(maps == DNaV):
                    assert libdir != DNaV, "need to provide libdir"
                    assert fns != DNaV, 'you need to provide fns' 
                    assert lmax != DNaV, "need to provide lmax"
                    assert spin != DNaV, "need to provide spin"
                else:
                    assert spin != DNaV, "need to provide spin"
                    assert lmax != DNaV, "need to provide lmax"
                    assert field != DNaV, "need to provide field"
                self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, libdir=libdir, fns=fns, simidxs=simidxs, spin=spin, geometry=geometry, field=field) if obs_lib == DNaV else obs_lib
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.obs_lib.libdir
                self.fns = self.obs_lib.fns
            if flavour == 'sky':
                assert libdir != DNaV, "need to provide libdir"
                assert fns != DNaV, 'you need to provide fns' 
                assert lmax != DNaV, "need to provide lmax"
                assert spin != DNaV, "need to provide spin"
                assert nlev != DNaV, "need to provide nlev"
                assert np.all(transfunction != DNaV), "need to provide transfunction"
                self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, libdir=libdir, fns=fns, space=space, simidxs=simidxs, spin=spin, epsilon=epsilon, geometry=geometry) if len_lib == DNaV else len_lib
                self.obs_lib = Xobs(len_lib=self.len_lib, space=space, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.len_lib.libdir
                self.fns = self.len_lib.fns
            if flavour == 'unl':
                assert libdir != DNaV, "need to provide libdir"
                assert fns != DNaV, 'you need to provide fns' 
                assert lmax != DNaV, "need to provide lmax"
                assert spin != DNaV, "need to provide spin"
                assert nlev != DNaV, "need to provide nlev"
                assert np.all(transfunction != DNaV), "need to provide transfunction"
                if libdir_phi != DNaV:
                    assert phi_field != DNaV, "need to provide phi_field"
                    assert fnsP != DNaV, "need to provide fnsP"
                    assert phi_lmax != DNaV, "need to provide phi_lmax"
                    assert phi_space != DNaV, "need to provide phi_space"
                    self.unl_lib = Xunl(lmax=lmax, libdir=libdir, fns=fns, fnsP=fnsP, phi_field=phi_field, simidxs=simidxs, libdir_phi=libdir_phi, space=space, phi_space=phi_space, phi_lmax=phi_lmax, geometry=geometry, spin=spin) if unl_lib == DNaV else unl_lib
                elif libdir_phi == DNaV:
                    assert phi_fn != DNaV, "need to provide phi_fn"
                    assert phi_lmax != DNaV, "need to provide phi_lmax"
                    assert phi_field != DNaV, "need to provide phi_field"
                    assert phi_space == 'cl', "please set phi_space='cl', just to be sure."
                    self.cls_lib = Cls(phi_lmax=phi_lmax, phi_fn=phi_fn, phi_field=phi_field, simidxs=simidxs)
                    self.unl_lib = Xunl(cls_lib=self.cls_lib, lmax=lmax, libdir=libdir, fns=fns, phi_field=phi_field, simidxs=simidxs, space=space, phi_space=phi_space, phi_lmax=phi_lmax, geometry=geometry, spin=spin) if unl_lib == DNaV else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, space=space, epsilon=epsilon, geometry=geometry)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, space=space, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.unl_lib.libdir
                self.fns = self.unl_lib.fns
        elif space in ['alm']:
            self.spin = 0 # there are genrally no qlms, ulms, therefore here we can safely assume that data is spin0
            spin = 0
            if flavour == 'obs':
                if libdir != DNaV:
                    self.libdir = libdir
                    if fns == DNaV:
                        assert 0, 'you need to provide fns' 
                    self.fns = fns
                    self.simidxs = simidxs
                    self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, libdir=libdir, fns=fns, simidxs=simidxs, spin=self.spin, geometry=geometry) if obs_lib == DNaV else obs_lib
                    self.noise_lib = self.obs_lib.noise_lib
                    self.libdir = self.obs_lib.libdir
                    self.fns = self.obs_lib.fns
            if flavour == 'sky':
                assert 0, 'implement if needed'
            if flavour == 'unl':
                if (libdir_phi == DNaV or libdir == DNaV) and cls_lib == DNaV:
                    cls_lib = Cls(lmax=lmax, CMB_fn=CMB_fn, phi_fn=phi_fn, phi_field=phi_field, simidxs=simidxs)
                self.cls_lib = cls_lib # just to be safe..
                self.unl_lib = Xunl(lmax=lmax, libdir=libdir, fns=fns, fnsP=fnsP, phi_field=phi_field, simidxs=simidxs, libdir_phi=libdir_phi, space=space, phi_space=phi_space, cls_lib=cls_lib, geometry=geometry, spin=self.spin) if unl_lib == DNaV else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, space=space, epsilon=epsilon, geometry=geometry)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, space=space, spin=self.spin, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.unl_lib.libdir
                self.fns = self.unl_lib.fns
        elif space == 'cl':
            self.spin = 0 # there are genrally no qcls, ucls, therefore here we can safely assume that data is spin0
            spin = 0
            if flavour == 'obs':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'sky':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'unl':
                assert np.all(transfunction != DNaV), "need to provide transfunction"
                assert lmax != DNaV, "need to provide lmax"
                assert nlev != DNaV, "need to provide nlev"
                assert phi_lmax != DNaV, "need to provide phi_lmax"
                if phi_fn != DNaV:
                    assert phi_field != DNaV, "need to provide phi_field"
                
                self.cls_lib = Cls(lmax=lmax, phi_lmax=phi_lmax, CMB_fn=CMB_fn, phi_fn=phi_fn, phi_field=phi_field, simidxs=simidxs)
                self.unl_lib = Xunl(cls_lib=self.cls_lib, lmax=lmax, fnsP=fnsP, phi_field=phi_field, libdir_phi=libdir_phi, phi_space=phi_space, simidxs=simidxs, geometry=geometry)
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, epsilon=epsilon, geometry=geometry)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = DNaV # settings this here explicit for a future me, so I see it easier
                self.fns = DNaV # settings this here explicit for a future me, so I see it easier

        self.geometry = self.obs_lib.geometry # Sim_generator() needs this. I let obs_lib decide the final geometry.
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_sky(self, simidx, space, field, spin):
        return self.len_lib.get_sim_sky(simidx=simidx, space=space, field=field, spin=spin)

    def get_sim_unl(self, simidx, space, field, spin):
        return self.unl_lib.get_sim_unl(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, simidx, space, field, spin):
        return self.obs_lib.get_sim_obs(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
    
    def get_sim_phi(self, simidx, space):
        return self.unl_lib.get_sim_phi(simidx=simidx, space=space)
    

    def isdone(self, simidx, field, spin, space='map', flavour='obs'):
        fn = '{}_space{}_spin{}_field{}_{}'.format(flavour, space, spin, field, simidx)
        if self.obs_lib.cacher.is_cached(fn):
            return True
        if field == 'polarization':
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.fns[0].format(simidx))) and os.path.exists(opj(self.libdir, self.fns[1].format(simidx))):
                    return True
        if field == 'temperature':
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.fns.format(simidx))):
                    return True
        return False
        
        
    # compatibility with Plancklens
    def hashdict(self):
        return {}
    # compatibility with Plancklens
    def get_sim_tmap(self, simidx):
        return self.get_sim_obs(simidx=simidx, space='map', field='temperature', spin=0)
    # compatibility with Plancklens
    def get_sim_pmap(self, simidx):
        return self.get_sim_obs(simidx=simidx, space='map', field='polarization', spin=2)