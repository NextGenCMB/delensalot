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

    def __init__(self, nlev, nside, lmax=None, lib_dir=None, fns=None, spin=None, space=None, geometry=None):
        self.geometry = geometry
        if geometry is None:
            self.geometry = ('healpix', {'nside':nside})
        self.geom_lib = get_geom(geometry)
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
        self.space = space
        if lib_dir is None:        
            self.nlev = nlev
            lib_dir_phas = os.environ['SCRATCH']+'/delensalot/sims/{}/phas/'.format(str(geometry))
            self.pix_lib_phas = phas.pix_lib_phas(lib_dir_phas, 3, (self.geom_lib.npix(),))
        else:
            if fns is None:
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
            if self.lib_dir is None:
                vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
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
                        noise = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.lmax)
                elif field == 'temperature':
                    noise = self.nlev['T'] / vamin * self.pix_lib_phas.get_sim(simidx, idf=0)
                    if space == 'alm':
                        noise = self.geom_lib.map2alm(noise, lmax=self.lmax)
            else:
                if field == 'polarization':
                    noise1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                    noise2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
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
                    noise = np.array(load_file(opj(self.lib_dir, self.fns.format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            noise = self.geom_lib.map2alm(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            noise = self.geom_lib.alm2map(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, noise)  
        return self.cacher.load(fn)


class Cls:
    def __init__(self, lmax, CAMB_fn=None, simidxs=None, phi_fn=None, phi_field='potential'):
        self.lmax = lmax
        self.simidxs = simidxs
        if CAMB_fn is None:
            self.CAMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat')
            self.CAMB_file = load_file(self.CAMB_fn)
        else:
            self.CAMB_fn = CAMB_fn
            self.CAMB_file = load_file(CAMB_fn)
        if phi_fn == 'None':
            self.phi_fn = None
        elif phi_fn is None:
            self.phi_fn = self.CAMB_fn
            self.phi_file = load_file(self.phi_fn)['pp']
            self.phi_field = phi_field # assuming that CAMB file is 'potential'
        else:
            self.phi_fn = phi_fn
            self.phi_file = load_file(self.phi_fn)
            self.phi_field = phi_field
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
            ClP = self.phi_file
            self.cacher.cache(fn, np.array(ClP))
        return self.cacher.load(fn)   


class Xunl:
    def __init__(self, lmax, cls_lib=None, lib_dir=None, fns=None, fnsP=None, simidxs=None, lib_dir_phi=None, phi_field='potential', phi_space=None, space=None, nside=None, geometry=None):
        self.geometry = geometry
        if geometry is None:
            self.geometry = ('healpix', {'nside':nside})
        self.geom_lib = get_geom(geometry)
        self.lib_dir = lib_dir
        self.space = space
    
        self.lib_dir_phi = lib_dir_phi
        self.phi_field = phi_field
        self.phi_space = phi_space

        self.nside = nside
        if lib_dir is None or lib_dir_phi is None: # need being generated
            self.lmax = lmax
            self.simidxs = simidxs
            if cls_lib is None:
                self.cls_lib = Cls(lmax=lmax, phi_field=phi_field)
            else:
                self.cls_lib = cls_lib
        if lib_dir is not None:
            self.fns = fns
            if self.space is None:
                assert 0, 'need to give space (map or alm)'
        if lib_dir_phi is not None:
            self.fnsP = fnsP
            if self.phi_space is None:
                assert 0, 'need to give phi_space (map or alm)'
        self.lmax_phi = lmax + 1024
            
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher


    def get_sim_unl(self, simidx, space, field, spin=2, geometry=None):
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
            if self.lib_dir is None:
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
                    unl1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                    unl2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
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
                    unl = np.array(load_file(opj(self.lib_dir, self.fns.format(simidx))))
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
            if self.lib_dir_phi is None:
                Clpf = self.cls_lib.get_sim_clphi(simidx)
                self.phi_field = self.cls_lib.phi_field
                Clp = self.clpf2clppot(Clpf)
                phi = self.clp2plm(Clp, simidx)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.lmax_phi, mmax=self.lmax, nthreads=4)
            else:
                phi = load_file(opj(self.lib_dir_phi, self.fnsP.format(simidx)))
                if self.phi_space == 'map':
                    phi = self.geom_lib.map2alm(phi, lmax=self.lmax_phi, mmax=self.lmax, nthreads=4)
                phi = self.pflm2plm(phi)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.lmax_phi, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, phi)
        return self.cacher.load(fn)
    

    def pflm2plm(self, philm):
        if self.phi_field == 'kappa':
            return klm2plm(philm, self.lmax_phi)
        elif self.phi_field == 'deflection':
            return dlm2plm(philm, self.lmax_phi)
        elif self.phi_field == 'potential':
            return philm


    def clpf2clppot(self, cl):
        if self.phi_field == 'kappa':
            return clk2clp(cl, self.lmax_phi)
        elif self.phi_field == 'deflection':
            return cld2clp(cl, self.lmax_phi)
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
        plm = hp.synalm(clp, self.lmax_phi)
        return plm


class Xsky:
    def __init__(self, nside, lmax, unl_lib=None, lib_dir=None, fns=None, simidxs=None, spin=None, epsilon=1e-7, space=None, geometry=None):
        self.geometry = geometry
        if geometry is None:
            self.geometry = ('healpix', {'nside':nside})
        self.geom_lib = get_geom(geometry)
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
        self.space = space
        if lib_dir is None: # need being generated
            self.unl_lib = unl_lib
            self.simidxs = simidxs
            self.epsilon = epsilon
        else:
            if fns is None:
                assert 0, 'you need to provide fns' 
            self.fns = fns

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
        if not self.cacher.is_cached(fn):
            fn_other = 'len_space{}_spin{}_field{}_{}'.format(space, self.spin, field, simidx)
            if not self.cacher.is_cached(fn_other):
                if self.lib_dir is None:
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
                    if field == 'polarization':
                        sky1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                        sky2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
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
                        sky = np.array(load_file(opj(self.lib_dir, self.fns.format(simidx))))
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
        ll = np.arange(0,self.unl_lib.lmax_phi+1,1)
        return lenspyx.alm2lenmap_spin(Xlm, hp.almxfl(philm,  np.sqrt(ll*(ll+1))), geometry=self.geometry, **kwargs)


class Xobs:

    def __init__(self, lmax, maps=None, transfunction=None, len_lib=None, noise_lib=None, lib_dir=None, fns=None, simidxs=None, nside=None, nlev=None, lib_dir_noise=None, fnsnoise=None, spin=None, space=None, geometry=None):
        self.geometry = geometry
        if geometry is None:
            self.geometry = ('healpix', {'nside':nside})
        self.geom_lib = get_geom(geometry)
        self.simidxs = simidxs
        self.lib_dir = lib_dir
        self.spin = spin
        self.lmax = lmax
        self.nside = nside
        self.space = space
        
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher
        self.maps = maps
        if self.maps is not None:
            fn = 'pmap_spin{}_{}'.format(spin, 0)
            self.cacher.cache(fn, np.array(self.maps))
        else:
            if lib_dir is None:
                if len_lib is None:
                    assert 0, "Either len_lib or lib_dir must be not None"
                else:
                    self.len_lib = len_lib
                if noise_lib is None:
                    if lib_dir_noise is None:
                        if nside is None or nlev is None:
                            assert 0, "Need nside and nlev for generating noise"
                    self.noise_lib = iso_white_noise(nside=nside, nlev=nlev, lmax=lmax, fns=fnsnoise,lib_dir=lib_dir_noise, space=space, spin=spin, geometry=self.geometry)
                else:
                    self.noise_lib = noise_lib
                self.transfunction = transfunction       
            elif lib_dir is not None:
                if fns is None:
                    assert 0, 'you need to provide fns' 
                self.fns = fns


    def get_sim_obs(self, simidx, space, field, spin=2):
        # TODO this is missing field=T and field=X (cross), 
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
        fn = 'obs_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            fn_other = 'len_space{}_spin{}_field{}_{}'.format(space, self.spin, field, simidx)
            if not self.cacher.is_cached(fn_other):
                if self.lib_dir is None: # sky maps come from len_lib, and we add noise
                    obs = self.sky2obs(
                        self.len_lib.get_sim_sky(simidx, spin=spin, space=space, field=field),
                        self.noise_lib.get_sim_noise(simidx, spin=spin, field=field, space=space),
                        spin=spin,
                        space=space,
                        field=field)
                elif self.lib_dir is not None:  # observed maps are somewhere
                    if field == 'polarization':
                        obs1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                        obs2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
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
                                    # obs = self.geom_lib.alm2map_spin(self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4), lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
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
                                    obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                    obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                                    obs = np.array([obs1, obs2])
                                else:
                                    obs = self.geom_lib.alm2map_spin(obs, lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
                    elif field == 'temperature':
                        obs = np.array(load_file(opj(self.lib_dir, self.fns.format(simidx))))
                        if self.space == 'map':
                            if space == 'alm':
                                obs = self.geom_lib.map2alm(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        elif self.space == 'alm':
                            if space == 'map':
                                obs = self.geom_lib.alm2map(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                obs = np.array(self.cacher.load(fn_other))
                if space == 'map':
                    obs = self.geom_lib.alm2map_spin(self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4), lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
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
                    return np.array(self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.lmax, mmax=self.lmax, nthreads=4)) + noise
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

    def __init__(self, flavour, space=None, maps=None, cls_lib=None, unl_lib=None, obs_lib=None, len_lib=None, noise_lib=None, lib_dir_noise=None, lib_dir=None, lib_dir_phi=None, fns=None, fnsP=None, simidxs=None, nside=None, lmax=None, transfunction=None, nlev=None, fnsnoise=None, spin=None, CAMB_fn=None, clphi_fn=None, phi_field=None, phi_space=None, epsilon=1e-7, geometry=None):
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
            lib_dir_noise (_type_, optional): _description_. Defaults to None.
            lib_dir (_type_, optional): _description_. Defaults to None.
            lib_dir_phi (_type_, optional): _description_. Defaults to None.
            fns (_type_, optional): _description_. Defaults to None.
            fnsP (_type_, optional): _description_. Defaults to None.
            simidxs (_type_, optional): _description_. Defaults to None.
            nside (_type_, optional): _description_. Defaults to None.
            lmax (_type_, optional): _description_. Defaults to None.
            transfunction (_type_, optional): _description_. Defaults to None.
            nlev (_type_, optional): _description_. Defaults to None.
            fnsnoise (_type_, optional): _description_. Defaults to None.
            spin (_type_, optional): _description_. Defaults to None.
            CAMB_fn (_type_, optional): _description_. Defaults to None.
            epsilon (_type_, optional): _description_. Defaults to 1e-7.
        """
        self.spin = spin
        self.lmax = lmax
        if space == 'map':
            if flavour == 'obs':
                if lib_dir is not None:
                    self.lib_dir = lib_dir
                    if fns is None:
                        assert 0, 'you need to provide fns' 
                    self.fns = fns
                    self.simidxs = simidxs
                    self.nside = nside
                    self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, lib_dir=lib_dir, fns=fns, simidxs=simidxs, nside=nside, spin=spin, geometry=geometry) if obs_lib is None else obs_lib
            if flavour == 'sky':
                self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, lib_dir=lib_dir, fns=fns, space=space, simidxs=simidxs, nside=nside, spin=spin, epsilon=epsilon, geometry=geometry) if len_lib is None else len_lib
                self.obs_lib = Xobs(len_lib=self.len_lib, space=space, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, spin=spin, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
            if flavour == 'unl':
                assert 0, 'implement if needed'
        elif space in ['alm']:
            if flavour == 'obs':
                if lib_dir is not None:
                    self.lib_dir = lib_dir
                    if fns is None:
                        assert 0, 'you need to provide fns' 
                    self.fns = fns
                    self.simidxs = simidxs
                    self.nside = nside
                    self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, lib_dir=lib_dir, fns=fns, simidxs=simidxs, nside=nside, spin=spin, geometry=geometry) if obs_lib is None else obs_lib
            if flavour == 'sky':
                assert 0, 'implement if needed'
            if flavour == 'unl':
                self.spin = 0 # there are genrally no qlms, ulms, therefore here we can safely assume that data is spin0
                if (lib_dir_phi is None or lib_dir is None) and cls_lib is None:
                    cls_lib = Cls(lmax=lmax, CAMB_fn=CAMB_fn, phi_fn=clphi_fn, phi_field=phi_field, simidxs=simidxs)
                self.cls_lib = cls_lib # just to be safe..
                self.unl_lib = Xunl(lmax=lmax, lib_dir=lib_dir, fns=fns, fnsP=fnsP, phi_field=phi_field, simidxs=simidxs, lib_dir_phi=lib_dir_phi, space=space,  phi_space=phi_space, cls_lib=cls_lib, geometry=geometry) if unl_lib is None else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=self.spin, space=space, epsilon=epsilon, geometry=geometry)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, space=space, spin=self.spin, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
        if space == 'cl':
            if flavour == 'obs':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'sky':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'unl':
                self.spin = 0 # there are genrally no qcls, ucls, therefore here we can safely assume that data is spin0
                self.cls_lib = Cls(lmax=lmax, CAMB_fn=CAMB_fn, phi_fn=clphi_fn, phi_field=phi_field, simidxs=simidxs)
                self.unl_lib = Xunl(cls_lib=cls_lib, lmax=lmax, fnsP=fnsP, phi_field=phi_field, lib_dir_phi=lib_dir_phi, phi_space=phi_space, simidxs=simidxs, geometry=geometry)
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin, epsilon=epsilon, geometry=geometry)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, spin=spin, geometry=geometry)
                self.noise_lib = self.obs_lib.noise_lib
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_sky(self, simidx, space, field, spin):
        return self.len_lib.get_sim_sky(simidx=simidx, space=space, field=field, spin=spin)

    def get_sim_unl(self, simidx, space, field, spin):
        return self.unl_lib.get_sim_unl(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, simidx, space, field, spin):
        return self.obs_lib.get_sim_obs(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
    

    def isdone(self, simidx, field, spin, space='map', flavour='obs'):
        fn = '{}_space{}_spin{}_field{}_{}'.format(flavour, space, spin, field, simidx)
        if self.cacher.is_cached(fn):
            return True
        if field == 'polarization':
            if os.path.exists(opj(self.obs_lib.lib_dir, self.obs_lib.fns[0].format(simidx))) and os.path.exists(opj(self.obs_lib.lib_dir, self.obs_lib.fns[1].format(simidx))):
                return True
        if field == 'temperature':
            if os.path.exists(opj(self.obs_lib.lib_dir, self.obs_lib.fns.format(simidx))):
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