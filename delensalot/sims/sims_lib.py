"""sims/sims_lib.py: library for collecting and handling simulations. Eventually, delensalot needs a `get_sim_pmap()` and `get_sim_tmap()`, which are the maps of the observed sky. But data may come as cls, unlensed alms, ..., . So this module allows to start analysis with,
    * cls,
    * alms_unl
    * alm_len + noise
    * obs_sky
    
and is a mapper between them, so that `get_sim_pmap()` and `get_sim_tmap()` always returns observed maps.
"""

import os
from os.path import join as opj
import numpy as np, healpy as hp

import logging
log = logging.getLogger(__name__)

import lenspyx
from lenspyx.lensing import get_geom as lp_get_geom
from plancklens.sims import phas
from delensalot.core import cachers
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV

import delensalot
from delensalot.utils import load_file, cli


def klm2plm(klm, lmax):
    k2p = 0.5 * np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)
    return hp.almxfl(klm, cli(k2p))

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


def get_dirname(s):
    return s.replace('(', '').replace(')', '').replace('{', '').replace('}', '').replace(' ', '').replace('\'', '').replace('\"', '').replace(':', '_').replace(',', '_').replace('[', '').replace(']', '')

def dict2roundeddict(d):
    for k,v in d.items():
        d[k] = np.around(v,3)
    return d

class iso_white_noise:
    """class for generating very simple isotropic white noise
    """
    def __init__(self, nlev, lmax=DNaV, libdir=DNaV, fns=DNaV, spin=DNaV, space=DNaV, geominfo=DNaV, libdir_suffix=DNaV):
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(geominfo)
        self.libdir = libdir
        self.spin = spin
        self.lmax = lmax
        self.space = space
        if libdir == DNaV:
            self.nlev = nlev
            assert libdir_suffix != DNaV, 'must give libdir_suffix'
            nlev_round = dict2roundeddict(self.nlev)
            self.libdir_phas = os.environ['SCRATCH']+'/simulation/{}/{}/phas/{}/'.format(libdir_suffix, get_dirname(str(geominfo)), get_dirname(str(sorted(nlev_round.items()))))
            self.pix_lib_phas = phas.pix_lib_phas(self.libdir_phas, 3, (self.geom_lib.npix(),))
        else:
            if fns == DNaV:
                assert 0, "must provide fns"
            self.fns = fns

        self.cacher = cachers.cacher_mem(safe=True)


    def get_sim_noise(self, simidx, space, field, spin=2):
        """_summary_

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
        if field == 'temperature' and 'T' not in self.nlev:
            assert 0, "need to provide T key in nlev"
        if field == 'polarization' and 'P' not in self.nlev:
            assert 0, "need to provide P key in nlev"
        fn = 'noise_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            if self.libdir == DNaV:
                if self.geominfo[0] == 'healpix':
                    vamin = np.sqrt(hp.nside2pixarea(self.geominfo[1]['nside'], degrees=True)) * 60
                else:
                    ## FIXME this is a rough estimate, based on total sky coverage / npix()
                    vamin =  np.sqrt(4*np.pi) * (180/np.pi) / self.geom_lib.npix() * 60
                if field == 'polarization':
                    noise1 = self.nlev['P'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=1)
                    noise2 = self.nlev['P'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=2) # TODO this always produces qu-noise in healpix geominfo?
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
                    noise = self.nlev['T'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=0)
                    if space == 'alm':
                        noise = self.geom_lib.map2alm(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                if field == 'polarization':
                    if self.spin == 2:
                        noise1 = load_file(opj(self.libdir, self.fns['Q'].format(simidx)))
                        noise2 = load_file(opj(self.libdir, self.fns['U'].format(simidx)))
                    elif self.spin == 0:
                        noise1 = load_file(opj(self.libdir, self.fns['E'].format(simidx)))
                        noise2 = load_file(opj(self.libdir, self.fns['B'].format(simidx)))
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
                    noise = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            noise = self.geom_lib.map2alm(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            noise = self.geom_lib.alm2map(noise, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, noise)  
        return self.cacher.load(fn)


class Cls:
    """class for accessing CAMB-like file for CMB power spectra, optionally a distinct file for the lensing potential
    """    
    def __init__(self, lmax=DNaV, phi_lmax=DNaV, CMB_fn=DNaV, phi_fn=DNaV, phi_field='potential'):
        assert lmax != DNaV, "need to provide lmax"
        self.lmax = lmax
        self.phi_lmax = phi_lmax
        if phi_lmax == DNaV:
            self.phi_lmax = lmax + 1024
        
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
            buff = load_file(self.phi_fn)
            if self.phi_fn.endswith('npy'):
                if len(buff) > 1:
                    self.phi_file = load_file(self.phi_fn)[:,1]
                else:
                    self.phi_file = load_file(self.phi_fn)
            else:
                self.phi_file = load_file(self.phi_fn)['pp']
            self.phi_field = phi_field # assuming that CAMB file is 'potential'
        else:
            self.phi_fn = phi_fn
            buff = load_file(self.phi_fn)
            if self.phi_fn.endswith('npy'):
                if len(buff) > 1:
                    self.phi_file = load_file(self.phi_fn)[:,1]
                else:
                    self.phi_file = load_file(self.phi_fn)
            else:
                self.phi_file = load_file(self.phi_fn)['pp']
            self.phi_field = phi_field
        log.debug("phi_fn is {}".format(self.phi_fn))
        self.cacher = cachers.cacher_mem(safe=True)


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
    """class for generating unlensed CMB and phi realizations from power spectra
    """    
    def __init__(self, lmax, cls_lib=DNaV, libdir=DNaV, fns=DNaV, fnsP=DNaV, libdir_phi=DNaV, phi_field='potential', phi_space=DNaV, phi_lmax=DNaV, space=DNaV, geominfo=DNaV, isfrozen=False, spin=DNaV):
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)
        self.libdir = libdir
        self.space = space
        self.spin = spin
        self.lmax = lmax
        self.libdir_phi = libdir_phi
        self.phi_lmax = phi_lmax
        
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
            #TODO make sure I didn't screw up phi_lmax
            if 'nside' in  self.geominfo[1]:
                geom_lmax = 3*self.geominfo[1]['nside']
            elif 'lmax' in  self.geominfo[1]:
                geom_lmax = self.geominfo[1]['lmax']
            else:
                geom_lmax = lmax + 1024
            self.phi_lmax = np.min([lmax + 1024, geom_lmax])
        self.isfrozen = isfrozen
            
        self.cacher = cachers.cacher_mem(safe=True)


    def get_sim_unl(self, simidx, space, field, spin=2):
        """returns an unlensed simulation field (temp,pol) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

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
                    if self.spin == 2:
                        unl1 = load_file(opj(self.libdir, self.fns['Q'].format(simidx)))
                        unl2 = load_file(opj(self.libdir, self.fns['U'].format(simidx)))
                    elif self.spin == 0:
                        unl1 = load_file(opj(self.libdir, self.fns['E'].format(simidx)))
                        unl2 = load_file(opj(self.libdir, self.fns['B'].format(simidx)))
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
                    unl = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            unl = self.geom_lib.map2alm(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            unl = self.geom_lib.alm2map(unl, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, unl)
        return self.cacher.load(fn)
    

    def get_sim_phi(self, simidx, space):
        """

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
                log.debug('generating phi from cl')
                Clpf = self.cls_lib.get_sim_clphi(simidx)
                self.phi_field = self.cls_lib.phi_field
                Clp = self.clpf2clppot(Clpf)
                phi = self.clp2plm(Clp, simidx)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
            else:
                # print('phi field at {} is {}'.format(opj(self.libdir_phi, self.fnsP.format(simidx)), self.phi_field))
                if self.phi_space == 'map':
                    phi = np.array(load_file(opj(self.libdir_phi, self.fnsP.format(simidx))), dtype=float)
                else:
                    phi = np.array(load_file(opj(self.libdir_phi, self.fnsP.format(simidx))), dtype=complex)
                if self.phi_space == 'map':
                    self.geominfo_phi = ('healpix', {'nside':hp.npix2nside(phi.shape[0])})
                    self.geomlib_phi = get_geom(self.geominfo_phi)
                    phi = self.geomlib_phi.map2alm(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
                phi = self.pflm2plm(phi)
                if space == 'map':
                    phi = self.geom_lib.alm2map(phi, lmax=self.phi_lmax, mmax=self.phi_lmax, nthreads=4)
            self.cacher.cache(fn, phi)
        return self.cacher.load(fn)
    

    def pflm2plm(self, philm):
        if self.phi_field == 'convergence':
            return klm2plm(philm, self.phi_lmax)
        elif self.phi_field == 'deflection':
            return dlm2plm(philm, self.phi_lmax)
        elif self.phi_field == 'potential':
            return philm


    def clpf2clppot(self, cl):
        if self.phi_field == 'convergence':
            return clk2clp(cl, self.phi_lmax)
        elif self.phi_field == 'deflection':
            return cld2clp(cl, self.phi_lmax)
        elif self.phi_field == 'potential':
            return cl


    def cl2alm(self, cls, field, seed):
        np.random.seed(int(seed)) # check if this starting point is random
        if field == 'polarization':
            alms = hp.synalm(cls, self.lmax, new=True)
            return alms[1:]
        elif field == 'temperature':
            alm = hp.synalm(cls, self.lmax)
            return alm[0]
    

    def clp2plm(self, clp, seed):
        np.random.seed(int(seed))
        plm = hp.synalm(clp, self.phi_lmax)
        return plm


class Xsky:
    """class for generating lensed CMB and phi realizations from unlensed realizations, using lenspyx for the lensing operation
    """    
    def __init__(self, lmax, unl_lib=DNaV, libdir=DNaV, fns=DNaV, spin=DNaV, epsilon=1e-7, space=DNaV, geominfo=DNaV, isfrozen=False, lenjob_geominfo=DNaV):
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)
        self.libdir = libdir
        self.fns = fns
        self.spin = spin
        self.lmax = lmax
        self.space = space
        if libdir == DNaV: # need being generated
            if unl_lib == DNaV:
                self.unl_lib = Xunl(lmax=lmax, geominfo=self.geominfo)
            else:
                self.unl_lib = unl_lib
            
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

        if lenjob_geominfo == DNaV:
            self.lenjob_geominfo = ('thingauss', {'lmax':lmax+1024, 'smax':3})
        else:
            self.lenjob_geominfo = lenjob_geominfo
        self.lenjob_geomlib = lp_get_geom(self.lenjob_geominfo)

        self.cacher = cachers.cacher_mem(safe=True)


    def get_sim_sky(self, simidx, space, field, spin=2):
        """returns a lensed simulation field (temperature, polarization) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

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
        log.debug('requesting "{}"'.format(fn))
        if not self.cacher.is_cached(fn):
            fn_other = 'sky_space{}_spin{}_field{}_{}'.format(space, self.spin, field, simidx)
            if not self.cacher.is_cached(fn_other):
                log.debug('..nothing cached..')
                if self.libdir == DNaV:
                    log.debug('.., generating.')
                    unl = self.unl_lib.get_sim_unl(simidx, space='alm', field=field, spin=0)
                    philm = self.unl_lib.get_sim_phi(simidx, space='alm')
                    
                    if field == 'polarization':
                        sky = self.unl2len(unl, philm, spin=2, epsilon=self.epsilon)
                        if space == 'map':
                            if spin == 0:
                                alm_buffer = self.lenjob_geomlib.map2alm_spin(sky, spin=2, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky = np.array([sky1, sky2])
                            elif spin == 2:
                                sky = self.lenjob_geomlib.map2alm_spin(np.copy(sky), spin=2, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                                sky = self.geom_lib.alm2map_spin(np.copy(sky), lmax=self.lmax, spin=2, mmax=self.lmax, nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm_spin(sky, lmax=self.lmax, spin=2, mmax=self.lmax, nthreads=4)
                    elif field == 'temperature':
                        sky = self.unl2len(unl, philm, spin=0, epsilon=self.epsilon)
                        if space == 'map':
                            sky = self.lenjob_geomlib.map2alm(np.copy(sky), lmax=self.lmax, mmax=self.lmax, nthreads=4)
                            sky = self.geom_lib.alm2map(np.copy(sky), lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                else:
                    log.debug('.., but stored on disk.')
                    if field == 'polarization':
                        if self.spin == 2:
                            sky1 = load_file(opj(self.libdir, self.fns['Q'].format(simidx)))
                            sky2 = load_file(opj(self.libdir, self.fns['U'].format(simidx)))
                        elif self.spin == 0:
                            sky1 = load_file(opj(self.libdir, self.fns['E'].format(simidx)))
                            sky2 = load_file(opj(self.libdir, self.fns['B'].format(simidx)))
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
                        sky = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                        if self.space == 'map':
                            if space == 'alm':
                                sky = self.geom_lib.map2alm(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                        elif self.space == 'alm':
                            if space == 'map':
                                sky = self.geom_lib.alm2map(sky, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            else:
                log.debug('found "{}"'.format(fn_other))
                sky = self.cacher.load(fn_other)
                if space == 'map':
                    sky = self.geom_lib.alm2map_spin(self.lenjob_geomlib.map2alm_spin(sky, spin=self.spin, lmax=self.lmax, mmax=self.lmax, nthreads=4), lmax=self.lmax, spin=spin, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)
    

    def unl2len(self, Xlm, philm, **kwargs):
        ll = np.arange(0,self.unl_lib.phi_lmax+1,1)
        return lenspyx.alm2lenmap_spin(Xlm, hp.almxfl(philm,  np.sqrt(ll*(ll+1))), geometry=self.lenjob_geominfo, **kwargs)


class Xobs:
    """class for generating/handling observed CMB realizations from sky maps together with a noise realization and transfer function to mimick an experiment
    """
    def __init__(self, lmax, maps=DNaV, transfunction=DNaV, len_lib=DNaV, unl_lib=DNaV, epsilon=DNaV, noise_lib=DNaV, libdir=DNaV, fns=DNaV, nlev=DNaV, libdir_noise=DNaV, fnsnoise=DNaV, spin=DNaV, space=DNaV, geominfo=DNaV, field=DNaV, cacher=DNaV, libdir_suffix=DNaV, modifier=DNaV):
        if modifier == DNaV:
            self.modifier = lambda x: x
        else:
            self.modifier = modifier
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)
        
        self.libdir = libdir
        self.fns = fns
        self.spin = spin
        self.lmax = lmax
        self.space = space
        self.noise_lib = noise_lib
        self.fullsky = True #FIXME make it dependent on userdata: if Xobs is set via simhandler, then check if user data is full sky or not.
        self.cacher = cachers.cacher_mem(safe=True)
        self.maps = maps
        if np.all(self.maps != DNaV):
            fn = 'obs_space{}_spin{}_field{}_{}'.format(space, spin, field, 0)
            self.cacher.cache(fn, np.array(self.maps))
        else:
            if libdir == DNaV:
                if len_lib == DNaV:
                    self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, libdir=libdir, fns=fns, space=space, epsilon=epsilon, geominfo=geominfo)
                else:
                    self.len_lib = len_lib
                if noise_lib == DNaV:
                    if libdir_noise == DNaV:
                        if nlev == DNaV:
                            assert 0, "need nlev for generating noise"
                        self.noise_lib = iso_white_noise(nlev=nlev, lmax=lmax, fns=fnsnoise,libdir=libdir_noise, space=space, geominfo=self.geominfo, libdir_suffix=libdir_suffix)
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
        """_summary_

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
        log.debug('requesting "{}"'.format(fn))
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
            log.debug('..nothing cached..')
            if self.libdir == DNaV: # sky data comes from len_lib, and we add noise
                log.debug('.., generating.')
                obs = self.sky2obs(
                    np.copy(self.len_lib.get_sim_sky(simidx, spin=spin, space=space, field=field)),
                    np.copy(self.noise_lib.get_sim_noise(simidx, spin=spin, field=field, space=space)),
                    spin=spin,
                    space=space,
                    field=field)
            elif self.libdir != DNaV:  # observed data is somewhere
                log.debug('.., but stored on disk.')
                if field == 'polarization':
                    if self.spin == 2:
                        if self.fns['Q'] == self.fns['U'] and self.fns['Q'].endswith('.fits'):
                            # Assume implicitly that Q is field=1, U is field=2
                            obs1 = load_file(opj(self.libdir, self.fns['Q'].format(simidx)), ifield=1)
                            obs2 = load_file(opj(self.libdir, self.fns['U'].format(simidx)), ifield=2)
                        else:
                            obs1 = load_file(opj(self.libdir, self.fns['Q'].format(simidx)))
                            obs2 = load_file(opj(self.libdir, self.fns['U'].format(simidx)))
                    elif self.spin == 0:
                        if self.fns['E'] == self.fns['B'] and self.fns['B'].endswith('.fits'):
                            # Assume implicitly that E is field=1, B is field=2
                            obs1 = load_file(opj(self.libdir, self.fns['E'].format(simidx)), ifield=1)
                            obs2 = load_file(opj(self.libdir, self.fns['B'].format(simidx)), ifield=2)
                        else:
                            obs1 = load_file(opj(self.libdir, self.fns['E'].format(simidx)))
                            obs2 = load_file(opj(self.libdir, self.fns['B'].format(simidx)))
                    obs1 = self.modifier(obs1)
                    obs2 = self.modifier(obs2)                
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
                    obs = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                    obs = self.modifier(obs)
                    if self.space == 'map':
                        if space == 'alm':
                            obs = self.geom_lib.map2alm(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            obs = self.geom_lib.alm2map(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                self.cacher.cache(fn, obs)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn):
            log.debug('found "{}"'.format(fn))
            pass
        elif self.cacher.is_cached(fn_otherspin):
            log.debug('found "{}"'.format(fn_otherspin))
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
            log.debug('found "{}"'.format(fn_otherspace))
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
                    obs = self.geom_lib.alm2map(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
                elif self.space == 'map':
                    obs = self.geom_lib.map2alm(obs, lmax=self.lmax, mmax=self.lmax, nthreads=4)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspacespin):
            log.debug('found "{}"'.format(fn_otherspacespin))
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
    """Entry point for data handling and generating simulations. Data can be cl, unl, len, or obs, .. and alms or maps. Simhandler connects the individual libraries and decides what can be generated. E.g.: If obs data provided, len data cannot be generated. This structure makes sure we don't "hallucinate" data

    """
    def __init__(self, flavour, space, geominfo=DNaV, maps=DNaV, field=DNaV, cls_lib=DNaV, unl_lib=DNaV, len_lib=DNaV, obs_lib=DNaV, noise_lib=DNaV, libdir=DNaV, libdir_noise=DNaV, libdir_phi=DNaV, fns=DNaV, fnsnoise=DNaV, fnsP=DNaV, lmax=DNaV, transfunction=DNaV, nlev=DNaV, spin=0, CMB_fn=DNaV, phi_fn=DNaV, phi_field=DNaV, phi_space=DNaV, epsilon=1e-7, phi_lmax=DNaV, libdir_suffix=DNaV, lenjob_geominfo=DNaV, cacher=cachers.cacher_mem(safe=True), modifier=DNaV):
        """Entry point for simulation data handling.
        Simhandler() connects the individual librariers together accordingly, depending on the provided data.
        It never stores data on disk itself, only in memory.
        It never 'hallucinates' data, i.e. if obs data provided, it will not generate len data. 

        Args:
            flavour      (str): Can be in ['obs', 'sky', 'unl'] and defines the type of data provided.
            space        (str): Can be in ['map', 'alm', 'cl'] and defines the space of the data provided.
            maps         (np.array, optional): These maps will be put into the cacher directly. They are used for settings in which no data is generated or accesed on disk, but directly provided (like in `delensalot.anafast()`) Defaults to DNaV.
            geominfo     (tuple, optional): Lenspyx geominfo descriptor, describes the geominfo of the data provided (e.g. `('healpix', 'nside': 2048)). Defaults to DNaV.
            field        (str, optional): the type of data provided, can be in ['temperature', 'polarization']. Defaults to DNaV.
            libdir       (str, optional): directory of the data provided. Defaults to DNaV.
            libdir_noise (str, optional): directory of the noise provided. Defaults to DNaV.
            libdir_phi   (str, optional): directory of the lensing potential provided. Defaults to DNaV.
            fns          (dict with str with formatter, optional): file names of the data provided. It expects `{'T': <filename{simidx}.something>, 'Q': <filename{simidx}.something>, 'U': <filename{simidx}.something>}`, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
            fnsnoise     (dict with str with formatter, optional): file names of the noise provided. It expects `{'T': <filename{simidx}.something>, 'Q': <filename{simidx}.something>, 'U': <filename{simidx}.something>}`, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
            fnsP         (str with formatter, optional): file names of the lensing potential provided. It expects `<filename{simidx}.something>, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
            lmax         (int, optional): Maximum l of the data provided. Defaults to DNaV.
            transfunction(np.array, optional): transfer function. Defaults to DNaV.
            nlev         (dict, optional): noise level of the individual fields. It expects `{'T': <value>, 'P': <value>}. Defaults to DNaV.
            spin         (int, optional): the spin of the data provided. Defaults to 0. Always defaults to 0 for temperature.
            CMB_fn       (str, optional): path+name of the file of the power spectra of the CMB. Defaults to DNaV.
            phi_fn       (str, optional): path+name of the file of the power spectrum of the lensing potential. Defaults to DNaV.
            phi_field    (str, optional): the type of potential provided, can be in ['potential', 'deflection', 'convergence']. This simulation library will automatically rescale the field, if needded. Defaults to DNaV.
            phi_space    (str, optional): can be in ['map', 'alm', 'cl'] and defines the space of the lensing potential provided.. Defaults to DNaV.
            phi_lmax     (_type_, optional): the maximum multipole of the lensing potential. if simulation library perfroms lensing, it is advisable that `phi_lmax` is somewhat larger than `lmax` (+ ~512-1024). Defaults to DNaV.
            epsilon      (float, optional): Lenspyx lensing accuracy. Defaults to 1e-7.
        """
        self.spin = spin
        self.lmax = lmax
        self.phi_lmax = phi_lmax
        self.flavour = flavour
        self.space = space
        self.nlev = nlev
        self.maps = maps
        self.transfunction = transfunction
        if space == 'map':
            if flavour == 'obs':
                if np.all(maps == DNaV):
                    assert libdir != DNaV, "need to provide libdir"
                    assert fns != DNaV, 'you need to provide fns' 
                    assert lmax != DNaV, "need to provide lmax"
                    assert spin != DNaV, "need to provide spin"
                else:
                    assert spin != DNaV, "need to provide spin"
                    assert lmax != DNaV, "need to provide lmax"
                    assert field != DNaV, "need to provide field"
                self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, libdir=libdir, fns=fns, spin=spin, geominfo=geominfo, field=field, libdir_suffix=libdir_suffix, modifier=modifier) if obs_lib == DNaV else obs_lib
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
                self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, libdir=libdir, fns=fns, space=space, spin=spin, epsilon=epsilon, geominfo=geominfo, lenjob_geominfo=lenjob_geominfo) if len_lib == DNaV else len_lib
                self.obs_lib = Xobs(len_lib=self.len_lib, space=space, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, geominfo=geominfo, libdir_suffix=libdir_suffix)
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
                    self.unl_lib = Xunl(lmax=lmax, libdir=libdir, fns=fns, fnsP=fnsP, phi_field=phi_field, libdir_phi=libdir_phi, space=space, phi_space=phi_space, phi_lmax=phi_lmax, geominfo=geominfo, spin=spin) if unl_lib == DNaV else unl_lib
                elif libdir_phi == DNaV:
                    assert phi_fn != DNaV, "need to provide phi_fn"
                    assert phi_lmax != DNaV, "need to provide phi_lmax"
                    assert phi_field != DNaV, "need to provide phi_field"
                    assert phi_space == 'cl', "please set phi_space='cl', just to be sure."
                    self.cls_lib = Cls(phi_lmax=phi_lmax, phi_fn=phi_fn, phi_field=phi_field)
                    self.unl_lib = Xunl(cls_lib=self.cls_lib, lmax=lmax, libdir=libdir, fns=fns, phi_field=phi_field, space=space, phi_space=phi_space, phi_lmax=phi_lmax, geominfo=geominfo, spin=spin) if unl_lib == DNaV else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, space=space, epsilon=epsilon, geominfo=geominfo, lenjob_geominfo=lenjob_geominfo)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, space=space, geominfo=geominfo, libdir_suffix=libdir_suffix)
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
                    self.obs_lib = Xobs(maps=maps, space=space, transfunction=transfunction, lmax=lmax, libdir=libdir, fns=fns, spin=self.spin, geominfo=geominfo, libdir_suffix=libdir_suffix) if obs_lib == DNaV else obs_lib
                    self.noise_lib = self.obs_lib.noise_lib
                    self.libdir = self.obs_lib.libdir
                    self.fns = self.obs_lib.fns
            if flavour == 'sky':
                assert 0, 'implement if needed'
            if flavour == 'unl':
                if (libdir_phi == DNaV or libdir == DNaV) and cls_lib == DNaV:
                    cls_lib = Cls(lmax=lmax, CMB_fn=CMB_fn, phi_fn=phi_fn, phi_field=phi_field)
                self.cls_lib = cls_lib # just to be safe..
                self.unl_lib = Xunl(lmax=lmax, libdir=libdir, fns=fns, fnsP=fnsP, phi_field=phi_field, libdir_phi=libdir_phi, space=space, phi_space=phi_space, cls_lib=cls_lib, geominfo=geominfo, spin=self.spin) if unl_lib == DNaV else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, space=space, epsilon=epsilon, geominfo=geominfo, lenjob_geominfo=lenjob_geominfo)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, space=space, spin=self.spin, geominfo=geominfo, libdir_suffix=libdir_suffix)
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
                
                self.cls_lib = Cls(lmax=lmax, phi_lmax=phi_lmax, CMB_fn=CMB_fn, phi_fn=phi_fn, phi_field=phi_field)
                self.unl_lib = Xunl(cls_lib=self.cls_lib, lmax=lmax, fnsP=fnsP, phi_field=phi_field, libdir_phi=libdir_phi, phi_space=phi_space, geominfo=geominfo)
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, epsilon=epsilon, geominfo=geominfo, lenjob_geominfo=lenjob_geominfo)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev=nlev, noise_lib=noise_lib, libdir_noise=libdir_noise, fnsnoise=fnsnoise, geominfo=geominfo, cacher=cacher, libdir_suffix=libdir_suffix)
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = DNaV # settings this here explicit for a future me, so I see it easier
                self.fns = DNaV # settings this here explicit for a future me, so I see it easier

        self.geominfo = self.obs_lib.geominfo # Sim_generator() needs this. I let obs_lib decide the final geominfo.

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
    
    def purgecache(self):
        print('sims_lib: purging cachers to release memory')
        libs = ['obs_lib', 'noise_lib', 'unl_lib', 'len_lib']
        for lib in libs:
            if lib in self.__dict__:
                if len(list(self.obs_lib.cacher._cache.keys())) > 0:
                    for key in np.copy(list(self.obs_lib.cacher._cache.keys())):
                        self.obs_lib.cacher.remove(key)

    def isdone(self, simidx, field, spin, space='map', flavour='obs'):
        fn = '{}_space{}_spin{}_field{}_{}'.format(flavour, space, spin, field, simidx)
        if self.obs_lib.cacher.is_cached(fn):
            return True
        if field == 'polarization':
            # print(opj(self.libdir, self.fns['Q'].format(simidx)))
            # print(opj(self.libdir, self.fns['U'].format(simidx)))
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.fns['Q'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['U'].format(simidx))):
                    return True
        if field == 'temperature':
            # print(opj(self.libdir, self.fns['T'].format(simidx)))
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.fns['T'].format(simidx))):
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


class anafast_clone():

    def __init__(self, geominfo):
        self.nside = geominfo[1]['nside']
        self.npixel = hp.nside2npix(self.nside)

    def map2alm(self, map, lmax, mmax, nthreads):
        return hp.map2alm(map, lmax=lmax)

    def map2alm_spin(self, map, spin, lmax, mmax, nthreads):
        return hp.map2alm_spin(map, lmax=lmax, spin=spin)

    def alm2map(self, alm, lmax, mmax, nthreads):
         return hp.alm2map(alm, lmax=lmax, nside=self.nside)

    def alm2map_spin(self, alm, spin, lmax, mmax, nthreads):
        return hp.alm2map_spin(alm, lmax=lmax, spin=spin, nside=self.nside)

    def npix(self):
        return self.npixel

def get_geom(geominfo):
    return lp_get_geom(geominfo)
    # return anafast_clone(geominfo)