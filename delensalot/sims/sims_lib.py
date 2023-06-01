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

    def __init__(self, nlev_p, nside, lmax=None, lib_dir=None, fns=None, spin=None):
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
        if lib_dir is None:        
            self.nlev_p = nlev_p
            lib_dir_phas = os.environ['SCRATCH']+'/delensalot/sims/nside{}/phas/'.format(nside) # TODO phas should go to sims dir..
            self.pix_lib_phas = phas.pix_lib_phas(lib_dir_phas, 3, (hp.nside2npix(nside),))
        else:
            if fns is None:
                assert 0, "must provide fns"
            self.fns = fns

        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_noise(self, simidx, spin=2):
        fn = 'noise_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
                noise1 = self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=1)
                noise2 = self.nlev_p / vamin * self.pix_lib_phas.get_sim(simidx, idf=2)
                if spin == 0:
                    noise1, noise2 = hp.alm2map_spin(hp.map2alm_spin([noise1, noise2], spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
            else:
                noise1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                noise2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
                if self.spin != spin:
                    noise1, noise2 = hp.alm2map_spin(hp.map2alm_spin([noise1, noise2], spin=self.spin, lmax=self.lmax), lmax=self.lmax, spin=spin, nside=self.nside)
            self.cacher.cache(fn, np.array([noise1, noise2]))  
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
    def __init__(self, lmax, cls_lib=None, lib_dir=None, fns=None, fnsP=None, simidxs=None, lib_dir_phi=None, phi_field='potential', phi_space=None):
        self.lib_dir = lib_dir
        self.lib_dir_phi = lib_dir_phi
        self.phi_field = phi_field
        self.phi_space = phi_space
        if lib_dir is None or lib_dir_phi is None: # need being generated
            self.lmax = lmax
            self.simidxs = simidxs
            if cls_lib is None:
                self.cls_lib = Cls(lmax=lmax, phi_field=phi_field)
            else:
                self.cls_lib = cls_lib
        if lib_dir is not None:
            self.fns = fns
        if lib_dir_phi is not None:
            self.fnsP = fnsP
            if self.phi_space is None:
                assert 0, 'need to give phi_space (map or alm)'
        self.lmax_phi = lmax + 1024
            
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
        fn = 'unl_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            assert 0, 'implement'
            unl=None
            self.cacher.cache(fn, np.array(unl))
        return self.cacher.load(fn)
    

    def get_sim_unllm(self, simidx):
        fn = 'unllm_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Cls = self.cls_lib.get_TEBunl(simidx)
                Eunl, Bunl = self.cl2alm(Cls, seed=simidx)
            else:
                Eunl = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                Bunl = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))               
            self.cacher.cache(fn, np.array([Eunl, Bunl]))

        return self.cacher.load(fn)


    def get_sim_philm(self, simidx):
        fn = 'philm_{}'.format(simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir_phi is None:
                Clpf = self.cls_lib.get_sim_clphi(simidx)
                self.phi_field = self.cls_lib.phi_field
                Clp = self.clpf2clppot(Clpf)
                phi = self.clp2plm(Clp, simidx)
            else:
                phi = load_file(opj(self.lib_dir_phi, self.fnsP.format(simidx)))
                if self.phi_space == 'map':
                    phi = hp.map2alm(phi)
                phi = self.pflm2plm(phi)
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


    def cl2alm(self, cls, seed):
        np.random.seed(seed)
        alms = hp.synalm(cls, self.lmax, new=True)
        return alms[1:]
    

    def clp2plm(self, clp, seed):
        np.random.seed(seed)
        plm = hp.synalm(clp, self.lmax_phi)
        return plm


class Xsky:
    def __init__(self, nside, lmax, unl_lib=None, lib_dir=None, fns=None, simidxs=None, spin=None, epsilon=1e-7):
        self.lib_dir = lib_dir
        self.spin = spin
        self.nside = nside
        self.lmax = lmax
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
        fn = 'len_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            assert 0, 'implement'
            sky = None
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)


    def get_sim_skymap(self, simidx, spin=2):
        fn = 'sky_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            if self.lib_dir is None:
                Qunllm, Uunllm = self.unl_lib.get_sim_unllm(simidx)
                philm = self.unl_lib.get_sim_philm(simidx)
                sky = self.unl2len(np.array([Qunllm, Uunllm]), philm, spin=self.spin, epsilon=self.epsilon)
                if spin == 0:
                    sky = hp.alm2map_spin(hp.map2alm_spin(sky, spin=2, lmax=self.lmax), lmax=self.lmax, spin=0, nside=self.nside)
            else:
                sky1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                sky2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
                sky = np.array([sky1, sky2])
                if self.spin != spin:
                    sky = hp.alm2map_spin(hp.map2alm_spin(sky, spin=self.spin, lmax=self.lmax), lmax=self.lmax, spin=spin, nside=self.nside)
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)
    

    def unl2len(self, Xlm, philm, **kwargs):
        # This is always polarization for now, therefore hardcoding spin # FIXME once we support temp
        kwargs['spin'] = 2
        ll = np.arange(0,self.unl_lib.lmax_phi+1,1)
        return lenspyx.alm2lenmap_spin(Xlm, hp.almxfl(philm,  np.sqrt(ll*(ll+1))), geometry=('healpix', {'nside': self.nside}), **kwargs)


class Xobs:

    def __init__(self, lmax, maps=None, transfunction=None, len_lib=None, noise_lib=None, lib_dir=None, fns=None, simidxs=None, nside=None, nlev_p=None, lib_dir_noise=None, fnsnoise=None, spin=None):
        self.simidxs = simidxs
        self.lib_dir = lib_dir
        self.spin = spin
        self.lmax = lmax
        self.nside = nside
        
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
                        if nside is None or nlev_p is None:
                            assert 0, "Need nside and nlev_p for generating noise"
                    self.noise_lib = iso_white_noise(nside=nside, nlev_p=nlev_p, lmax=lmax, fns=fnsnoise,lib_dir=lib_dir_noise, spin=spin)
                else:
                    self.noise_lib = noise_lib
                self.transfunction = transfunction       
                self.nlev_p = nlev_p
            elif lib_dir is not None:
                if fns is None:
                    assert 0, 'you need to provide fns' 
                self.fns = fns


    def get_sim_obs(self, simidx, space, field, spin=2):
        """returns an observed simulation field (temp,pol,cross) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

        Args:
            simidx (_type_): _description_
            space (_type_): _description_
            field (_type_): _description_
            spin (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """        
        fn = 'len_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            assert 0, 'implement'
            sky = None
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)


    def get_sim_pmap(self, simidx, spin=2):  
        fn = 'pmap_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            fn_other = 'pmap_spin{}_{}'.format(self.spin, simidx)
            if not self.cacher.is_cached(fn_other):
                if self.lib_dir is None: # sky maps come from len_lib, and we add noise
                    obs = self.sky2obs(self.len_lib.get_sim_skymap(simidx, spin=spin), self.noise_lib.get_sim_noise(simidx, spin=spin), spin=spin)
                elif self.lib_dir is not None:  # observed maps are somewhere
                    obs1 = load_file(opj(self.lib_dir, self.fns[0].format(simidx)))
                    obs2 = load_file(opj(self.lib_dir, self.fns[1].format(simidx)))
                    obs = np.array([obs1, obs2])
                    if self.spin != spin:
                        obs = hp.alm2map_spin(hp.map2alm_spin(obs, spin=self.spin, lmax=self.lmax), lmax=self.lmax, spin=spin, nside=self.nside)
                self.cacher.cache(fn, np.array(obs))
            else:
                obs1, obs2 = self.cacher.load(fn_other)
                obs = hp.alm2map_spin(hp.map2alm_spin(np.array([obs1, obs2]), spin=self.spin, lmax=self.lmax), lmax=self.lmax, spin=spin, nside=self.nside)
                self.cacher.cache(fn, np.array(obs))
        return self.cacher.load(fn)
    

    def get_sim_tmap(self, simidx):  
        assert 0, 'implement if needed'


    def sky2obs(self, sky, noise, spin):
        sky = hp.map2alm_spin(sky, spin, lmax=self.lmax)
        hp.almxfl(sky[0], self.transfunction, inplace=True)
        hp.almxfl(sky[1], self.transfunction, inplace=True)
        beamed = np.array(hp.alm2map_spin(sky, self.nside, spin, hp.Alm.getlmax(elm.size)))
        return beamed+noise  # np.array([beamed[0] + noise[0], beamed[1] + noise[1]])
   
  
    def get_sim_noise(self, simidx, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin)
  

class Simhandler:

    def __init__(self, flavour, space=None, maps=None, cls_lib=None, unl_lib=None, obs_lib=None, len_lib=None, noise_lib=None, lib_dir_noise=None, lib_dir=None, lib_dir_phi=None, fns=None, fnsP=None, simidxs=None, nside=None, lmax=None, transfunction=None, nlev_p=None, fnsnoise=None, spin=None, CAMB_fn=None, clphi_fn=None, phi_field=None, phi_space=None, epsilon=1e-7):
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
            nlev_p (_type_, optional): _description_. Defaults to None.
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
                    self.obs_lib = Xobs(maps=maps, transfunction=transfunction, lmax=lmax, lib_dir=lib_dir, fns=fns, simidxs=simidxs, nside=nside, spin=spin) if obs_lib is None else obs_lib
            if flavour == 'sky':
                self.len_lib = Xsky(unl_lib=unl_lib, lmax=lmax, lib_dir=lib_dir, fns=fns, simidxs=simidxs, nside=nside, spin=spin, epsilon=epsilon) if len_lib is None else len_lib
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, spin=spin)
                self.noise_lib = self.obs_lib.noise_lib
            if flavour == 'unl':
                assert 0, 'implement if needed'
        elif space in ['alm']:
            if flavour == 'obs':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'sky':
                assert 0, 'implement if needed'
            if flavour == 'unl':
                self.spin = 0 # there are genrally no qlms, ulms, therefore here we can safely assume that data is spin0
                if (lib_dir_phi is None or lib_dir is None) and cls_lib is None:
                    cls_lib = Cls(lmax=lmax, CAMB_fn=CAMB_fn, phi_fn=clphi_fn, phi_field=phi_field, simidxs=simidxs)
                self.cls_lib = cls_lib # just to be safe..
                self.unl_lib = Xunl(lmax=lmax, lib_dir=lib_dir, fns=fns, fnsP=fnsP, phi_field=phi_field, simidxs=simidxs, lib_dir_phi=lib_dir_phi, phi_space=phi_space, cls_lib=cls_lib) if unl_lib is None else unl_lib
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=self.spin, epsilon=epsilon)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, spin=self.spin)
                self.noise_lib = self.obs_lib.noise_lib
        if space == 'cl':
            if flavour == 'obs':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'sky':
                assert 0, 'implement if needed' # unlikely this will ever be needed
            if flavour == 'unl':
                self.spin = 0 # there are genrally no qcls, ucls, therefore here we can safely assume that data is spin0
                self.cls_lib = Cls(lmax=lmax, CAMB_fn=CAMB_fn, phi_fn=clphi_fn, phi_field=phi_field, simidxs=simidxs)
                self.unl_lib = Xunl(cls_lib=cls_lib, lmax=lmax, fnsP=fnsP, phi_field=phi_field, lib_dir_phi=lib_dir_phi, phi_space=phi_space, simidxs=simidxs)
                self.len_lib = Xsky(unl_lib=self.unl_lib, lmax=lmax, simidxs=simidxs, nside=nside, spin=spin, epsilon=epsilon)
                self.obs_lib = Xobs(len_lib=self.len_lib, transfunction=transfunction, lmax=lmax, nlev_p=nlev_p, noise_lib=noise_lib, nside=nside, lib_dir_noise=lib_dir_noise, fnsnoise=fnsnoise, spin=spin)
                self.noise_lib = self.obs_lib.noise_lib
        self.cacher = cachers.cacher_mem(safe=True) #TODO might as well use a numpy cacher

    def get_sim_sky(self, simidx, space, field, spin):
        assert 0, 'implement'
        return self.len_lib.get_sim_sky(simidx=simidx, space=space, field=field, spin=spin)

    def get_sim_unl(self, simidx, space, field, spin):
        assert 0, 'implement'
        return self.unl_lib.get_sim_unl(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, simidx, space, field, spin):
        assert 0, 'implement'
        return self.unl_lib.get_sim_obs(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_pmap(self, simidx, spin=2):
        fn = 'pmap_spin{}_{}'.format(spin, simidx)
        if not self.cacher.is_cached(fn):
            self.cacher.cache(fn, self.obs_lib.get_sim_pmap(simidx, spin=spin))
        return self.cacher.load(fn)
    
    def get_sim_noise(self, simidx, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin)

    def get_sim_skymap(self, simidx, spin=2):
        return self.len_lib.get_sim_skymap(simidx, spin=spin)

    def get_sim_unllm(self, simidx):
        return self.unl_lib.get_sim_unllm(simidx)
    
    def isdone(self, simidx):
        fn = 'pmap_spin{}_{}'.format(2, simidx)
        if self.cacher.is_cached(fn):
            return True
        if os.path.exists(self.obs_lib.fns[0](simidx)):
            return True
    
    # compatibility with Plancklens
    def hashdict(self):
        return {}

    # compatibility with Plancklens
    def get_sim_tmap(self, simidx, spin=2):
        return self.obs_lib.get_sim_tmap(simidx, spin=spin)