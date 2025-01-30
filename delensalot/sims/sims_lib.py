"""sims/sims_lib.py: library for collecting and handling simulations. Eventually, delensalot needs a `get_sim_pmap()` and `get_sim_tmap()`, which are the maps of the observed sky. But data may come as cls, priensed alms, ..., . So this module allows to start analysis with,
    * cls,
    * alms_pri
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


def contains_DNaV(d, DNaV, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()
    
    if isinstance(d, dict):
        return any(
            key not in ignore_keys and contains_DNaV(v, DNaV, ignore_keys)
            for key, v in d.items()
        )
    return d == DNaV


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
        self.geom_lib = get_geom(self.geominfo)
        self.libdir = libdir
        self.spin = spin
        self.lmax = lmax
        self.space = space
        if libdir == DNaV:
            self.nlev = nlev
            assert libdir_suffix != DNaV, 'must give libdir_suffix'
            nlev_round = dict2roundeddict(self.nlev)
            self.libdir_phas = os.environ['SCRATCH']+'/simulation/{}/{}/phas/{}/'.format(libdir_suffix, get_dirname(str(self.geominfo)), get_dirname(str(sorted(nlev_round.items()))))
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
                            alm_buffer = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                            noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                            noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                            noise = np.array([noise1, noise2])
                    elif space == 'alm':
                        noise = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                elif field == 'temperature':
                    noise = self.nlev['T'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=0)
                    if space == 'alm':
                        noise = self.geom_lib.map2alm(noise, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
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
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif self.spin == 2:
                                noise = self.geom_lib.map2alm_spin(noise, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)  
                        elif space == 'map':
                            if self.spin != spin:
                                if self.spin == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(noise[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(noise[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    noise = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lm_max[0], spin=spin, mmax=self.lm_max[1], nthreads=4)
                                elif self.spin == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(noise, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    noise = np.array([noise1, noise2])
                    elif self.space == 'alm':
                        if space == 'map':
                            if spin == 0:
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif spin == 2:
                                noise = self.geom_lib.alm2map_spin(noise, spin=spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)       
                elif field == 'temperature':
                    noise = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                    if self.space == 'map':
                        if space == 'alm':
                            noise = self.geom_lib.map2alm(noise, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            noise = self.geom_lib.alm2map(noise, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            self.cacher.cache(fn, noise)  
        return self.cacher.load(fn)


class Cls:
    """class for accessing CAMB-like file for CMB power spectra, optionally a distinct file for the lensing field (grad and curl component), and birefringence
    """    
    def __init__(self, CMB_info=DNaV, sec_info={}):
        """
        only secondaries in dict will be generated even if Cls exist.
        secondaries = { 
            'phi':{
                'fn': CMB_fn,
                'components':['pp', 'ww'],
                'scale':'p',
            },
            'bf':{
                'fn': CMB_fn,
                'components':['ff'],
                'scale':'p',
            },
        }
        """

        # NOTE if CMB_fn is None, I assume the run does not have CMB spectra (this is the case when prialm CMB are provided, but secondaries alms are generated from Cls)
        # TODO add support for field-field (phi-bf) correlations
        if CMB_info['fns'] == DNaV:
            self.CMB_fn = opj(os.path.dirname(delensalot.__file__), 'data', 'cls', 'FFP10_wdipole_lenspotentialCls.dat')
            self.Cl_dict = load_file(self.CMB_fn)
            # FIXME need to initialize the secondaries with the same CMB_fn
        elif CMB_info['fns'] is None: # only need cl_dict for secondaries
            self.CMB_fn = None
            self.Cl_dict = {}
        else:
            self.CMB_fn = CMB_info['fns']
            self.Cl_dict = load_file(CMB_info['fns'])
        
        self.sec_info = sec_info

        # NOTE now I either replace or delete secondaries from the dict.
        # If there are secondaries listed in the secondaries parameter, I will replace the Cl_dct values with it.
        # If not, I will delete the Cl_dict entries as I assume this run is performed without them.
        # If fn point to the same CMB_fn, I keep them
        if sec_info is not None:
            secondaries_keep_list = [key for key, value in sec_info.items()]
            secondaries_pop_list = [key for key in ['pp', 'ww', 'ff'] if key not in secondaries_keep_list]
            for key, value in secondaries_pop_list:
                del self.Cl_dict[key]
            for key, value in sec_info.items():
                if value['fn'] is not CMB_info['fns']:
                    for component in value['components']:
                        self.Cl_dict[component] = load_file(value['fn'])[component]

        self.cacher = cachers.cacher_mem(safe=True)


    def get_clCMBpri(self, simidx, components=['tt', 'ee', 'bb', 'te'], lmax=None):
        fn = f'clcmb_{components}_{simidx}'
        if not self.cacher.is_cached(fn):
            Cls = np.array([self.Cl_dict[key][:lmax+1] for key in components]) if lmax is not None else np.array([self.Cl_dict[key] for key in components])
            self.cacher.cache(fn, Cls)
        return self.cacher.load(fn)   
        
    
    def get_clphi(self, simidx, components=['pp'], lmax=None):
        fn = f'clphi_{components}_{simidx}'
        if not self.cacher.is_cached(fn):
            Cls = np.array([self.Cl_dict[key][:lmax+1] for key in components]) if lmax is not None else np.array([self.Cl_dict[key] for key in components])
            self.cacher.cache(fn, np.array(Cls))
        return self.cacher.load(fn)


    def get_clbf(self, simidx, components=['ff'], lmax=None):
        fn = f'clbf_{components}_{simidx}'
        if not self.cacher.is_cached(fn):
            Cls = np.array([self.Cl_dict[key][:lmax+1] for key in components]) if lmax is not None else np.array([self.Cl_dict[key] for key in components])
            self.cacher.cache(fn, np.array(Cls))
        return self.cacher.load(fn)
    

    def get_clsec(self, simidx, secondary, components=['pp'], lmax=None):
        fn = f'cl{secondary}_{components}_{simidx}'
        if not self.cacher.is_cached(fn):
            Cls = np.array([self.Cl_dict[key][:lmax+1] for key in components]) if lmax is not None else np.array([self.Cl_dict[key] for key in components])
            self.cacher.cache(fn, np.array(Cls))
        return self.cacher.load(fn)
    

class Xpri:
    """class for generating primary CMB, and secondary realizations from power spectra
    """    
    def __init__(self, cls_lib=DNaV, geominfo=DNaV, CMB_info=DNaV, sec_info=DNaV):
        self.CMB_info = CMB_info
        self.sec_info = sec_info

        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)

        if CMB_info['libdir'] == DNaV and any(value['fn'] == DNaV for value in sec_info.values()):
            assert 0, 'need to provide either CMB alms or secondary alms, or use a different flavour to run this'

        if CMB_info['libdir'] == DNaV or any(value['fn'] == DNaV for value in sec_info.values()):
            if cls_lib == DNaV:
                secondaries = {key: {'fn':DNaV, 'components':value['components']} for key, value in sec_info.items()}
                self.cls_lib = Cls(CMB_fn=DNaV, secondaries=secondaries) # NOTE I pick all CMB components anyway
            else:
                self.cls_lib = cls_lib
        else:
            if CMB_info['libdir'] != DNaV:
                for key, value in CMB_info.items():
                    if value == DNaV:
                        assert 0, f'need to provide {key}'
            else:
                for key in ['space', 'scale', 'modifier', 'lm_max', 'fn', 'components']:
                    if any(value['space'] == DNaV for value in sec_info.values()):
                        assert 0, 'need to provide {key} for all secondaries'

        self.cacher = cachers.cacher_mem(safe=True)


    def get_sim_primordial(self, simidx, space, field, spin=2):
        """returns an priensed simulation field (temp,pol) in space (map, alm) and as spin (0,2). Note, spin is only applicable for pol, and returns QU for spin=2, and EB for spin=0.

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
        fn = f'primordial_space{space}_spin{spin}_field{field}_{simidx}'
        if not self.cacher.is_cached(fn):
            if self.CMB_info['libdir'] == DNaV:
                Cls = self.cls_lib.get_clCMBpri(simidx)
                pri = np.array(self.cl2alm(Cls, field=field, seed=simidx))
                if space == 'map':
                    if field == 'polarization':
                        if spin == 2:
                            pri = self.geom_lib.alm2map_spin(pri, lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif spin == 0:
                            pri1 = self.geom_lib.alm2map(pri[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            pri2 = self.geom_lib.alm2map(pri[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            pri = np.array([pri1, pri2])
                    elif field == 'temperature':
                        pri = self.geom_lib.alm2map(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            else:
                if field  == 'polarization':
                    if self.CMB_info['spin'] == 2:
                        pri1 = load_file(opj(self.CMB_info['libdir'], self.CMB_info['fns']['Q'].format(simidx)))
                        pri2 = load_file(opj(self.CMB_info['libdir'], self.CMB_info['fns']['U'].format(simidx)))
                    elif self.CMB_info['spin'] == 0:
                        pri1 = load_file(opj(self.CMB_info['libdir'], self.CMB_info['fns']['E'].format(simidx)))
                        pri2 = load_file(opj(self.CMB_info['libdir'], self.CMB_info['fns']['B'].format(simidx)))
                    pri =  np.array([pri1, pri2])
                    if self.CMB_info['space'] == 'map':
                        if space == 'alm':
                            if self.CMB_info['spin'] == 2:
                                pri = self.geom_lib.map2alm_spin(pri, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            elif self.CMB_info['spin'] == 0:
                                pri1 = self.geom_lib.map2alm(pri[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                pri2 = self.geom_lib.map2alm(pri[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                pri = np.array([pri1, pri2])
                        elif space == 'map':
                            if self.CMB_info['spin'] != spin:
                                if self.CMB_info['spin'] == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(pri[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(pri[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    pri = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                elif self.CMB_info['spin'] == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(pri, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    pri1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    pri2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    pri = np.array([pri1, pri2])
                    elif self.CMB_info['space'] == 'alm':
                        if space == 'map':
                            if spin == 0:
                                pri = self.geom_lib.alm2map(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            elif spin == 2:
                                pri = self.geom_lib.alm2map_spin(pri, spin=spin, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                elif field == 'temperature':
                    pri = np.array(load_file(opj(self.CMB_info['libdir'], self.CMB_info['fns']['T'].format(simidx))))
                    if self.CMB_info['space'] == 'map':
                        if space == 'alm':
                            pri = self.geom_lib.map2alm(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif self.CMB_info['space'] == 'alm':
                        if space == 'map':
                            pri = self.geom_lib.alm2map(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, pri)
        return self.cacher.load(fn)
    

    def get_sim_secondary(self, simidx, space, secondary, component=None):
        """ returns a secondary (phi, bf) in space (map, alm) and as component (grad, curl, if applicable). If secondary or component is None, it returns all, respectively.
        Args:
            simidx (_type_): _description_
            space (_type_): _description_

        Returns:
            _type_: _description_
        """    

        if secondary is None:
                return [self.get_sim_secondary(simidx, space, key, component=None) for key in self.sec_info.keys()]
        if component is None:
                return [self.get_sim_secondary(simidx, space, secondary, component=comp) for comp in self.sec_info[secondary]['components']]
        
        fn = f'{secondary}{component}_space{space}_{simidx}'
        if not self.cacher.is_cached(fn):
            if self.sec_info[secondary]['libdir'] == DNaV:
                log.debug(f'generating {secondary}{component} from cl')
                Clpf = self.cls_lib.get_clsec(simidx, secondary, component)
                self.sec_info[secondary]['scale'] = self.cls_lib.sec_info[secondary]['scale']
                Clp = self.clsecsf2clsecp(secondary, Clpf)
                sec = self.clp2seclm(secondary, Clp, simidx)
                ## If it comes from CL, like Gauss secs, then sec modification must happen here
                sec = self.sec_info[secondary]['modifier'](sec)
                if space == 'map':
                    sec = self.geom_lib.alm2map(sec, lmax=self.sec_info[secondary]['lm_max'][0], mmax=self.sec_info[secondary]['lm_max'][1], nthreads=4)
            else:
                ## Existing sec is loaded, this e.g. is a kappa map on disk
                if self.sec_info[secondary]['space'] == 'map':
                    sec = np.array(load_file(opj(self.sec_info[secondary]['libdir'], self.sec_info[secondary]['fns'][component].format(simidx))), dtype=float)
                else:
                    sec = np.array(load_file(opj(self.sec_info[secondary]['libdir'], self.sec_info[secondary]['fns'][component].format(simidx))), dtype=complex)
                if self.sec_info[secondary]['space'] == 'map':
                    self.geominfo_sec = ('healpix', {'nside':hp.npix2nside(sec.shape[0])})
                    self.geomlib_sec = get_geom(self.geominfo_sec)
                    sec = self.geomlib_sec.map2alm(sec, lmax=self.sec_info[secondary]['lm_max'][0], mmax=self.sec_info[secondary]['lm_max'][1], nthreads=4)
                ## sec modifcation
                sec = self.sec_info[secondary]['modifier'](sec)
                sec = self.pflm2plm(secondary, sec)
                if space == 'map':
                    sec = self.geom_lib.alm2map(sec, lmax=self.sec_info[secondary]['lm_max'][0], mmax=self.sec_info[secondary]['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, sec)
        return self.cacher.load(fn)
    

    def pflm2plm(self, secondary, seclm):
        # NOTE naming convention is sec, but it can be grad or curl
        if self.sec_info[secondary]['scale'] == 'convergence':
            return klm2plm(seclm, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'deflection':
            return dlm2plm(seclm, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'potential':
            return seclm


    def clsecsf2clsecp(self, secondary, cl):
        # NOTE naming convention is sec, but it can be grad or curl
        if self.sec_info[secondary]['scale'] == 'convergence':
            return clk2clp(cl, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'deflection':
            return cld2clp(cl, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'potential':
            return cl


    def cl2alm(self, cls, field, seed):
        np.random.seed(int(seed))
        if field == 'polarization':
            alms = hp.synalm(cls, self.CMB_info['lm_max'][0], new=True)
            return alms[1:]
        elif field == 'temperature':
            alm = hp.synalm(cls, self.CMB_info['lm_max'][0])
            return alm[0]
    

    def clp2seclm(self, secondary, clp, seed):
        np.random.seed(int(seed)+112233) # different seed for secondaries
        sec = hp.synalm(clp, self.sec_info[secondary]['lm_max'][0])
        return sec


class Xsky:
    """class for generating lensed CMB and phi realizations from priensed realizations, using lenspyx for the lensing operation
    """    
    def __init__(self, pri_lib=DNaV, geominfo=DNaV, CMB_info=DNaV, sec_info=DNaV, operators=DNaV):
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)

        if CMB_info['libdir'] == DNaV: # needs being generated
            if pri_lib != DNaV:
                self.pri_lib = pri_lib
            else:
                assert 0, 'need to provide pri_lib'
        else:
            for key, val in CMB_info.items():
                if val == DNaV:
                    assert 0, 'need to provide {}'.format(key)
        self.CMB_info = CMB_info

        self.sec_info = sec_info

        # if lenjob_geominfo == DNaV:
        #     self.lenjob_geominfo = ('thingauss', {'lmax':lmax+1024, 'smax':3})
        # else:
        #     self.lenjob_geominfo = lenjob_geominfo
        # self.lenjob_geomlib = lp_get_geom(self.lenjob_geominfo)
        self.operators = operators

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
        
        # NOTE Logic as follows: there is a cacher and a disk. If something is already in cache, no need to load it from disk. If spin X is requested but spin Y is stored, reuse, just convert. If none of it, generate
        fn = f'sky_space{space}_spin{spin}_field{field}_{simidx}'
        log.debug('requesting "{}"'.format(fn))
        if not self.cacher.is_cached(fn):
            fn_other = f'sky_space{space}_spin{self.CMB_info['spin']}_field{field}_{simidx}'
            if not self.cacher.is_cached(fn_other):
                log.debug('..nothing cached..')
                if self.CMB_info['libdir'] == DNaV:
                    log.debug('.., generating.')
                    pri = self.pri_lib.get_sim_pri(simidx, space='alm', field=field, spin=0)
                    philm = self.pri_lib.get_sim_phi(simidx, space='alm')
                    curllm = self.pri_lib.get_sim_curl(simidx, space='alm')
                    plms = [philm, curllm]
                    if field == 'polarization':
                        # TODO act operators
                        # for operator in self.operators:
                        #     pri = operator(pri)
                        sky = self.pri2len(pri, plms, spin=2)
                        bflm = self.pri_lib.get_sim_bf(simidx, space='map')
                        sky = self.pri2bf(pri, bflm, spin=2, epsilon=self.epsilon)

                        if space == 'map':
                            if spin == 0:
                                alm_buffer = self.lenjob_geomlib.map2alm_spin(sky, spin=2, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky = np.array([sky1, sky2])
                            elif spin == 2:
                                sky = self.lenjob_geomlib.map2alm_spin(np.copy(sky), spin=2, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky = self.geom_lib.alm2map_spin(np.copy(sky), lmax=self.CMB_info['lm_max'][0], spin=2, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm_spin(sky, lmax=self.CMB_info['lm_max'][0], spin=2, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif field == 'temperature':
                        sky = self.pri2len(pri, plms, spin=0, epsilon=self.epsilon)
                        if space == 'map':
                            sky = self.lenjob_geomlib.map2alm(np.copy(sky), lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            sky = self.geom_lib.alm2map(np.copy(sky), lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                else:
                    log.debug('.., but stored on disk.')
                    if field == 'polarization':
                        if self.CMB_info['spin'] == 2:
                            sky1 = load_file(opj(self.CMB_info['libdir'], self.fns['Q'].format(simidx)))
                            sky2 = load_file(opj(self.CMB_info['libdir'], self.fns['U'].format(simidx)))
                        elif self.CMB_info['spin'] == 0:
                            sky1 = load_file(opj(self.CMB_info['libdir'], self.fns['E'].format(simidx)))
                            sky2 = load_file(opj(self.CMB_info['libdir'], self.fns['B'].format(simidx)))
                        sky = np.array([sky1, sky2])
                        if self.space == 'map':
                            if space == 'alm':
                                if self.CMB_info['spin'] == 0:
                                    sky1 = self.geom_lib.map2alm(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky2 = self.geom_lib.map2alm(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky = np.array([sky1, sky2])
                                else:
                                    sky = self.geom_lib.map2alm_spin(sky, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            elif space == 'map':
                                if self.CMB_info['spin'] != spin:
                                    if self.CMB_info['spin'] == 0:
                                        alm_buffer1 = self.geom_lib.map2alm(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                        alm_buffer2 = self.geom_lib.map2alm(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                        sky = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    elif self.CMB_info['spin'] == 2:
                                        alm_buffer = self.geom_lib.map2alm_spin(sky, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                        sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                        sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                        sky = np.array([sky1, sky2])
                        elif self.space == 'alm':
                            if space == 'map':
                                if spin == 0:
                                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky = np.array([sky1, sky2])
                                else:
                                    sky = self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif field == 'temperature':
                        sky = np.array(load_file(opj(self.CMB_info['libdir'], self.fns['T'].format(simidx))))
                        if self.space == 'map':
                            if space == 'alm':
                                sky = self.geom_lib.map2alm(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif self.space == 'alm':
                            if space == 'map':
                                sky = self.geom_lib.alm2map(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            else:
                log.debug('found "{}"'.format(fn_other))
                sky = self.cacher.load(fn_other)
                if space == 'map':
                    sky = self.geom_lib.alm2map_spin(self.lenjob_geomlib.map2alm_spin(sky, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4), lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, np.array(sky))
        return self.cacher.load(fn)
    

    def pri2len(self, Xlm, plms, **kwargs):
        # if phi_lmax == DNaV:
        #     self.phi_lmax = lmax + 1024
        # if curl_lmax == DNaV:
        #     self.curl_lmax = lmax + 1024  
        ll = np.arange(0,self.pri_lib.phi_lmax+1,1)
        if len(plms) == 2:
            plm, olm = plms
            dplm = hp.almxfl(plm,  np.sqrt(ll*(ll+1)))
            dolm = hp.almxfl(olm,  np.sqrt(ll*(ll+1)))
            dlms = [dplm, dolm]
        elif len(plms) == 1:
            plm = plms[0]
            dplm = hp.almxfl(plm,  np.sqrt(ll*(ll+1)))
            dlms = [dplm]
        else:
            assert 0, 'wrong dimension of plms, should be a list of either gradient or gradient and curl'
        return lenspyx.alm2lenmap_spin(Xlm, dlms, geometry=self.lenjob_geominfo, **kwargs)
    
    def pri2bf(self, Xmap, bfmap, **kwargs):
        ll = np.arange(0,self.pri_lib.phi_lmax+1,1)
        return np.exp(-np.imag*bfmap)*Xmap


class Xobs:
    """class for generating/handling observed CMB realizations from sky maps together with a noise realization and transfer function to mimick an experiment
    """
    def __init__(self, maps=DNaV, sky_lib=DNaV, geominfo=DNaV, CMB_info=DNaV, obs_info=DNaV):
        self.CMB_info = CMB_info
        self.obs_info = obs_info
        
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)
        
        self.noise_lib = obs_info['noise_lib']
        
        self.maps = maps
        if np.all(self.maps != DNaV):
            fn = f'obs_space{CMB_info['space']}_spin{CMB_info['spin']}_field{CMB_info['field']}_0'
            self.cacher.cache(fn, np.array(self.maps))
        else:
            if CMB_info['libdir'] == DNaV:
                if sky_lib == DNaV:
                    assert 0, 'need to provide sky_lib'
                else:
                    self.sky_lib = sky_lib
                if np.all(obs_info['transfunction'] == DNaV):
                    assert 0, 'need to give transfunction'     
            else:
                for key, val in CMB_info.items():
                    if val == DNaV:
                        assert 0, 'need to provide {}'.format(key)

        self.fullsky = True #FIXME make it dependent on userdata: if Xobs is set via simhandler, then check if user data is full sky or not.
        self.cacher = cachers.cacher_mem(safe=True)


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
        fn = 'obs_space{space}_spin{spin}_field{field}_{simidx}'
        log.debug(f'requesting "{fn}"')
        fn_otherspin = f'obs_space{space}_spin{self.spin}_field{field}_{simidx}'
        fn_otherspace = ''
        fn_otherspacespin = ''
        if self.space == 'alm':
            fn_otherspace = f'obs_spacealm_spin0_field{field}_{simidx}'
        elif self.space == 'map':
            fn_otherspace = f'obs_spacemap_spin{spin}_field{field}_{simidx}'
        if self.space == f'alm':
            fn_otherspacespin = f'obs_spacealm_spin0_field{field}_{simidx}'
        elif self.space == 'map':
            fn_otherspacespin = f'obs_spacemap_spin{self.spin}_field{field}_{simidx}'

        if not self.cacher.is_cached(fn) and not self.cacher.is_cached(fn_otherspin) and not self.cacher.is_cached(fn_otherspacespin) and not self.cacher.is_cached(fn_otherspace):
            log.debug('..nothing cached..')
            if self.libdir == DNaV: # sky data comes from sky_lib, and we add noise
                log.debug('.., generating.')
                obs = self.sky2obs(
                    np.copy(self.sky_lib.get_sim_sky(simidx, spin=spin, space=space, field=field)),
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
                    obs1 = self.CMB_modifier(obs1)
                    obs2 = self.CMB_modifier(obs2)                
                    obs = np.array([obs1, obs2])
                    if self.space == 'map':
                        if space == 'map':
                            if self.spin != spin:
                                if self.spin == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    obs = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.lm_max[0], spin=spin, mmax=self.lm_max[1], nthreads=4)
                                elif self.spin == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    obs1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    obs2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                    obs = np.array([obs1, obs2])
                        elif space == 'alm':
                            if self.spin == 0:
                                obs1 = self.geom_lib.map2alm(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                obs2 = self.geom_lib.map2alm(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            if spin == 0:
                                obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.alm2map_spin(obs, lmax=self.lm_max[0], spin=spin, mmax=self.lm_max[1], nthreads=4)
                elif field == 'temperature':
                    obs = np.array(load_file(opj(self.libdir, self.fns['T'].format(simidx))))
                    obs = self.CMB_modifier(obs)
                    if self.space == 'map':
                        if space == 'alm':
                            obs = self.geom_lib.map2alm(obs, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    elif self.space == 'alm':
                        if space == 'map':
                            obs = self.geom_lib.alm2map(obs, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
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
                    obs1 = self.geom_lib.map2alm(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    obs2 = self.geom_lib.map2alm(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    obs = np.array([obs1, obs2])
                    obs = self.geom_lib.alm2map_spin(obs, lmax=self.lm_max[0], spin=self.spin, mmax=self.lm_max[1], nthreads=4)
                else:
                    obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    obs = np.array([obs1, obs2])
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspace):
            log.debug('found "{}"'.format(fn_otherspace))
            obs = np.array(self.cacher.load(fn_otherspace))
            if field == 'polarization':
                if self.space == 'alm':
                    if spin == 0:
                        obs1 = self.geom_lib.alm2map(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                        obs2 = self.geom_lib.alm2map(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                        obs = np.array([obs1, obs2])
                    elif spin == 2:
                        obs = self.geom_lib.alm2map_spin(obs, lmax=self.lm_max[0], spin=spin, mmax=self.lm_max[1], nthreads=4)
                elif self.space == 'map':
                    if self.spin == 0:
                        alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                        alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                        obs = np.array([alm_buffer1, alm_buffer2])
                    elif self.spin == 2:
                        obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            elif field == 'temperature':
                if self.space == 'alm': 
                    obs = self.geom_lib.alm2map(obs, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                elif self.space == 'map':
                    obs = self.geom_lib.map2alm(obs, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspacespin):
            log.debug('found "{}"'.format(fn_otherspacespin))
            obs = np.array(self.cacher.load(fn_otherspacespin))
            if self.space == 'alm':
                obs = self.geom_lib.alm2map_spin(obs, lmax=self.lm_max[0], spin=spin, mmax=self.lm_max[1], nthreads=4)
            elif self.space == 'map':
                obs = self.geom_lib.map2alm_spin(obs, spin=self.spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            self.cacher.cache(fn, obs)
        return self.cacher.load(fn)
    

    def sky2obs(self, sky, noise, spin, space, field):
        if field == 'polarization':
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.map2alm(sky[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    sky2 = self.geom_lib.map2alm(sky[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = self.geom_lib.map2alm_spin(sky, spin=spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            hp.almxfl(sky[0], self.transfunction, inplace=True)
            hp.almxfl(sky[1], self.transfunction, inplace=True)
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = np.array(self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4))
                return sky + noise
            else:
                return sky + noise
        elif field == 'temperature':
            if space == 'map':
                sky = self.geom_lib.map2alm(sky, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)
            hp.almxfl(sky, self.transfunction, inplace=True)
            if space == 'map':
                return np.array(self.geom_lib.alm2map(sky, lmax=self.lm_max[0], mmax=self.lm_max[1], nthreads=4)) + noise
            else:
                return sky + noise


    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
  

    simulationdata = DLENSALOT_Simulation(
        flavour = 'unl',
        maps = DNaV,
        CMB_info = {
            'libdir': DNaV,
            'space': 'cl',
            'spin': 0,
            'lm_max': [4096,4096],
            'fns': DNaV,
        },
        sec_info = {
            'phi':{
                'libdir': libdir_phi,
                'fn': fnsP,
                'components': ['pp', 'ww'],
                'space':'alm',
                'scale':'p',
                'modifier': phi_modifier,
                'lm_max': phi_lm_max,
            },
            'bf':{
                'libdir': libdir_bf,
                'fn': fnsBF,
                'components': ['ff'],
                'space':'alm',
                'scale':'p',
                'modifier': bf_modifier,
                'lm_max': bf_lm_max,
            },
        },
        obs_info = {
            'noise_info': {
                'libdir': libdir_noise,
                'fns': fnsnoise,
                'nlev': nlev,
                'space': space,
                'geominfo': self.geominfo,
                'libdir_suffix': libdir_suffix
            },
            'transfunction': transfunction,
        },
    ),

class Simhandler:
    """Entry point for data handling and generating simulations. Data can be cl, pri, len, or obs, .. and alms or maps. Simhandler connects the individual libraries and decides what can be generated. E.g.: If obs data provided, len data cannot be generated. This structure makes sure we don't "hallucinate" data

    """
    # def __init__(self, flavour, space, geominfo=DNaV, maps=DNaV, field=DNaV, cls_lib=DNaV, pri_lib=DNaV, sky_lib=DNaV, obs_lib=DNaV, noise_lib=DNaV, libdir=DNaV, libdir_noise=DNaV, libdir_phi=DNaV, fns=DNaV,
    #              fnsnoise=DNaV, fnsP=DNaV, fnsC=DNaV, fnsBF=DNaV,  lmax=DNaV, transfunction=DNaV, nlev=DNaV, spin=0, CMB_fn=DNaV, phi_fn=DNaV, bf_fn=DNaV, phi_field=DNaV, bf_field=DNaV, phi_space=DNaV, bf_space=DNaV,
    #              bf_lmax=DNaV, epsilon=1e-7, phi_lmax=DNaV, libdir_suffix=DNaV, lenjob_geominfo=DNaV, cacher=cachers.cacher_mem(safe=True), CMB_modifier=DNaV, phi_modifier=DNaV, bf_modifier=DNaV, fields=DNaV,
    #              curl_field=DNaV, curl_space=DNaV, curl_lmax=DNaV, fnsB=DNaV, libdir_bf=DNaV):
        
    def __init__(self, flavour, maps=DNaV, geominfo=DNaV, CMB_info=DNaV, sec_info=DNaV, obs_info=DNaV, operators=DNaV):
        """Entry point for simulation data handling.
        Simhandler() connects the individual librariers together accordingly, depending on the provided data.
        It never stores data on disk itself, only in memory.
        It never 'hallucinates' data, i.e. if obs data provided, it will not generate len data. 

        Args:
            flavour      (str): Can be in ['obs', 'sky', 'pri'] and defines the type of data provided.
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
            fnsBF         (str with formatter, optional): file names of the lensing potential provided. It expects `<filename{simidx}.something>, where `{simidx}` is used by the libraries to format the simulation index into the name. Defaults to DNaV.
            lmax         (int, optional): Maximum l of the data provided. Defaults to DNaV.
            transfunction(np.array, optional): transfer function. Defaults to DNaV.
            nlev         (dict, optional): noise level of the individual fields. It expects `{'T': <value>, 'P': <value>}. Defaults to DNaV.
            spin         (int, optional): the spin of the data provided. Defaults to 0. Always defaults to 0 for temperature.
            CMB_fn       (str, optional): path+name of the file of the power spectra of the CMB. Defaults to DNaV.
            phi_fn       (str, optional): path+name of the file of the power spectrum of the lensing potential. Defaults to DNaV.
            phi_field    (str, optional): the type of potential provided, can be in ['potential', 'deflection', 'convergence']. This simulation library will automatically rescale the field, if needded. Defaults to DNaV.
            phi_space    (str, optional): can be in ['map', 'alm', 'cl'] and defines the space of the lensing potential provided.. Defaults to DNaV.
            phi_lmax     (_type_, optional): the maximum multipole of the lensing potential. if simulation library perfroms lensing, it is advisable that `phi_lmax` is somewhat larger than `lmax` (+ ~512-1024). Defaults to DNaV.
            bf_lmax      (np.array, optional): beam function. Defaults to DNaV.
            bf_space      (np.array, optional): beam function. Defaults to DNaV.
            bf_field      (np.array, optional): beam function. Defaults to DNaV.
            bf_fn      (np.array, optional): beam function. Defaults to DNaV.
            epsilon      (float, optional): Lenspyx lensing accuracy. Defaults to 1e-7.
            CMB_modifier (callable, optional): operation defined in the callable will be applied to each of the input maps/alms/cls
            phi_modifier (callable, optional): operation defined in the callable will be applied to the input phi lms
            bf_modifier  (callable, optional): operation defined in the callable will be applied to the input beam function lms

        """
        if CMB_info['space'] == 'alm':
            assert CMB_info['spin'] == 0, "spin has to be 0 for alm space"

        if flavour == 'obs':
            assert CMB_info['space'] in ['map','alm'], "obs CMB data can only be in map or alm space"
            if np.all(maps == DNaV):
                for key, val in CMB_info.items():
                    if val == DNaV:
                        assert 0, 'need to provide {}'.format(key)
            else:
                for key in ['spin', 'lm_max', 'field']:
                    if CMB_info[key] == DNaV:
                        assert 0, 'need to provide {}'.format(key)
            
            self.obs_lib = Xobs(maps=maps, geominfo=geominfo, CMB_info=CMB_info)

            self.noise_lib = self.obs_lib.noise_lib
            self.libdir = self.obs_lib.libdir
            self.fns = self.obs_lib.fns
        else:
            if flavour == 'sky':
                assert CMB_info['space'] in ['map','alm'], "sky CMB data can only be in map or alm space"
                assert all(contains_DNaV(obs_info, DNaV)), "need to provide complete obs_info"

                self.sky_lib = Xsky(pri_lib=DNaV, geominfo=geominfo, CMB_info=CMB_info, sec_info=sec_info, operators=operators)
                
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.sky_lib.CMB_info['libdir']
                self.fns = self.sky_lib.CMB_info['fns']
            elif flavour == 'pri':
                assert all(contains_DNaV(sec_info, DNaV, ignore_keys={'libdir'})), "need to provide complete sec_info"
                assert all(contains_DNaV(obs_info, DNaV)), "need to provide complete obs_info"
                # TODO if space is cl need to initalize cls_lib, otherwise not needed
                sec_info = {key: {k:v for k,v in sec_info[key].items() if k in ['fn', 'components','scale']} for key in sec_info.keys()}
                self.cls_lib = Cls(CMB_info=CMB_info, sec_info=sec_info)
                self.pri_lib = Xpri(cls_lib=self.cls_lib, geominfo=geominfo, CMB_info=CMB_info, sec_info=sec_info)
                self.sky_lib = Xsky(pri_lib=self.pri_lib, geominfo=geominfo, CMB_info=CMB_info, sec_info=sec_info, operators=operators)
                
                self.libdir = self.pri_lib.libdir
                self.fns = self.pri_lib.fns
            
            if noise_info['libdir'] == DNaV:
                # FIXME this looks wrong
                noise_info = obs_info['noise_info']
                noise_lib = iso_white_noise(space='alm' if CMB_info['space']=='cl' else CMB_info['space'], geominfo=self.geominfo, noise_info=noise_info)
                obs_info.update({'noise_lib':noise_lib})
            self.obs_lib = Xobs(maps=DNaV, sky_lib=self.sky_lib, geominfo=geominfo, CMB_info=CMB_info, obs_info=obs_info)

        self.flavour = flavour
        self.maps = maps
        self.geominfo = self.obs_lib.geominfo # Sim_generator() needs this. I let obs_lib decide the final geominfo.
        self.spin = CMB_info['spin']
        self.lm_max = CMB_info['lm_max']
        self.space = CMB_info['space']
        self.nlev = noise_lib.nlev
        self.transfunction = obs_info['transfunction']


    def get_sim_sky(self, simidx, space, field, spin):
        return self.sky_lib.get_sim_sky(simidx=simidx, space=space, field=field, spin=spin)

    def get_sim_pri(self, simidx, space, field, spin):
        return self.pri_lib.get_sim_pri(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, simidx, space, field, spin):
        return self.obs_lib.get_sim_obs(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
    
    def get_sim_phi(self, simidx, space):
        return self.pri_lib.get_sim_phi(simidx=simidx, space=space)
    
    def get_sim_bf(self, simidx, space):
        return self.pri_lib.get_sim_bf(simidx=simidx, space=space)
    
    def purgecache(self):
        log.info('sims_lib: purging cachers to release memory')
        libs = ['obs_lib', 'noise_lib', 'pri_lib', 'sky_lib']
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
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.fns['Q'].format(simidx))) and os.path.exists(opj(self.libdir, self.fns['U'].format(simidx))):
                    return True
        if field == 'temperature':
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