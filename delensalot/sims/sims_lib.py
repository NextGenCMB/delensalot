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
import copy

import hashlib
import logging
log = logging.getLogger(__name__)

import lenspyx
from lenspyx.lensing import get_geom as lp_get_geom
from plancklens.sims import phas
from delensalot.core import cachers
from delensalot.config.metamodel import DEFAULT_NotAValue as DNaV

from delensalot.utility.utils_hp import Alm, almxfl, alm_copy
import delensalot
from delensalot.utils import load_file_wsec, cli
from delensalot.sims import operator_secondary


def check_dict(d):
    for key, val in d.items():
        if isinstance(val, dict) or isinstance(val, list) or isinstance(val, np.ndarray):
            if np.any(val == DNaV):
                assert 0, 'need to provide {}'.format(key)
        else:
            if val == DNaV:
                assert 0, 'need to provide {}'.format(key)

def contains_DNaV(d, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()
    
    if isinstance(d, dict):
        return any(
            key not in ignore_keys and contains_DNaV(v, ignore_keys)
            for key, v in d.items()
        )
    elif isinstance(d, list) or isinstance(d, np.ndarray):
        return any(v == DNaV for v in d)
    else:
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
    def __init__(self, geominfo=DNaV, noise_info=DNaV):
        self.geominfo = geominfo
        if geominfo == DNaV:
            self.geominfo = ('healpix', {'nside':2048})
        self.geom_lib = get_geom(self.geominfo)

        if noise_info['libdir'] == DNaV:
            assert noise_info['libdir_suffix'] != DNaV, 'must give libdir_suffix'
            nlev_round = dict2roundeddict(noise_info['nlev'])
            self.libdir_phas = os.environ['SCRATCH']+'/simulation/{}/{}/phas/{}/'.format(noise_info['libdir_suffix'], get_dirname(str(self.geominfo)), get_dirname(str(sorted(nlev_round.items()))))
            self.pix_lib_phas = phas.pix_lib_phas(self.libdir_phas, 3, (self.geom_lib.npix(),))
        else:
            if noise_info['fns'] == DNaV:
                assert 0, "must provide fns"
        self.noise_info = noise_info

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
        if field == 'temperature' and 'T' not in self.noise_info['nlev']:
            assert 0, "need to provide T key in nlev"
        if field == 'polarization' and 'P' not in self.noise_info['nlev']:
            assert 0, "need to provide P key in nlev"
        fn = 'noise_space{}_spin{}_field{}_{}'.format(space, spin, field, simidx)
        if not self.cacher.is_cached(fn):
            if self.noise_info['libdir'] == DNaV:
                if self.geominfo[0] == 'healpix':
                    vamin = np.sqrt(hp.nside2pixarea(self.geominfo[1]['nside'], degrees=True)) * 60
                else:
                    ## FIXME this is a rough estimate, based on total sky coverage / npix()
                    vamin =  np.sqrt(4*np.pi) * (180/np.pi) / self.geom_lib.npix() * 60
                if field == 'polarization':
                    noise1 = self.noise_info['nlev']['P'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=1)
                    noise2 = self.noise_info['nlev']['P'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=2) # TODO this always produces qu-noise in healpix geominfo?
                    noise = np.array([noise1, noise2])
                    if space == 'map':
                        if spin == 0:
                            alm_buffer = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                            noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                            noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                            noise = np.array([noise1, noise2])
                    elif space == 'alm':
                        noise = self.geom_lib.map2alm_spin(noise, spin=2, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                elif field == 'temperature':
                    noise = self.noise_info['nlev']['T'] / vamin * self.pix_lib_phas.get_sim(int(simidx), idf=0)
                    if space == 'alm':
                        noise = self.geom_lib.map2alm(noise, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
            else:
                if field == 'polarization':
                    if self.noise_info['spin'] == 2:
                        noise1 = load_file_wsec(opj(self.noise_info['libdir'], self.noise_info['fns']['Q'].format(simidx)))
                        noise2 = load_file_wsec(opj(self.noise_info['libdir'], self.noise_info['fns']['U'].format(simidx)))
                    elif self.noise_info['spin'] == 0:
                        noise1 = load_file_wsec(opj(self.noise_info['libdir'], self.noise_info['fns']['E'].format(simidx)))
                        noise2 = load_file_wsec(opj(self.noise_info['libdir'], self.noise_info['fns']['B'].format(simidx)))
                    noise = np.array([noise1, noise2])
                    if self.noise_info['space'] == 'map':
                        if space == 'alm':
                            if self.noise_info['spin'] == 0:
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif self.noise_info['spin'] == 2:
                                noise = self.geom_lib.map2alm_spin(noise, spin=self.noise_info['spin'], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)  
                        elif space == 'map':
                            if self.noise_info['spin'] != spin:
                                if self.noise_info['spin'] == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(noise[0], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(noise[1], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                    noise = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.noise_info['lm_max'][0], spin=spin, mmax=self.noise_info['lm_max'][1], nthreads=4)
                                elif self.noise_info['spin'] == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(noise, spin=self.noise_info['spin'], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                    noise1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                    noise2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                    noise = np.array([noise1, noise2])
                    elif self.noise_info['space'] == 'alm':
                        if space == 'map':
                            if spin == 0:
                                noise1 = self.geom_lib.map2alm(noise[0], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                noise2 = self.geom_lib.map2alm(noise[1], lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                                noise = np.array([noise1, noise2])
                            elif spin == 2:
                                noise = self.geom_lib.alm2map_spin(noise, spin=spin, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)       
                elif field == 'temperature':
                    noise = np.array(load_file_wsec(opj(self.noise_info['libdir'], self.noise_info['fns']['T'].format(simidx))))
                    if self.noise_info['space'] == 'map':
                        if space == 'alm':
                            noise = self.geom_lib.map2alm(noise, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
                    elif self.noise_info['space'] == 'alm':
                        if space == 'map':
                            noise = self.geom_lib.alm2map(noise, lmax=self.noise_info['lm_max'][0], mmax=self.noise_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, noise)  
        return self.cacher.load(fn)


class Cls:
    """class for accessing CAMB-like file for CMB power spectra, optionally a distinct file for the lensing field (grad and curl component), and birefringence
    """    
    def __init__(self, fid_info=DNaV):
        # NOTE if CMB_fn is None, I assume the run does not have CMB spectra (this is the case when prialm CMB are provided, but secondaries alms are generated from Cls)
        # TODO add support for field-field (phi-bf) correlations
        if fid_info == DNaV or fid_info['fn'] == DNaV:
            if fid_info == DNaV:
                fid_info = {'fns': None, 'libdir': None}
            fid_info['fn'] = 'FFP10_wdipole_secondaries_lens_birefringence.dat'
            fid_info['libdir'] = opj(os.path.dirname(delensalot.__file__), 'data', 'cls')
            self.Cl_dict = load_file_wsec(opj(fid_info['libdir'], fid_info['fn']))
            # FIXME need to initialize the secondaries with the same CMB_fn
        elif fid_info['fn'] is None: # only need cl_dict for secondaries
            self.CMB_fn = None
            self.Cl_dict = {}
        else:
            self.CMB_fn = fid_info['fn']
            self.Cl_dict = load_file_wsec(opj(fid_info['libdir'], fid_info['fn']))
        self.fid_info = fid_info

        # NOTE now I either replace or delete secondaries from the dict.
        # If there are secondaries listed in the secondaries parameter, I will replace the Cl_dct values with it.
        # If not, I will delete the Cl_dict entries as I assume this run is performed without them.
        # If fn point to the same CMB_fn, I keep them
        secondaries_keep_list = [val for sublist in fid_info['sec_components'].values() for val in (sublist if isinstance(sublist, list) else [sublist])]
        secondaries_pop_list = [key for key in ['pp', 'pt', 'pe', 'ww', 'wt', 'we', 'wp', 'ff', 'ft', 'fe', 'fp', 'fw'] if key not in secondaries_keep_list]
        for key in secondaries_pop_list:
            del self.Cl_dict[key]
        if fid_info['fn'] != DNaV and fid_info['fn'] != fid_info['fn_sec']:
            sec_file = load_file_wsec(opj(fid_info['libdir'], fid_info['fn_sec']))
            for component in secondaries_keep_list:
                self.Cl_dict[component] = sec_file[component]
        self.secondaries = secondaries_keep_list

        self.cacher = cachers.cacher_mem(safe=True)


    def get_clCMBpri(self, simidx, components=['tt', 'ee', 'bb', 'te'], lmax=None):
        components = [components] if isinstance(components, str) else components
        fn = f"clcmb_{components}_{simidx}"
        if not self.cacher.is_cached(fn):
            Cls = np.array([self.Cl_dict[key][:lmax+1] for key in components]) if lmax is not None else np.array([self.Cl_dict[key] for key in components])
            self.cacher.cache(fn, Cls)
        return self.cacher.load(fn)   
    

    def get_clsec(self, simidx, secondary=None, components=None, lmax=None):
        # if isinstance(secondary, str):
        if secondary is None:
            return [self.get_clsec(simidx, sec, components, lmax) for sec in self.fid_info['sec_components'].keys()]
        if components is None:
            return np.array([self.get_clsec(simidx, secondary, comp, lmax).squeeze() for comp in self.fid_info['sec_components'][secondary]])
        components = [components] if isinstance(components, str) else components
        fn = f"clssec{secondary}_{components}_{simidx}"
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

        if CMB_info['libdir'] == DNaV or any(value['fns'] == DNaV for value in sec_info.values()):
            if cls_lib == DNaV:
                sec_info = {key: {'fns':DNaV, 'component':value['component'], 'libdir': DNaV, 'scale': 'p'} for key, value in sec_info.items()}
                self.cls_lib = Cls(CMB_info=DNaV, sec_info=sec_info) # NOTE I pick all CMB components anyway
            else:
                self.cls_lib = cls_lib
        else:
            if CMB_info['libdir'] != DNaV:
                for key, value in CMB_info.items():
                    if value == DNaV:
                        assert 0, f'need to provide {key}'
            else:
                for key in ['space', 'scale', 'modifier', 'lm_max', 'fns', 'component']:
                    if any(value['space'] == DNaV for value in sec_info.values()):
                        assert 0, 'need to provide {key} for all secondaries'

        self.cacher = cachers.cacher_mem(safe=True)


    def get_sim_pri(self, simidx, space, field, spin=2):
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
        fn = f"primordial_space{space}_spin{spin}_field{field}_{simidx}"
        if not self.cacher.is_cached(fn):
            if self.CMB_info['libdir'] == DNaV:
                Cls = self.cls_lib.get_clCMBpri(simidx, lmax=self.CMB_info['lm_max'][0])
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
                        pri1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['Q'].format(simidx)))
                        pri2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['U'].format(simidx)))
                    elif self.CMB_info['spin'] == 0:
                        pri1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['E'].format(simidx)))
                        pri2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['B'].format(simidx)))
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
                    pri = np.array(load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['T'].format(simidx))))
                    if self.CMB_info['space'] == 'map':
                        if space == 'alm':
                            pri = self.geom_lib.map2alm(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif self.CMB_info['space'] == 'alm':
                        if space == 'map':
                            pri = self.geom_lib.alm2map(pri, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, pri)
        return self.cacher.load(fn)
    

    def get_sim_sec(self, simidx, space, secondary=None, component=None):
        """ returns a secondary (phi, bf) in space (map, alm) and as component (grad, curl, if applicable). If secondary or component is None, it returns all, respectively.
        Args:
            simidx (_type_): _description_
            space (_type_): _description_

        Returns:
            _type_: _description_
        """
        if secondary is None:
            return [self.get_sim_sec(simidx, space, key, component=component) for key in self.sec_info.keys()]
        if secondary not in self.sec_info.keys():
            print(f"secondary {secondary} not available")
            return np.array([[None]])
        if isinstance(component, str) and component not in self.sec_info[secondary]['component']:
            print(f"component {component} of {secondary} not available")
            return np.array([[None]])
        if component is None:
            return np.array([self.get_sim_sec(simidx, space, secondary, component=comp) for comp in self.sec_info[secondary]['component']])
        if (isinstance(component, list) or isinstance(component, np.ndarray)) and len(component)>1:
            for comp in component:
                if comp not in self.sec_info[secondary]['component']:
                    print(f"component {comp} not available, removing from list")
                    component.remove(comp)
            return np.array([self.get_sim_sec(simidx, space, secondary, component=comp) for comp in component])
        

        fn = f"{secondary}{component}_space{space}_{simidx}"
        if not self.cacher.is_cached(fn):
            if self.sec_info[secondary]['libdir'] == DNaV:
                print(f'generating {secondary} {component} from cl')
                log.debug(f'generating {secondary}{component} from cl')
                Clpf = self.cls_lib.get_clsec(simidx, secondary, component*2).squeeze()
                self.sec_info[secondary]['scale'] = self.cls_lib.fid_info['scale']
                Clp = self.clsecsf2clsecp(secondary, Clpf)
                sec = self.clp2seclm(secondary, Clp, simidx)
                ## If it comes from CL, like Gauss secs, then sec modification must happen here
                sec = self.sec_info[secondary]['modifier'](sec)
                if space == 'map':
                    sec = self.geom_lib.alm2map(sec, lmax=self.sec_info[secondary]['lm_max'][0], mmax=self.sec_info[secondary]['lm_max'][1], nthreads=4)
            else:
                ## Existing sec is loaded, this e.g. is a kappa map on disk
                if self.sec_info[secondary]['space'] == 'map':
                    sec = np.array(load_file_wsec(opj(self.sec_info[secondary]['libdir'], self.sec_info[secondary]['fns'][component].format(simidx))), dtype=float)
                else:
                    sec = np.array(load_file_wsec(opj(self.sec_info[secondary]['libdir'], self.sec_info[secondary]['fns'][component].format(simidx))), dtype=complex)
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
        if self.sec_info[secondary]['scale'] == 'c':
            return klm2plm(seclm, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'd':
            return dlm2plm(seclm, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'p':
            return seclm


    def clsecsf2clsecp(self, secondary, cl):
        # NOTE naming convention is sec, but it can be grad or curl
        if self.sec_info[secondary]['scale'] == 'c':
            return clk2clp(cl, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'd':
            return cld2clp(cl, self.sec_info[secondary]['lm_max'][0])
        elif self.sec_info[secondary]['scale'] == 'p':
            return cl
        else:
            assert 0, f"scale not recognized: {self.sec_info[secondary]['scale']}"


    def cl2alm(self, cls, field, seed):
        np.random.seed(int(seed))
        alms = hp.synalm(cls, self.CMB_info['lm_max'][0], new=True)
        if field == 'polarization':
            return alms[1:]
        elif field == 'temperature':
            return alms[0]
    

    def clp2seclm(self, secondary, clp, seed):
        combined_str = f"{secondary}_{seed}".encode()
        hashed_seed = int(hashlib.sha256(combined_str).hexdigest(), 16) % (2**32)  # Convert to 32-bit int
        np.random.seed(hashed_seed)
        sec = hp.synalm(clp, self.sec_info[secondary]['lm_max'][0])
        return sec


class Xsky:
    """class for generating lensed CMB and phi realizations from priensed realizations, using lenspyx for the lensing operation
    """    
    def __init__(self, pri_lib=DNaV, geominfo=DNaV, CMB_info=DNaV, sec_info=DNaV, operator_info=DNaV):
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
            check_dict(CMB_info)
        self.CMB_info = CMB_info

        self.sec_info = sec_info
        self.operator_info = operator_info
        self.operators = [self.get_operator(key, op) for key, op in operator_info.items()]
        self.cacher = cachers.cacher_mem(safe=True)


    def get_operator(self, opk, opv):
        if opk == 'birefringence':
            return operator_secondary.birefringence(opv)
        elif opk == 'lensing':
            return operator_secondary.lensing(opv)


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
        fn = f"sky_space{space}_spin{spin}_field{field}_{simidx}"
        log.debug(f"requesting{fn}")
        self.lenjob_geomlib = self.operators[0].geomlib
        if not self.cacher.is_cached(fn):
            fn_other = f"sky_space{space}_spin{self.CMB_info['spin']}_field{field}_{simidx}"
            if not self.cacher.is_cached(fn_other):
                log.debug('..nothing cached..')
                if self.CMB_info['libdir'] == DNaV:
                    log.debug('.., generating.')
                    pri = self.pri_lib.get_sim_pri(simidx, space='alm', field=field, spin=0)
                    for operator in self.operators:
                        sec = self.pri_lib.get_sim_sec(0, space='alm', secondary=operator.ID)
                        if operator.ID == 'lensing': 
                            sec = np.array([alm_copy(s, None, operator.LM_max[0], operator.LM_max[1]) for s in sec])
                            h2d = np.sqrt(np.arange(operator.LM_max[0] + 1, dtype=float) * np.arange(1, operator.LM_max[0] + 2, dtype=float))
                            [almxfl(s, h2d, operator.LM_max[1], True) for s in sec]
                        operator.set_field(sec)
                        pri = operator.act(pri, spin=2 if field == 'polarization' else 0)
                    sky = pri
                    if field == 'polarization':
                        sky = self.operators[0].geomlib.alm2map_spin(sky, lmax=self.CMB_info['lm_max'][0], spin=2, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        if space == 'map':
                            if spin == 0:
                                alm_buffer = self.lenjob_geomlib.map2alm_spin(sky, spin=2, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky = np.array([sky1, sky2])
                            elif spin == 2:
                                sky = self.lenjob_geomlib.map2alm_spin(copy.copy(sky), spin=2, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                sky = self.geom_lib.alm2map_spin(copy.copy(sky), lmax=self.CMB_info['lm_max'][0], spin=2, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm_spin(sky, lmax=self.CMB_info['lm_max'][0], spin=2, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif field == 'temperature':
                        sky = self.operators[0].geomlib.alm2map(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        if space == 'map':
                            sky = self.lenjob_geomlib.map2alm(copy.copy(sky), lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                            sky = self.geom_lib.alm2map(copy.copy(sky), lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif space == 'alm':
                            sky = self.lenjob_geomlib.map2alm(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                else:
                    log.debug('.., but stored on disk.')
                    if field == 'polarization':
                        if self.CMB_info['spin'] == 2:
                            sky1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['Q'].format(simidx)))
                            sky2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['U'].format(simidx)))
                        elif self.CMB_info['spin'] == 0:
                            sky1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['E'].format(simidx)))
                            sky2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['B'].format(simidx)))
                        sky = np.array([sky1, sky2])
                        if self.CMB_info['space'] == 'map':
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
                        elif self.CMB_info['space'] == 'alm':
                            if space == 'map':
                                if spin == 0:
                                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    sky = np.array([sky1, sky2])
                                else:
                                    sky = self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif field == 'temperature':
                        sky = np.array(load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['T'].format(simidx))))
                        if self.CMB_info['space'] == 'map':
                            if space == 'alm':
                                sky = self.geom_lib.map2alm(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        elif self.CMB_info['space'] == 'alm':
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
        
        if CMB_info['libdir'] == DNaV:
            self.noise_lib = obs_info['noise_lib']
        else:
            self.noise_lib = None
        
        self.maps = maps
        if np.all(self.maps != DNaV):
            fn = f"obs_space{CMB_info['space']}_spin{CMB_info['spin']}_field{CMB_info['field']}_0"
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
                check_dict(CMB_info)

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
            assert self.CMB_info['spin'] == spin, "can only provide existing data"
            assert self.CMB_info['space'] == space, "can only provide existing data"
        fn = f'obs_space{space}_spin{spin}_field{field}_{simidx}'
        log.debug(f'requesting "{fn}"')
        fn_otherspin = f"obs_space{space}_spin{self.CMB_info['spin']}_field{field}_{simidx}"
        fn_otherspace = ''
        fn_otherspacespin = ''
        if self.CMB_info['space'] == 'alm':
            fn_otherspace = f"obs_spacealm_spin0_field{field}_{simidx}"
        elif self.CMB_info['space'] == 'map':
            fn_otherspace = f"obs_spacemap_spin{spin}_field{field}_{simidx}"
        if self.CMB_info['space'] == "alm":
            fn_otherspacespin = f"obs_spacealm_spin0_field{field}_{simidx}"
        elif self.CMB_info['space'] == 'map':
            fn_otherspacespin = f"obs_spacemap_spin{self.CMB_info['spin']}_field{field}_{simidx}"

        if not self.cacher.is_cached(fn) and not self.cacher.is_cached(fn_otherspin) and not self.cacher.is_cached(fn_otherspacespin) and not self.cacher.is_cached(fn_otherspace):
            log.debug('..nothing cached..')
            if self.CMB_info['libdir'] == DNaV: # sky data comes from sky_lib, and we add noise
                log.debug('.., generating.')
                obs = self.sky2obs(
                    np.copy(self.sky_lib.get_sim_sky(simidx, spin=spin, space=space, field=field)),
                    np.copy(self.noise_lib.get_sim_noise(simidx, spin=spin, field=field, space=space)),
                    spin=spin,
                    space=space,
                    field=field)

            elif self.CMB_info['libdir'] != DNaV:  # observed data is somewhere
                log.debug('.., but stored on disk.')
                if field == 'polarization':
                    if self.CMB_info['spin'] == 2:
                        if self.CMB_info['fns']['Q'] == self.CMB_info['fns']['U'] and self.CMB_info['fns']['Q'].endswith('.fits'):
                            # Assume implicitly that Q is field=1, U is field=2
                            obs1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['Q'].format(simidx)), ifield=1)
                            obs2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['U'].format(simidx)), ifield=2)
                        else:
                            obs1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['Q'].format(simidx)))
                            obs2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['U'].format(simidx)))
                    elif self.CMB_info['spin'] == 0:
                        if self.CMB_info['fns']['E'] == self.CMB_info['fns']['B'] and self.CMB_info['fns']['B'].endswith('.fits'):
                            # Assume implicitly that E is field=1, B is field=2
                            obs1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['E'].format(simidx)), ifield=1)
                            obs2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['B'].format(simidx)), ifield=2)
                        else:
                            obs1 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['E'].format(simidx)))
                            obs2 = load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['B'].format(simidx)))
                    obs1 = self.CMB_info['modifier'](obs1)
                    obs2 = self.CMB_info['modifier'](obs2)                
                    obs = np.array([obs1, obs2])
                    if self.CMB_info['space'] == 'map':
                        if space == 'map':
                            if self.CMB_info['spin'] != spin:
                                if self.CMB_info['spin'] == 0:
                                    alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    obs = self.geom_lib.alm2map_spin([alm_buffer1,alm_buffer2], lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                elif self.CMB_info['spin'] == 2:
                                    alm_buffer = self.geom_lib.map2alm_spin(obs, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    obs1 = self.geom_lib.alm2map(alm_buffer[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    obs2 = self.geom_lib.alm2map(alm_buffer[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                    obs = np.array([obs1, obs2])
                        elif space == 'alm':
                            if self.CMB_info['spin'] == 0:
                                obs1 = self.geom_lib.map2alm(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                obs2 = self.geom_lib.map2alm(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.map2alm_spin(obs, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif self.CMB_info['space'] == 'alm':
                        if space == 'map':
                            if spin == 0:
                                obs1 = self.geom_lib.alm2map(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                obs2 = self.geom_lib.alm2map(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                                obs = np.array([obs1, obs2])
                            else:
                                obs = self.geom_lib.alm2map_spin(obs, lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                elif field == 'temperature':
                    obs = np.array(load_file_wsec(opj(self.CMB_info['libdir'], self.CMB_info['fns']['T'].format(simidx))))
                    obs = self.CMB_info['modifier'](obs)
                    if self.CMB_info['space'] == 'map':
                        if space == 'alm':
                            obs = self.geom_lib.map2alm(obs, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    elif self.CMB_info['space'] == 'alm':
                        if space == 'map':
                            obs = self.geom_lib.alm2map(obs, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                self.cacher.cache(fn, obs)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn):
            log.debug('found "{}"'.format(fn))
            pass
        elif self.cacher.is_cached(fn_otherspin):
            log.debug('found "{}"'.format(fn_otherspin))
            obs = np.array(self.cacher.load(fn_otherspin))
            if space == 'map':
                if self.CMB_info['spin'] == 2:
                    obs1 = self.geom_lib.map2alm(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    obs2 = self.geom_lib.map2alm(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    obs = np.array([obs1, obs2])
                    obs = self.geom_lib.alm2map_spin(obs, lmax=self.CMB_info['lm_max'][0], spin=self.CMB_info['spin'], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                else:
                    obs = self.geom_lib.map2alm_spin(obs, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    obs1 = self.geom_lib.alm2map(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    obs2 = self.geom_lib.alm2map(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    obs = np.array([obs1, obs2])
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspace):
            log.debug('found "{}"'.format(fn_otherspace))
            obs = np.array(self.cacher.load(fn_otherspace))
            if field == 'polarization':
                if self.CMB_info['space'] == 'alm':
                    if spin == 0:
                        obs1 = self.geom_lib.alm2map(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        obs2 = self.geom_lib.alm2map(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        obs = np.array([obs1, obs2])
                    elif spin == 2:
                        obs = self.geom_lib.alm2map_spin(obs, lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
                elif self.CMB_info['space'] == 'map':
                    if self.CMB_info['spin'] == 0:
                        alm_buffer1 = self.geom_lib.map2alm(obs[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        alm_buffer2 = self.geom_lib.map2alm(obs[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                        obs = np.array([alm_buffer1, alm_buffer2])
                    elif self.CMB_info['spin'] == 2:
                        obs = self.geom_lib.map2alm_spin(obs, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            elif field == 'temperature':
                if self.CMB_info['space'] == 'alm': 
                    obs = self.geom_lib.alm2map(obs, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                elif self.CMB_info['space'] == 'map':
                    obs = self.geom_lib.map2alm(obs, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, obs)
        elif self.cacher.is_cached(fn_otherspacespin):
            log.debug('found "{}"'.format(fn_otherspacespin))
            obs = np.array(self.cacher.load(fn_otherspacespin))
            if self.CMB_info['space'] == 'alm':
                obs = self.geom_lib.alm2map_spin(obs, lmax=self.CMB_info['lm_max'][0], spin=spin, mmax=self.CMB_info['lm_max'][1], nthreads=4)
            elif self.CMB_info['space'] == 'map':
                obs = self.geom_lib.map2alm_spin(obs, spin=self.CMB_info['spin'], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            self.cacher.cache(fn, obs)
        return self.cacher.load(fn)
    

    def sky2obs(self, sky, noise, spin, space, field):
        if field == 'polarization':
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.map2alm(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    sky2 = self.geom_lib.map2alm(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = self.geom_lib.map2alm_spin(sky, spin=spin, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            hp.almxfl(sky[0], self.obs_info['transfunction'], inplace=True)
            hp.almxfl(sky[1], self.obs_info['transfunction'], inplace=True)
            if space == 'map':
                if spin == 0:
                    sky1 = self.geom_lib.alm2map(sky[0], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    sky2 = self.geom_lib.alm2map(sky[1], lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
                    sky = np.array([sky1, sky2])
                elif spin == 2:
                    sky = np.array(self.geom_lib.alm2map_spin(sky, spin=spin, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4))
                return sky + noise
            else:
                return sky + noise
        elif field == 'temperature':
            if space == 'map':
                sky = self.geom_lib.map2alm(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)
            hp.almxfl(sky, self.obs_info['transfunction'], inplace=True)
            if space == 'map':
                return np.array(self.geom_lib.alm2map(sky, lmax=self.CMB_info['lm_max'][0], mmax=self.CMB_info['lm_max'][1], nthreads=4)) + noise
            else:
                return sky + noise


    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
  

class Simhandler:
    """Entry point for data handling and generating simulations.
    Data can be cl, pri, len, or obs, .. and alms or maps. Simhandler connects the individual libraries and decides what can be generated.
    E.g.: If obs data provided, len data cannot be generated.
    """ 
    def __init__(self, flavour, maps=DNaV, geominfo=DNaV, fid_info=DNaV, CMB_info=DNaV, sec_info=DNaV, obs_info=DNaV, operator_info=DNaV):
        """Entry point for simulation data handling.
        Simhandler() connects the individual librariers together accordingly, depending on the provided data.
        It never stores data on disk itself, only in memory.

        Args:
            flavour (str): 'obs', 'sky', 'pri'
            maps (np.ndarray): CMB data in map space
            geominfo (tuple): geometry information
            CMB_info (dict): CMB information
            sec_info (dict): secondary CMB information
            obs_info (dict): observation information
            operator_info (dict): operator_info for secondaries
        """
        if CMB_info['space'] == 'alm':
            assert CMB_info['spin'] == 0, "spin has to be 0 for alm space"

        if flavour == 'obs':
            assert CMB_info['space'] in ['map','alm'], "obs CMB data can only be in map or alm space"
            if np.all(maps == DNaV):
                check_dict(CMB_info)
            else:
                for key in ['spin', 'lm_max', 'field']:
                    if CMB_info[key] == DNaV:
                        assert 0, 'need to provide {}'.format(key)
            self.cls_lib = Cls(fid_info=fid_info) # NOTE I need cls_lib always, because I need to know the fiducial cls for chh, i.e. N0 for the curvature update
            self.obs_lib = Xobs(maps=maps, geominfo=geominfo, CMB_info=CMB_info)
        else:
            if flavour == 'sky':
                assert CMB_info['space'] in ['map','alm'], "sky CMB data can only be in map or alm space"
                assert not (contains_DNaV(obs_info)), "need to provide complete obs_info"
                self.cls_lib = Cls(fid_info=fid_info) # NOTE I need cls_lib always, because I need to know the fiducial cls for chh, i.e. N0 for the curvature update
                self.sky_lib = Xsky(pri_lib=DNaV, geominfo=geominfo, CMB_info=copy.copy(CMB_info), sec_info=sec_info, operator_info=operator_info)
                
                self.noise_lib = self.obs_lib.noise_lib
                self.libdir = self.sky_lib.CMB_info['libdir']
                self.fns = self.sky_lib.CMB_info['fns']
            elif flavour == 'pri':
                assert not (contains_DNaV(sec_info, ignore_keys={'libdir', 'fns'})), f"need to provide complete sec_info, {sec_info}"
                assert not (contains_DNaV(obs_info, ignore_keys={'libdir', 'fns'})), f"need to provide complete obs_info, {obs_info}"
                # TODO if space is cl need to initalize cls_lib, otherwise not needed
                self.cls_lib = Cls(fid_info=fid_info)
                CMB_info.update({'libdir':DNaV, 'fns':DNaV})
                self.pri_lib = Xpri(cls_lib=self.cls_lib, geominfo=geominfo, CMB_info=copy.copy(CMB_info), sec_info=sec_info)
                self.sky_lib = Xsky(pri_lib=self.pri_lib, geominfo=geominfo, CMB_info=copy.copy(CMB_info), sec_info=sec_info, operator_info=operator_info)

            if obs_info['noise_info'].get('libdir') == DNaV:
                # FIXME this looks wrong
                noise_info = obs_info['noise_info']
                noise_lib = iso_white_noise(geominfo=geominfo, noise_info=noise_info)
                obs_info.update({'noise_lib':noise_lib})
                self.noise_lib = noise_lib
            self.obs_lib = Xobs(maps=DNaV, sky_lib=self.sky_lib, geominfo=geominfo, CMB_info=copy.copy(CMB_info), obs_info=obs_info)
        
        self.fid_info = self.cls_lib.fid_info
        self.obs_info = obs_info
        self.CMB_info = self.obs_lib.CMB_info
        self.sec_info = sec_info
        self.operator_info = operator_info

        self.flavour = flavour
        self.maps = maps
        self.geominfo = self.obs_lib.geominfo # Sim_generator() needs this. I let obs_lib decide the final geominfo.
        self.spin = self.CMB_info['spin']
        self.lm_max = self.CMB_info['lm_max']
        self.space = self.CMB_info['space']
        self.libdir = self.CMB_info['libdir']
        self.fns = self.CMB_info['fns']

        self.transfunction = self.obs_info['transfunction']
        self.noise_lib = self.obs_lib.noise_lib
        if self.noise_lib is not None:
            self.nlev = noise_lib.noise_info['nlev']
            self.libdir_suffix = noise_lib.noise_info['libdir_suffix']


    def get_sim_sky(self, simidx, space, field, spin):
        return self.sky_lib.get_sim_sky(simidx=simidx, space=space, field=field, spin=spin)

    def get_sim_pri(self, simidx, space, field, spin):
        return self.pri_lib.get_sim_pri(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_obs(self, simidx, space, field, spin):
        return self.obs_lib.get_sim_obs(simidx=simidx, space=space, field=field, spin=spin)
    
    def get_sim_noise(self, simidx, space, field, spin=2):
        return self.noise_lib.get_sim_noise(simidx, spin=spin, space=space, field=field)
    
    def get_sim_sec(self, simidx, space, secondary=None, component=None):
        return self.pri_lib.get_sim_sec(simidx=simidx, space=space, secondary=secondary, component=component)
    
    def get_sim_fidCMB(self, simidx, components):
        return self.cls_lib.get_clCMBpri(simidx=simidx, components=components)

    def get_sim_fidsec(self, simidx, secondary=None, components=None):
        return self.cls_lib.get_clsec(simidx=simidx, secondary=secondary, components=components)
    
    
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
                if os.path.exists(opj(self.libdir, self.CMB_info['fns']['Q'].format(simidx))) and os.path.exists(opj(self.libdir, self.CMB_info['fns']['U'].format(simidx))):
                    return True
        if field == 'temperature':
            if self.libdir != DNaV and self.fns != DNaV:
                if os.path.exists(opj(self.libdir, self.CMB_info['fns']['T'].format(simidx))):
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