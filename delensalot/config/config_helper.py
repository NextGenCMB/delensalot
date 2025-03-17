#!/usr/bin/env python

"""config_helper.py: functions and constants which come in handy for configuration files.
"""
import re, sys
import numpy as np
import healpy as hp
import copy
from itertools import product
import importlib.util as iu

from delensalot.config.etc.errorhandler import DelensalotError

PLANCKLENS_keys_fund = ['ptt', 'xtt', 'p_p', 'x_p', 'p', 'x', 'stt', 's', 'ftt','f_p', 'f','dtt', 'ntt','n', 'a_p',
                    'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                    'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']
PLANCKLENS_keys = PLANCKLENS_keys_fund + ['p_tp', 'x_tp', 'p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb', 'ptt_bh_n',
                                'ptt_bh_s', 'ptt_bh_f', 'ptt_bh_d', 'dtt_bh_p', 'stt_bh_p', 'ftt_bh_d',
                                'p_bh_s', 'p_bh_n']

class LEREPI_Constants:
    fs_edges = np.arange(2, 3000, 20)
    ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
    lowell_edges = np.array([2, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
    cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
    SPDP_edges = np.concatenate([np.arange(2,200,50),np.logspace(np.log(2e2),np.log(4000),40, base=np.e, dtype=int)]) # these are used for calculating residual power spectra on SPDP patch.
    SPDP2_edges = np.arange(2,4000,30) # these are used for calculating residual power spectra on SPDP patch.
    AoA_edges = np.array([2, 30, 70, 110, 140, 170, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])

class data_functions:

    def a2r(val):
        """arcmin2radian converter
        """
        return val / 180 / 60 * np.pi

    def r2a(val):
        """radian2arcmin converter
        """
        return val * 180 * 60 / np.pi
        

    def c2a(val):
        """Cl2arcmin converter
        """
        return np.sqrt(val) * (60 * 180 / np.pi)
        

    def get_nlev_mask(ratio, rhits):
        """Mask built thresholding the relative hit counts map
        """
        mask = np.where(rhits/np.max(rhits) < 1. / ratio, 0., rhits)
        return np.nan_to_num(mask)


    def get_zbounds(rhits, hits_ratio=np.inf):
        """Cos-tht bounds for thresholded mask
        """
        pix = np.where(data_functions.get_nlev_mask(hits_ratio, rhits))[0]
        tht, phi = hp.pix2ang(2048, pix)
        zbounds = np.cos(np.max(tht)), np.cos(np.min(tht))
        return zbounds


    def extend_zbounds(zbounds, degrees=5.):
        zbounds_len = [np.cos(np.arccos(zbounds[0]) + degrees / 180 * np.pi), np.cos(np.arccos(zbounds[1]) - degrees / 180 * np.pi)]
        zbounds_len[0] = max(zbounds_len[0], -1.)
        zbounds_len[1] = min(zbounds_len[1],  1.)

        return zbounds_len
    

def generate_plancklenskeys(input_str):
    def split_at_first(s, blacklist={'t', 'e', 'b'}):
        match = re.search(f"[{''.join(blacklist)}]", s)
        if match:
            return s[:match.start()], s[match.start():]
        return s, ''
    lensing_components = {'p', 'w'}
    birefringence_components = {'f'}
    valid_suffixes = {'p', 'ee', 'eb'}
    transtable = str.maketrans({'p':"p", 'f':"a", 'w':"x"})
    if "_" in input_str:
        components_part, suffix = input_str.split('_')
    else:
        components_part, suffix = split_at_first(input_str)  # last character as suffix
    lensing = sorted(components_part[i] for i in range(len(components_part)) if components_part[i] in lensing_components)
    birefringence = sorted(components_part[i] for i in range(len(components_part)) if components_part[i] in birefringence_components)
    secondary_key = {}
    if lensing:
        secondary_key['lensing'] = {comp: comp.translate(transtable) + "_" + suffix if "_" in input_str else comp.translate(transtable)+ suffix for comp in lensing}
    if birefringence:
        secondary_key['birefringence'] = {comp: comp.translate(transtable) + "_" + suffix if "_" in input_str else comp.translate(transtable) + suffix for comp in birefringence}

    for sec, comp in secondary_key.items():
        for co, c in comp.items():
            if c.endswith('tp'):
                secondary_key[sec][co] = secondary_key[sec][co].replace('_tp', '')
                if secondary_key[sec][co] == 'a':
                    secondary_key[sec][co] = 'a_p'
            if c.endswith('tt') or c.endswith('eb'):
                if secondary_key[sec][co] == 'att':
                    print(f"Turning {c} into a_p so that we can generate a valid starting point. This will fail if no polarization data provided")
                    secondary_key[sec][co] = 'a_p'
                if secondary_key[sec][co] == 'a_eb':
                    print(f"Turning {c} into a_p so that we can generate a valid starting point. This will fail if no polarization data provided")
                    secondary_key[sec][co] = 'a_p'
    for sec, val in secondary_key.items():
        for comp in val.values():
            if comp not in PLANCKLENS_keys:
                raise DelensalotError(f"Your input '{input_str}' is not a valid key, it generated '{comp}' which is not a valid Plancklens key.")
    print(f'the generated secondary keys for Plancklens are {input_str} - > {secondary_key}')
    return secondary_key


def load_config(directory, descriptor):
    """Helper method for loading the configuration file.

    Args:
        directory (_type_): The directory to read from.
        descriptor (_type_): Identifier with which the configuration file is stored in memory.

    Returns:
        object: the configuration file
    """        
    spec = iu.spec_from_file_location('configfile', directory)
    p = iu.module_from_spec(spec)
    sys.modules[descriptor] = p
    spec.loader.exec_module(p)

    return p.delensalot_model


def set_nested_attr(obj, attr_path, value):
    """Sets a nested attribute based on a dot-separated path."""
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def get_nested_attr(obj, attr_path):
    """Gets a nested attribute based on a dot-separated path."""
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def scan_config(config, param_dict):
    """
    Iterates over all combinations of parameters in param_dict.

    :param config: The configuration object.
    :param param_dict: Dictionary with attribute paths as keys and lists of values.
    """
    param_keys = list(param_dict.keys())  # Get the attribute paths
    param_values = list(param_dict.values())  # Get lists of values

    # Store original values to restore later
    original_values = {key: get_nested_attr(config, key) for key in param_keys}

    # Iterate over all combinations of parameter values
    for combination in product(*param_values):
        for key, value in zip(param_keys, combination):
            set_nested_attr(config, key, value)
        config.fill_with_defaults()
        yield config  # Yield the modified configuration

    # Restore original values after iteration
    for key, original_value in original_values.items():
        set_nested_attr(config, key, original_value)


def dprint(dictionary, depth=0, max_depth=2):
    if isinstance(dictionary, dict):
        max_key_length = max(len(str(key)) for key in dictionary)  # Get longest key length
        for key, value in dictionary.items():
            # Add a tab for each depth level
            indent = '\t' * depth
            # If the value is a dictionary and the current depth is less than max_depth, recurse
            if isinstance(value, dict) and depth < max_depth:
                print(f"{indent}{key.ljust(max_key_length)} : ")
                dprint(value, depth + 1, max_depth)  # Recursively print the nested dictionary
            else:
                print(f"{indent}{key.ljust(max_key_length)} : {value}")
    else:
        print(dictionary)


def filter_secondary_and_component(dct, allowed_chars):
    dct = copy.deepcopy(dct)
    forbidden_chars_in_sec = 'teb'  # NOTE this filters the last part in case of non-symmetrized keys (e.g. 'pee')
    allowed_set = set("".join(c for c in allowed_chars if c not in forbidden_chars_in_sec))
    sec_to_remove = []
    for key, value in dct.items():
        if 'component' in value:
            value['component'] = [char for char in value['component'] if char in allowed_set]
            if not value['component']:
                sec_to_remove.append(key)
    for key in sec_to_remove:
        del dct[key]
    return dct

class CMBExperiment:
    def get_config(exp):
        """Returns noise levels, beam size and multipole cuts for some configurations

        """
        sN_uKaminP = None
        if exp == 'Planck':
            sN_uKamin = 35.
            Beam_FWHM_amin = 7.
            ellmin = 10
            ellmax = 2048
        elif exp == 'S4':
            sN_uKamin = 1.5
            Beam_FWHM_amin = 3.
            ellmin = 10
            ellmax = 3000
        elif exp == 'S4_opti':
            sN_uKamin = 1.
            Beam_FWHM_amin = 1.
            ellmin = 10
            ellmax = 3000
        elif exp == 'SO_opti':
            sN_uKamin = 11.
            Beam_FWHM_amin = 4.
            ellmin = 10
            ellmax = 3000
        elif exp == 'SO':
            sN_uKamin = 3.
            Beam_FWHM_amin = 3.
            ellmin = 10
            ellmax = 3000
        else:
            sN_uKamin = 0
            Beam_FWHM_amin = 0
            ellmin = 0
            ellmax = 0
            assert 0, '%s not implemented' % exp
        sN_uKaminP = sN_uKaminP or np.sqrt(2.) * sN_uKamin
        return sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax



    def cmbs4_06b():
        zbounds_len = [-0.9736165659024625, -0.4721687661208586]
        pbounds_exl = np.array((113.20399439681668, 326.79600560318335)) #These were the pbounds as defined with the old itercurv conv.
        pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
        pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
        scarf_job = us.scarfjob()
        scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
        return scarf_job, [pb_ctr / 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len, zbounds_len

    def cmbs4_08b_healpix():
        zbounds_len = [-0.9736165659024625, -0.4721687661208586]
        pbounds_exl = np.array((113.20399439681668, 326.79600560318335))
        pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
        pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
        scarf_job = us.scarfjob()
        scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
        return scarf_job, [pb_ctr/ 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len, zbounds_len

    def cmbs4_08b_healpix_oneq():
        zbounds_len = np.cos( (90 + 25) / 180 * np.pi), np.cos( (90 - 25) / 180 * np.pi)
        pb_ctr = 0. / 180 * np.pi
        pb_extent = 50. / 180 * np.pi
        scarf_job = us.scarfjob()
        scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
        return scarf_job, [pb_ctr, pb_extent], zbounds_len, zbounds_len

    def cmbs4_08b_healpix_onp(): # square of 50 deg by 50 deg
        extent_deg = 40. # 40 deg ensures each and every point at least 6 deg. away from mask
        zbounds_len = (np.cos(extent_deg / 180 * np.pi), 1.)
        zbounds_ninv =  (np.cos(34. / 180 * np.pi), 1.)
        pb_ctr = 0. / 180 * np.pi
        pb_extent = 360 / 180 * np.pi
        scarf_job = us.scarfjob()
        scarf_job.set_healpix_geometry(2048, zbounds=zbounds_ninv)
        return scarf_job, [pb_ctr, pb_extent], zbounds_len, zbounds_ninv

    def full_sky_healpix():
        scarf_job = us.scarfjob()
        scarf_job.set_healpix_geometry(2048)
        return scarf_job, [0, 2 * np.pi], (-1.,1.), (-1.,1)