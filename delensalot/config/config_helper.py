#!/usr/bin/env python

"""config_helper.py: functions and constants which come in handy for configuration files.
"""
import re, sys
import numpy as np
import healpy as hp

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

        Args:
            val (_type_): _description_

        Returns:
            _type_: _description_
        """
        return val / 180 / 60 * np.pi

    def r2a(val):
        """radian2arcmin converter

        Args:
            val (_type_): _description_

        Returns:
            _type_: _description_
        """
        return val * 180 * 60 / np.pi
        


    def c2a(val):
        """Cl2arcmin converter

        Args:
            val (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.sqrt(val) * (60 * 180 / np.pi)
        

    def get_nlev_mask(ratio, rhits):
        """Mask built thresholding the relative hit counts map
            Note:
                Same as 06b
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
