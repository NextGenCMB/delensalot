'''Script for calculating delensed ILC and Blens spectra, using precaulculated Btemplates as input. Use 'Generate_Btemplate.py' for calulcating Btemplate input.
''' 

from __future__ import print_function
import os, sys
from os.path import join as opj
import hashlib
import argparse

import numpy as np
import healpy as hp
from astropy.io import fits

import plancklens
from plancklens.helpers import mpi
from plancklens import utils
from plancklens.sims import planck2018_sims

from lerepi.config.cmbs4.data import data_08d as sims_if
from component_separation.MSC.MSC import pospace as ps

mpi.barrier = lambda : 1 # redefining the barrier

ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])

parser = argparse.ArgumentParser(description='Delensing maps using Btemplates')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=100, help='Maximal simulation index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
parser.add_argument('-edges', dest='edges', type=str, default='cmbs4', help='Edges identifier. See file for edges definitions.')
args = parser.parse_args()


def getfn_blm_lensc(ana_p, simidx, fg, it):
    '''Lenscarf output using Catherinas E and B maps'''
    rootstr = '/global/cscratch1/sd/sebibel/cmbs4/'
    return rootstr+ana_p+'p_p_sim%04d/wflms/btempl_p%03d_e%03d_lmax1024.npy'%(simidx, it, it)
                
def getfn_qumap_cs(simidx):
    '''Component separated polarisation maps lm, i.e. lenscarf input'''
    return sims_may.get_sim_pmap(simidx)


if args.edges == 'bk14':
    edges = bk14_edges
elif args.edges == 'ioreco':
    edges = ioreco_edges
elif args.edges == 'cmbs4':
    edges = cmbs4_edges

'''
#################################################################
##################   Most important settings   ##################
#################################################################
'''
simid_lower = args.simid_lower
simid_upper = args.simid_upper
simids = np.arange(simid_lower, simid_upper+1)
fg = args.fg
# CMB_S4 mask only needed for rotating ILC maps
cmbs4_mask = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08d.%s_umilta_220502'%fg + '/ILC_mask_08d_smooth_30arcmin.fits'))
Nmf = 100
analysis_path = '08d_%s_rNone_MF%d_OBD200/'%(fg,Nmf)
nlevels = [2., 5., 10., 100.]
itmax = 4
'''
#################################################################
##################  /Most important settings   ##################
#################################################################
'''

edges_center = (edges[1:]+edges[:-1])/2.
nside = 2048
lmax_cl = 2048
lmax_lib = 3*lmax_cl-1

beam = 2.3
lmax_transf = 4000
transf = hp.gauss_beam(beam / 180. / 60. * np.pi, lmax=lmax_transf)

cls_path = opj(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_len = utils.camb_clfile(opj(cls_path, 'FFP10_wdipole_lensedCls.dat'))
clg_templ = cls_len['ee']
clc_templ = cls_len['bb']

clg_templ[0] = 1e-32
clg_templ[1] = 1e-32

sha_edges = hashlib.sha256()
sha_edges.update(str(edges).encode())

dirid = sha_edges.hexdigest()[:4]


if __name__ == '__main__':
    lib_nm = dict()
    bcl_L_nm, bcl_cs_nm, bwfcl_cs_nm = np.zeros(shape=(len(nlevels),len(edges))), np.zeros(shape=(len(nlevels),len(edges))), np.zeros(shape=(len(nlevels),len(edges)))
    outputdata = np.zeros(shape=(6,len(nlevels),len(edges)-1))

    for nlev in nlevels:
        sims_may  = sims_if.ILC_May2022(fg, mask_suffix=int(nlev))
        nlev_mask = sims_may.get_mask() 
        lib_nm.update({nlev: ps.map2cl_binned(nlev_mask, clc_templ[:lmax_lib], edges, lmax_lib)})
        
    for simid in simids[mpi.rank::mpi.size]:
        dirroot = '/global/cscratch1/sd/sebibel/cmbs4/'+analysis_path+'plotdata/'
        if not(os.path.isdir(dirroot + '{}'.format(dirid))):
            os.makedirs(dirroot + '{}'.format(dirid))
        file_op = dirroot + '{}'.format(dirid) + '/ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg)
        print('will store file at:', file_op)
        
        qumap_cs_buff = getfn_qumap_cs(simid)
        eblm_cs_buff = hp.map2alm_spin(qumap_cs_buff*cmbs4_mask, 2, lmax_cl)
        bmap_cs_buff = hp.alm2map(eblm_cs_buff[1], nside)
        for nlevi, nlev in enumerate(nlevels):
            sims_may  = sims_if.ILC_May2022(fg, mask_suffix=int(nlev))
            nlev_mask = sims_may.get_mask() 
            bcl_cs_nm = lib_nm[nlev].map2cl(bmap_cs_buff)
            blm_L_buff = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(simid), lmax=lmax_cl), transf)
            bmap_L_buff = hp.alm2map(blm_L_buff, nside)
            bcl_L_nm = lib_nm[nlev].map2cl(bmap_L_buff)

            blm_lensc_MAP_buff = np.load(getfn_blm_lensc(analysis_path, simid, fg, itmax))
            bmap_lensc_MAP_buff = hp.alm2map(blm_lensc_MAP_buff, nside=nside)
            blm_lensc_QE_buff = np.load(getfn_blm_lensc(analysis_path, simid, fg, 0))
            bmap_lensc_QE_buff = hp.alm2map(blm_lensc_QE_buff, nside=nside)
 
            bcl_Llensc_MAP_nm = lib_nm[nlev].map2cl(bmap_L_buff-bmap_lensc_MAP_buff)    
            bcl_Llensc_QE_nm = lib_nm[nlev].map2cl(bmap_L_buff-bmap_lensc_QE_buff)

            bcl_cslensc_MAP_nm = lib_nm[nlev].map2cl(bmap_cs_buff-bmap_lensc_MAP_buff)
            bcl_cslensc_QE_nm = lib_nm[nlev].map2cl(bmap_cs_buff-bmap_lensc_QE_buff)
            
            outputdata[0][nlevi] = bcl_L_nm
            outputdata[1][nlevi] = bcl_cs_nm
            
            outputdata[2][nlevi] = bcl_Llensc_MAP_nm
            outputdata[3][nlevi] = bcl_cslensc_MAP_nm  
            
            outputdata[4][nlevi] = bcl_Llensc_QE_nm           
            outputdata[5][nlevi] = bcl_cslensc_QE_nm
            print('mpi.rank {}: nlev {} ({}/{}) done. Sim {}/{} done.'.format(mpi.rank, str(nlev),nlevi+1,len(nlevels), simid+1, len(simids)))
        np.save(file_op, outputdata)

