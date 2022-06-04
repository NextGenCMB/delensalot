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

from cmbs4 import sims_08b

from component_separation.MSC.MSC import pospace as ps

mpi.barrier = lambda : 1 # redefining the barrier

bk14_edges = np.array([2,55,90,125,160,195,230,265,300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]) # BK14 is [ 55  90 125 160 195 230 265 300], from bk14 = h5py.File('/global/homes/s/sebibel/notebooks/CMBS4/datasharing/likedata_BK14.mat', 'r')['likedata']['lval'][0,:]
ioreco_edges = np.array([2, 30, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
cmbs4_edges = np.array([2, 30, 60, 90, 120, 150, 180, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])

parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
parser.add_argument('-sidl', dest='simid_lower', type=int, default=0, help='Minimal simulation index')
parser.add_argument('-sidu', dest='simid_upper', type=int, default=500, help='Maximal simulation index')
parser.add_argument('-fg', dest='fg', type=str, default='00', help='Foreground model. Either 00, 07, or 09')
parser.add_argument('-bs', dest='blm_suffix', type=str, default='', help='Suffix string. Defines where the plms are found and where the blm will be stored.')

parser.add_argument('-edges', dest='edges', type=str, default='bk14', help='Edges identifier. See file for edges definitions.')

args = parser.parse_args()

'''
Most important settings
'''
simid_lower = args.simid_lower
simid_upper = args.simid_upper
fg = args.fg
blm_suffix = args.blm_suffix # Could be '_mfsubtraced'
    
if args.edges == 'bk14':
    edges = bk14_edges
elif args.edges == 'ioreco':
    edges = ioreco_edges
elif args.edges == 'cmbs4':
    edges = cmbs4_edges
    
simids = np.arange(simid_lower, simid_upper)#np.array([0,500])#
'''
/Most important settings
'''

edges_center = (edges[1:]+edges[:-1])/2.
sims_cmbs4May  = sims_08b.caterinaILC_May12(fg)
sims_cmbs4Sep  = sims_08b.caterinaILC_Sep12(fg)

nside = 2048
lmax_cl = 2048
lmax_lib = 3*lmax_cl-1

# CMB_S4 mask only needed for rotating ILC maps
cmbs4_mask = np.nan_to_num(hp.read_map('/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/foreground_cleaned_maps/08b.%s_umilta_210511/'%fg + '/ILC_mask_08b_smooth_30arcmin.fits'))

def getfn_blm_lensc(simidx, fg, it):
    '''Lenscarf output using Catherinas E and B maps'''
    if blm_suffix == '':
        if it==12:
            rootstr = '/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/lt_recons/'
            if fg == '00':
                return rootstr+'08b.%02d_sebibel_210708_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(fg), it, simidx)
            elif fg == '07':
                return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(fg), it, simidx)
            elif fg == '09':
                return rootstr+'/08b.%02d_sebibel_210910_ilc_iter/blm_csMAP_obd_scond_lmaxcmb4000_iter_%03d_elm011_sim_%04d.fits'%(int(fg), it, simidx)
        elif it==0:
            return '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/ffi_p_it0/blm_%04d_it0.npy'%(fg, simidx, simidx)   
    else:
        rootstr = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%s_lmax4000/zb_terator_p_p_%04d_nofg_OBD_solcond_3apr20/'%(fg, simidx)
        return rootstr + 'ffi_p_it%s%s/blm%s_%04d_it%d.npy'%(it, blm_suffix, blm_suffix, simidx, it)

def getfn_qumap_cs(simidx):
    '''Component separated polarisation maps lm, i.e. lenscarf input'''
    if simidx>=200:
        return sims_cmbs4Sep.get_sim_pmap(simidx)
    else:
        return sims_cmbs4May.get_sim_pmap(simidx)

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
sha_edges.update((str(edges)+str(blm_suffix)).encode())

nlevels = [1.2, 1.5, 1.7, 2., 2.3, 5., 10., 50.]
dirid = sha_edges.hexdigest()[:4]
dirroot = '/global/cscratch1/sd/sebibel/cmbs4/s08b/cILC2021_%2s_lmax4000_plotdata/'%(fg)

if not(os.path.isdir(dirroot)):
    os.mkdir(dirroot)
if not(os.path.isdir(dirroot + '{}'.format(dirid))):
    os.mkdir(dirroot + '{}'.format(dirid))

if __name__ == '__main__':
    """
    This calculates for each noiselevel and each fg sim-ds, QE and MAP delensed blensing and bpower spectra.
    TODO: add bwf delensed
    """
    lib_nm = dict()
    bcl_L_nm, bcl_cs_nm, bwfcl_cs_nm = np.zeros(shape=(len(nlevels),len(edges))), np.zeros(shape=(len(nlevels),len(edges))), np.zeros(shape=(len(nlevels),len(edges)))
    outputdata = np.zeros(shape=(6,len(nlevels),len(edges)-1))

    for nlev in nlevels:
        nlev_mask = sims_08b.get_nlev_mask(nlev)
        lib_nm.update({nlev: ps.map2cl_binned(nlev_mask, clc_templ[:lmax_lib], edges, lmax_lib)})
        
    for simid in simids[mpi.rank::mpi.size]:
        qumap_cs_buff = getfn_qumap_cs(simid)
        eblm_cs_buff = hp.map2alm_spin(qumap_cs_buff*cmbs4_mask, 2, lmax_cl)
        bmap_cs_buff = hp.alm2map(eblm_cs_buff[1], nside)
        for nlevi, nlev in enumerate(nlevels):
            nlev_mask = sims_08b.get_nlev_mask(nlev)     
            bcl_cs_nm = lib_nm[nlev].map2cl(bmap_cs_buff)
            blm_L_buff = hp.almxfl(utils.alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(simid), lmax=lmax_cl), transf)
            bmap_L_buff = hp.alm2map(blm_L_buff, nside)
            bcl_L_nm = lib_nm[nlev].map2cl(bmap_L_buff)
            if blm_suffix == '':
                blm_lensc_MAP_buff = hp.read_alm(getfn_blm_lensc(simid, fg, 12))
                bmap_lensc_MAP_buff = hp.alm2map(blm_lensc_MAP_buff, nside=nside)
                blm_lensc_QE_buff = np.load(getfn_blm_lensc(simid, fg, 0))
                bmap_lensc_QE_buff = hp.alm2map(blm_lensc_QE_buff, nside=nside)
            else:
                blm_lensc_MAP_buff = np.load(getfn_blm_lensc(simid, fg, 12))
                bmap_lensc_MAP_buff = hp.alm2map(blm_lensc_MAP_buff, nside=nside)
                blm_lensc_QE_buff = np.load(getfn_blm_lensc(simid, fg, 0))
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

        np.save(dirroot + '{}'.format(dirid) + '/Lenscarf_plotdata_ClBBwf_sim%04d_fg%2s_res2b3acm.npy'%(simid, fg), outputdata)
        print('mpi.rank {} nlev {} mask: {} / {} done.'.format(mpi.rank, str(nlev), simid+1, len(simids)))