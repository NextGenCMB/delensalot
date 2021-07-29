"""Iterative reconstruction for idealized pol data


"""


from lenscarf.iterators import cs_iterator as scarf_iterator, steps

from plancklens.qcinv import cd_solve
import itercurv


from itercurv.remapping.utils import alm_copy
from lenscarf import remapping

import numpy as np
import os
from plancklens import utils, qresp
from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam, almxfl, synalm, alm_copy
from lenscarf.opfilt.opfilt_iso_pp import alm_filter_nlev
from lenscarf.opfilt.opfilt_iso_ee_wl import alm_filter_nlev_wl

from lenscarf import utils_scarf

TEMP =  '/global/cscratch1/sd/jcarron/cmbs4/temp/s4P_fs'

lmin, lmax, mmax, beam, nlev_t, nlev_p = (30, 3000, 3000, 1., 1.5, 1.5 * np.sqrt(2.))
lmax_qlm, mmax_qlm, lmax_unl, mmax_unl = (4000, 4000, 4000, 4000)
zbounds, zbounds_len = (-1.,1.), (-1.,1.)
pb_ctr, pb_extent = (0., 2 * np.pi)
tol = 1e-3

cls_path = os.path.join(os.path.dirname(itercurv.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_grad = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_gradlensedCls.dat'))
cls_weights_qe =  utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))


def build_sim(idx, transf, nlev_p):
    lmax = len(transf) - 1
    mmax = lmax
    from plancklens.sims import planck2018_sims
    eblm = np.array([alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_elm(idx), None, lmax, mmax),
                     alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(idx), None, lmax, mmax)])
    almxfl(eblm[0], transf, mmax, inplace=True)
    almxfl(eblm[1], transf, mmax, inplace=True)
    eblm[0] += synalm(np.ones(lmax + 1) * (transf > 0) * (nlev_p / 180 / 60 * np.pi) ** 2, lmax, mmax)
    eblm[1] += synalm(np.ones(lmax + 1) * (transf > 0) * (nlev_p / 180 / 60 * np.pi) ** 2, lmax, mmax)
    return eblm



transf   =  gauss_beam(1./180 / 60 * np.pi, lmax=lmax) * (np.arange(lmax + 1) >= lmin)
transf_i = cli(transf)
fel =  cli(cls_len['ee'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)
fbl =  cli(cls_len['bb'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)
fel_unl =  cli(cls_unl['ee'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)
fbl_unl =  cli(cls_unl['bb'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)
chain_descr = [[0, ["diag_cl"], lmax_unl, 2048, np.inf, 1e-3, cd_solve.tr_cg, cd_solve.cache_mem()]]



def get_itlib(vscarf):
    tr = int(os.environ.get('OMP_NUM_THREADS', 8))
    cpp = cls_unl['pp'][:lmax_qlm + 1]
    if not os.path.exists(TEMP + '/plm0.npy'):
        if not os.path.exists(TEMP + '/eblm_dat.npy'):
            eblm = build_sim(0, transf, nlev_p)
            np.save(TEMP + '/eblm_dat.npy', eblm)
        eblm_wf = np.load(TEMP + '/eblm_dat.npy')
        almxfl(eblm_wf[0], transf_i * fel * cls_len['ee'][:lmax + 1], mmax, inplace=True)
        almxfl(eblm_wf[1], transf_i * fbl * cls_len['bb'][:lmax + 1], mmax, inplace=True)

        sc_job = utils_scarf.scarfjob()
        sc_job.set_thingauss_geometry(lmax_unl, 2)
        d_geo = utils_scarf.pbdGeometry(sc_job.geom, utils_scarf.pbounds(pb_ctr, pb_extent))
        isoppfilter = alm_filter_nlev(nlev_p, transf, (lmax, lmax))
        plm0 = isoppfilter.get_qlms(eblm, eblm_wf, d_geo, lmax_qlm, lmax_qlm)[0]
        R = qresp.get_response('p_p', lmax, 'p', cls_len, cls_len, {'e': fel, 'b': fbl}, lmax_qlm=lmax_qlm)[0]
        almxfl(plm0, utils.cli(R), mmax_qlm, True)
        almxfl(plm0, cpp * utils.cli(cpp + utils.cli(R)), mmax_qlm, True)
        np.save(TEMP + '/plm0.npy', plm0)

    plm0 = np.load(TEMP + '/plm0.npy')
    R_unl = qresp.get_response('p_p', lmax, 'p', cls_unl, cls_unl, {'e': fel_unl, 'b': fbl_unl}, lmax_qlm=lmax_qlm)[0]
    H0_unl = cli(R_unl)

    ffi = remapping.deflection(d_geo, 1.7, np.zeros_like(plm0), mmax_qlm, tr, tr)
    isofilter = alm_filter_nlev_wl(nlev_p, ffi, transf, (lmax_unl, lmax_unl), (lmax, lmax))
    stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5)
    k_geom = isofilter.ffi.geom
    itlib = scarf_iterator.iterator_cstmf( TEMP + '/s4iterator_fullsky_p_p_%04d'%idx + args.scarf, vscarf[0], (lmax_qlm, mmax_qlm), eblm,
            plm0, plm0 * 0., H0_unl, cpp, cls_unl, isofilter, k_geom, chain_descr, stepper)
    return itlib

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-cmb', dest='cmb', action='store_true', help='cmb-only rec')
    parser.add_argument('-btempl', dest='btempl', action='store_true', help='build B-templ for last iter > 0')
    parser.add_argument('-scarf', dest='scarf', type=str, default='', help='minimal sim index')

    #vscarf: 'p' 'k' 'd' for bfgs variable
    #add a 'f' to use full sky in once-per iteration kappa thingy
    #add a 'r' for real space attenuation of the step instead of harmonic space
    # add a '0' for no mf
    soltn_cond = lambda it: True
    tol_iter = lambda it : 1e-3
    args = parser.parse_args()
    from plancklens.helpers import mpi
    mpi.barrier = lambda : 1 # redefining the barrier
    from itercurv.iterators.statics import rec as Rec
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        lib_dir_iterator = TEMP + '/s4iterator_fullsky_p_p_%04d'%idx + args.scarf
        if Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            jobs.append(idx)

    for idx in jobs[mpi.rank::mpi.size]:
        lib_dir_iterator = TEMP + '/s4iterator_fullsky_p_p_%04d'%idx + args.scarf
        if args.itmax >= 0 and Rec.maxiterdone(lib_dir_iterator) < args.itmax:
            itlib = get_itlib(args.scarf)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                chain_descr = [[0, ["diag_cl"], lmax_unl, 2048, np.inf, tol_iter(i), cd_solve.tr_cg, cd_solve.cache_mem()]]
                itlib.chain_descr  = chain_descr
                itlib.soltn_cond = soltn_cond(i)

                print("doing iter " + str(i))
                itlib.iterate(i, 'p')