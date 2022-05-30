"""user interface to application level. All necessary user - iterator-functionalities should be accessible via this module.

Returns:
    _type_: _description_
"""

import os
import numpy as np
import argparse

from plancklens.helpers import mpi

from lenscarf.iterators.statics import rec as Rec

def init(TEMP):
    if not os.path.exists(TEMP):
        os.makedirs(TEMP)

def get_parser_btemp():
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-btempl', dest='btempl', action='store_true', help='build B-templ for last iter > 0')

    return parser.parse_args()

def get_parser_paramfile():

    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-k', dest='k', type=str, default='p_p', help='rec. type')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-tol', dest='tol', type=float, default=5., help='-log10 of cg tolerance default')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')

    return parser.parse_args()


def collect_jobs(parser:argparse, libdir_iterators):        
    jobs = []
    for idx in np.arange(parser.imin, parser.imax + 1):
        lib_dir_iterator = libdir_iterators(parser.k, idx, parser.v)
        if Rec.maxiterdone(lib_dir_iterator) < parser.itmax:
            jobs.append(idx)
    return jobs


def run(TEMP, get_itlib, jobs, args):
    mpi.barrier = lambda : 1 # redefining the barrier
    for idx in jobs[mpi.rank::mpi.size]:
        TEMP_it = TEMP + '/iterator_p_p_%04d_OBD' % idx
        if args.itmax >= 0 and rec.maxiterdone(TEMP_it) < args.itmax:
            itlib = get_itlib(qe_key, idx)
            for i in range(args.itmax + 1):
                print("****Iterator: setting cg-tol to %.4e ****"%tol_iter(i))
                print("****Iterator: setting solcond to %s ****"%soltn_cond(i))
                chain_descr = [[0, ["diag_cl"], lmax_filt, nside, np.inf, tol_iter(i), cd_solve.tr_cg, cd_solve.cache_mem()]]
                itlib.chain_descr  = chain_descr
                itlib.soltn_cond = soltn_cond(i)

                print("Starting iter " + str(i))
                itlib.iterate(i, 'p')
            # Produces B-template for last iteration
            if args.btempl and args.itmax > 0:
                blm = cs_iterator.get_template_blm(args.itmax, args.itmax, lmax_b=2048, lmin_plm=1)  