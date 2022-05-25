"""user interface to application level. All necessary user - iterator-functionalities should be accessible via this module.

Returns:
    _type_: _description_
"""

import os
import numpy as np
import argparse

from plancklens.helpers import mpi
from lenscarf.iterators.statics import rec


def init(TEMP):
    if not os.path.exists(TEMP):
        os.makedirs(TEMP)

def get_parser():
    parser = argparse.ArgumentParser(description='test iterator full-sky with pert. resp.')
    parser.add_argument('-itmax', dest='itmax', type=int, default=-1, help='maximal iter index')
    parser.add_argument('-imin', dest='imin', type=int, default=-1, help='minimal sim index')
    parser.add_argument('-imax', dest='imax', type=int, default=-1, help='maximal sim index')
    parser.add_argument('-btempl', dest='btempl', action='store_true', help='build B-templ for last iter > 0')

    return parser


def collect_jobs(TEMP, args):
    jobs = []
    for idx in np.arange(args.imin, args.imax + 1):
        TEMP_it = TEMP + '/iterator_p_p_%04d_OBD' % idx
        if rec.maxiterdone(TEMP_it) < args.itmax:
            jobs.append(idx)


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