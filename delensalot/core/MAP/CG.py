import sys
import numpy as np

from delensalot.core.cg import cd_solve, cd_monitors
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl, alm_copy

class multigrid_stage(object):
    def __init__(self, ids, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache):
        self.depth = ids
        self.pre_ops_descr = pre_ops_descr
        self.lmax = lmax
        self.nside = nside
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.tr = tr
        self.cache = cache
        self.pre_ops = []


#NOTE this is actually not a multigrid anymore, check line 35. pure diagonal here.
class conjugate_gradient:
    def __init__(self, pre_op_diag, chain_descr, s_cls, debug_log_prefix=None, plogdepth=0):
        self.debug_log_prefix = debug_log_prefix
        self.plogdepth = plogdepth
        self.chain_descr = chain_descr
        self.s_cls = s_cls
        stages = {}
        for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache] in self.chain_descr:
            stages[id] = multigrid_stage(id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache)
            for pre_op_descr in pre_ops_descr:  # recursively add all stages to stages[0]
                stages[id].pre_ops.append(pre_op_diag)
        self.bstage = stages[0]  # these are the pre_ops called in cd_solve
        self.logger = (lambda iter, eps, stage=self.bstage, **kwargs: self.log(stage, iter, eps, **kwargs))


    def solve(self, soltn, tpn_alm, fwd_op):
        self.watch = cd_monitors.stopwatch()
        self.iter_tot = 0
        self.prev_eps = None

        monitor = cd_monitors.monitor_basic(self.dot_op, logger=self.logger, iter_max=self.bstage.iter_max, eps_min=self.bstage.eps_min, d0=self.dot_op(tpn_alm, tpn_alm))
        cd_solve.cd_solve(soltn, tpn_alm, fwd_op, self.bstage.pre_ops, self.dot_op, monitor, tr=self.bstage.tr, cache=self.bstage.cache)


    def dot_op(self, elm1, elm2):
        lmax = Alm.getlmax(elm1.size, None)
        ell = np.arange(0, lmax + 1)
        weights = 2 * ell + 1
        return np.sum(alm2cl(elm1, elm2, lmax, lmax, None)[0:] * weights)
    

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth > self.plogdepth:
            return

        log_str = '   ' * stage.depth + '(%4d, %04d) [%s] (%d, %.8f)' % (
        stage.nside, stage.lmax, str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

        if self.debug_log_prefix is not None:
            log = open(self.debug_log_prefix + 'stage_all.dat', 'a')
            log.write(log_str)
            log.close()

            if stage.depth == 0:
                f_handle = self.debug_log_prefix + 'stage_soltn_' + str(stage.depth) + '_%04d'%iter +'.npy'
                np.save(f_handle,  kwargs['soltn'])

                #f_handle = self.debug_log_prefix + 'stage_resid_' + str(stage.depth) + '.npy'
                #np.save(f_handle, kwargs['resid']]])
                #f_handle.close()

            log_str = '%05d %05d %10.6e %05d %s\n' % (self.iter_tot, int(elapsed), eps, iter, str(elapsed))
            log = open(self.debug_log_prefix + 'stage_' + str(stage.depth) + '.dat', 'a')
            log.write(log_str)
            log.close()

            if (self.prev_eps is not None) and (self.prev_stage.depth > stage.depth):
                log_final_str = '%05d %05d %10.6e %s\n' % (
                self.iter_tot - 1, int(self.prev_elapsed), self.prev_eps, str(self.prev_elapsed))

                log = open(self.debug_log_prefix + 'stage_final_' + str(self.prev_stage.depth) + '.dat', 'a')
                log.write(log_final_str)
                log.close()

            self.prev_stage = stage
            self.prev_eps = eps
            self.prev_elapsed = elapsed