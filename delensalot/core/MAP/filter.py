
import numpy as np

from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.core.cg import cd_solve, cd_monitors, multigrid
from . import operator

class base:
    def __init__(self, operator_descs, **filter_desc):
        self.operators = [operator(operator_desc) for operator_desc in operator_descs]


    def get_XWF(self, field):
        cg_sol_curr, cg_it_curr = self._get_cg_startingpoint(it)

        if cg_it_curr < it - 1:
            # CG inversion
            # TODO operators must be passed to the CG solver
            # TODO operator must be updated with current field estimates
            self.cg.solve(cg_sol_curr, self.data, self.operators) # build new cg that knows how to apply the operators to obtain the forward operation
            self.mchain.solve(cg_sol_curr, self.data, dot_op=self.filter.dot_op())
            self.wf_cacher.cache(self.wf_fns.format(it=it - 1), cg_sol_curr)


    def get_ivf(self, eblm_dat, eblm_wf):
        # self.filter.
        for operator in self.operators:
            # FIXME correct structure - and add all operators from fields: 'B^t N^{-1}(X^{\rm dat} - B D X^{WF})`
            operator.act(eblm_dat, eblm_wf)


    def _get_cg_startingpoint(self, it):
        for _it in np.arange(it - 1, -1, -1):
            if self.wf_cacher.is_cached(self.wf_fns.format(it=_it)):
                return self.wf_cacher.load(self.wf_fns.format(it=_it)), _it
        if callable(self.wflm0):
            return self.wflm0(), -1
        return np.zeros((1, Alm.getsize(self.lmax_filt, self.mmax_filt)), dtype=complex).squeeze(), -1
    



    # # This is the actual ivf. Need to replace operations by operator.
    # def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray, q_pbgeom:pbdGeometry, map_out=None):
    #     """Builds inverse variance weighted map to feed into the QE
    #         :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`
    #     """
    #     assert len(eblm_dat) == 2
    #     ebwf = self.ffi.lensgclm(np.atleast_2d(eblm_wf), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
    #     almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
    #     almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
    #     ebwf += eblm_dat
    #     almxfl(ebwf[0], self.inoise_1_elm * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
    #     almxfl(ebwf[1], self.inoise_1_blm * 0.5,            self.mmax_len, True)
    #     return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.ffi.sht_tr, map=map_out)
    