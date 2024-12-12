
class base:
    def __init__(self, **cg_desc):
        self.operators = cg_desc['operators']


    def fwd_op(self, X):
        #FIXME needs to go both ways
        for operator in self.operators:
            buff = operator.act(X)
            X = buff


    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        self.tim.reset()
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = elm.reshape((1, elm.size))
        eblm = self.ffi.lensgclm(elm_2d, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        self.tim.add('lensgclm fwd')
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)
        self.tim.add('transf')

        # NB: inplace is fine but only if precision of elm array matches that of the interpolator
        self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
                                 backwards=True, gclm_out=elm_2d, out_sht_mode='GRAD_ONLY')
        #elm[:] = self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
        #                 backwards=True, out_sht_mode='GRAD_ONLY').squeeze()
        # elm[:] = self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True, out_sht_mode='GRAD_ONLY')
        self.tim.add('lensgclm bwd')
        if self.verbose:
            print(self.tim)