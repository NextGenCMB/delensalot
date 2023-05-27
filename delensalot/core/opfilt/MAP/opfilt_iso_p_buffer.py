    def apply_alm(self, elm:np.ndarray):

        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)

        eblm = self.ffi.lensgclm(np.array([elm, np.zeros_like(elm)]), self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)

        # backward lensing with magn. mult. here
        eblm = self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol, backwards=True)
        elm[:] = eblm[0]



    def apply_alm_old(self, elm:np.ndarray):
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        # View to the same array for GRAD_ONLY mode:
        elm_2d = elm.reshape((1, elm.size))
        eblm = self.ffi.lensgclm(elm_2d, self.mmax_sol, 2, self.lmax_len, self.mmax_len)
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)

        # NB: inplace is fine but only if precision of elm array matches that of the interpolator
        self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
                                 backwards=True, gclm_out=elm_2d, out_sht_mode='GRAD_ONLY')
        #elm[:] = self.ffi.lensgclm(eblm, self.mmax_len, 2, self.lmax_sol, self.mmax_sol,
        #                 backwards=True, out_sht_mode='GRAD_ONLY').squeeze()
