"""
Module for simulations of birefrengence
"""

from delensalot.sims import generic
import numpy as np
import healpy as hp
import plancklens.shts as shts
import os


class sims_cmb_bir(generic.sims_cmb_len):


    def get_sim_alm(self, idx, field, ret=True):
        if field == "alpha":
            return self.get_sim_alpha(idx, ret)
        else:
            return super().get_sim_alm(idx, field, ret)
        
    def get_sim_alpha(self, idx, ret=True):
        fname = os.path.join(self.lib_dir, 'sim_%04d_alpha.fits' % idx)
        if not os.path.exists(fname):
            alpha_lm = self.get_sim_alpha(self.offset_index(idx, self.offset_plm[0], self.offset_plm[1]))
            if self.cache_plm:
                hp.write_alm(fname, alpha_lm)
        if ret:
            return hp.read_alm(fname)

    def _cache_eblm(self, idx):
        elm = self.unlcmbs.get_sim_elm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
        blm = None if 'b' not in self.fields else self.unlcmbs.get_sim_blm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))

        alpha_lm = self.get_sim_alpha(idx, True)
        alpha = shts.alm2map(alpha_lm, nside=self.nside_lens, lmax=self.lmax)

        spin = 2

        Q, U = shts.alm2map_spin(np.array([elm, blm]), self.nside_lens, spin, self.lmax)
        cos, sin = np.cos(spin*alpha), np.sin(spin*alpha)
        #this assumes no primordial B-modes at the surface of last scattering
        Qprime = Q*cos - U*sin
        Uprime = Q*sin + U*cos
        elm, blm = shts.map2alm_spin([Qprime, Uprime], spin, lmax=self.lmax)

        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx), elm)
        del elm
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx), blm)