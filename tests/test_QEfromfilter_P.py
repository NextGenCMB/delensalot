"""Builds a QE with the opfilt_iso_pp instance and plots


"""
import numpy as np
import os
import pylab as pl
from plancklens import utils, qresp
import plancklens
from dlensalot.utils import cli
from dlensalot.utils_hp import gauss_beam, almxfl, alm2cl, synalm, alm_copy
from dlensalot.opfilt.opfilt_iso_pp import alm_filter_nlev
from dlensalot import utils_scarf
from plancklens.sims import planck2018_sims

cls_path = os.path.join(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

def build_sim(idx, transf, nlev_p):
    lmax = len(transf) - 1
    mmax = lmax
    eblm = np.array([alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_elm(idx), None, lmax, mmax),
                     alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_blm(idx), None, lmax, mmax)])
    almxfl(eblm[0], transf, mmax, inplace=True)
    almxfl(eblm[1], transf, mmax, inplace=True)
    eblm[0] += synalm(np.ones(lmax + 1) * (transf > 0) * (nlev_p / 180 / 60 * np.pi) ** 2, lmax, mmax)
    eblm[1] += synalm(np.ones(lmax + 1) * (transf > 0) * (nlev_p / 180 / 60 * np.pi) ** 2, lmax, mmax)
    return eblm

lmin, lmax, mmax, beam, nlev_t, nlev_p = (30, 3000, 3000, 1., 1.5, 1.5 * np.sqrt(2.))
transf   =  gauss_beam(1./180 / 60 * np.pi, lmax=lmax) * (np.arange(lmax + 1) >= lmin)
transf_i = cli(transf)
fel =  cli(cls_len['ee'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)
fbl = cli(cls_len['bb'][:lmax + 1] + (nlev_p / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)

eblm = build_sim(0, transf, nlev_p)
eblm_wf = np.copy(eblm)
almxfl(eblm_wf[0], transf_i * fel * cls_len['ee'][:lmax + 1], mmax, inplace=True)
almxfl(eblm_wf[1], transf_i * fbl * cls_len['bb'][:lmax + 1], mmax, inplace=True)

lmax_qlm, mmax_qlm = (4096, 4096)
sc_job = utils_scarf.scarfjob()
sc_job.set_thingauss_geometry(4096, 2)
d_geo = utils_scarf.pbdGeometry(sc_job.geom, utils_scarf.pbounds(0, 2. * np.pi))
isoppfilter = alm_filter_nlev(nlev_p, transf, (lmax, lmax))
G = isoppfilter.get_qlms(eblm, eblm_wf, d_geo, lmax_qlm, mmax_qlm)[0]
R = qresp.get_response('p_p', lmax, 'p', cls_len, cls_len, {'e': fel, 'b': fbl}, lmax_qlm=lmax_qlm)[0]
almxfl(G, utils.cli(R), mmax_qlm, True)

ls = np.arange(2, 3000+ 1)
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi * alm2cl(G, G, lmax_qlm, mmax_qlm, lmax_qlm)[ls])
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi / R[ls])
plm_in = alm_copy(planck2018_sims.cmb_unl_ffp10.get_sim_plm(0), None, lmax_qlm, mmax_qlm)
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi * alm2cl(G, plm_in , lmax_qlm, mmax_qlm, lmax_qlm)[ls])
pl.show()