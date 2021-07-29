"""Builds a QE with the opfilt_iso_pp instance and plots


"""
import numpy as np
import os
import pylab as pl
from plancklens import utils, qresp
import plancklens
from lenscarf.utils import cli
from lenscarf.utils_hp import gauss_beam, almxfl, alm2cl, synalm, alm_copy
from lenscarf.opfilt.opfilt_iso_tt import alm_filter_nlev
from lenscarf import utils_scarf
from plancklens.sims import planck2018_sims

cls_path = os.path.join(os.path.dirname(plancklens.__file__), 'data', 'cls')
cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

def build_sim(idx, transf, nlev_t):
    lmax = len(transf) - 1
    mmax = lmax
    tlm = alm_copy(planck2018_sims.cmb_len_ffp10.get_sim_tlm(idx), None, lmax, mmax)
    almxfl(tlm, transf, mmax, inplace=True)
    tlm += synalm(np.ones(lmax + 1) * (transf > 0) * (nlev_t / 180 / 60 * np.pi) ** 2, lmax, mmax)
    return tlm

lmin, lmax, mmax, beam, nlev_t, nlev_p = (30, 3000, 3000, 1., 1.5, 1.5 * np.sqrt(2.))
transf   =  gauss_beam(1./180 / 60 * np.pi, lmax=lmax) * (np.arange(lmax + 1) >= lmin)
transf_i = cli(transf)
ftl =  cli(cls_len['tt'][:lmax + 1] + (nlev_t / 180 / 60 * np.pi) ** 2 * cli(transf ** 2)) * (transf > 0)

tlm = build_sim(0, transf, nlev_t)
tlm_wf = np.copy(tlm)
almxfl(tlm_wf, transf_i * ftl * cls_len['tt'][:lmax + 1], mmax, inplace=True)

lmax_qlm, mmax_qlm = (4096, 4096)
sc_job = utils_scarf.scarfjob()
sc_job.set_thingauss_geometry(4096, 2)
d_geo = utils_scarf.pbdGeometry(sc_job.geom, utils_scarf.pbounds(0, 2. * np.pi))
isoppfilter = alm_filter_nlev(nlev_t, transf, (lmax, lmax))
G = isoppfilter.get_qlms(tlm, tlm_wf, d_geo, lmax_qlm, mmax_qlm)[0]
R = qresp.get_response('ptt', lmax, 'p', cls_len, cls_len, {'t': ftl}, lmax_qlm=lmax_qlm)[0]
almxfl(G, utils.cli(R), mmax_qlm, True)

ls = np.arange(2, 3000+ 1)
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi * alm2cl(G, G, lmax_qlm, mmax_qlm, lmax_qlm)[ls])
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi / R[ls])
plm_in = alm_copy(planck2018_sims.cmb_unl_ffp10.get_sim_plm(0), None, lmax_qlm, mmax_qlm)
pl.loglog(ls, ls ** 2 * (ls + 1.) ** 2 * 1e7 / 2 / np.pi * alm2cl(G, plm_in , lmax_qlm, mmax_qlm, lmax_qlm)[ls])
pl.show()