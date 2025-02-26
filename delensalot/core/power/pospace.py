"""Position-space deconvolution routines

    Based on plancklens wigners implementation

"""
import healpy as hp
import numpy as np

# from plancklens.wigners import wigners
from plancklens import utils


def map2alm(tmap, lmax=None, mmax=None, zbounds=np.array([-1., 1.]), iter=0):
    return hp.map2alm(tmap, lmax=lmax, mmax=mmax, iter=0)
def alm2map(alm, nside, lmax=None, mmax=None, zbounds=np.array([-1., 1.]), iter=0):
    return hp.alm2map(alm, nside, lmax=lmax, mmax=mmax, verbose=False)
def map2alm_spin(maps, spin, lmax=None, mmax=None, zbounds=np.array([-1., 1.]), iter=0):
    return hp.map2alm_spin(maps, spin, lmax=lmax, mmax=mmax)
def alm2map_spin(alms, nside, spin, lmax, mmax=None, zbounds=np.array([-1., 1.]), iter=0):
    return hp.alm2map_spin(alms, nside, spin, lmax, mmax=mmax)

class mod:
    pass
hph = mod()
hph.map2alm = map2alm
hph.map2alm_spin = map2alm_spin
hph.alm2map_spin = alm2map_spin
hph.alm2map = alm2map


def _wwi(ww, verbose=False):
    """Inverse of mask 2pcf to plugin

    """
    ret = np.zeros(len(ww), dtype=float)
    ii, = np.where(ww > 0.)
    if verbose and len(ii) < ww.size:
        print("*** inverse ww: not all elements > 0")
    ret[ii] = 1. / ww[ii]
    return ret

def map2cl(tmap, mask, lmax, lmax_mask, tmap2=None, npts=None, ww=None, zbounds=np.array([-1.,1.])):
    """Position space 2pcf deconvolver for spin-0 maps.

        This assumes all distances can be probed

    """
    assert mask.size == tmap.size, (mask.size, tmap.size)

    nside = hp.npix2nside(tmap.size)
    if npts is None:
        npts = min(15000, 2 * max(lmax, lmax_mask))
    xg, wg = wigners.get_xgwg(-1., 1., npts)
    palm = hph.map2alm(tmap * mask, lmax=min(3 * nside - 1, lmax + lmax_mask), zbounds=zbounds)
    if ww is None :
        ww = wigners.wignerpos(hp.alm2cl(hph.map2alm(mask, lmax=lmax_mask)), xg, 0, 0) # mask 2pcf
    if tmap2 is None:
        return wigners.wignercoeff(wigners.wignerpos(hp.alm2cl(palm), xg, 0, 0) * _wwi(ww, verbose=True) * wg, xg, 0, 0, lmax=lmax)
    else:
        palm2 = hph.map2alm(tmap2 * mask, lmax=min(3 * nside - 1, lmax + lmax_mask), zbounds=zbounds)
        return wigners.wignercoeff(wigners.wignerpos(hp.alm2cl(palm, alms2=palm2), xg, 0, 0) * _wwi(ww, verbose=True) * wg, xg, 0, 0, lmax=lmax)

def map2cl_spin(qumap, spin, mask, lmax, lmax_mask, qumap2=None, spin2=None, ret_eb_be=False,
                npts=None, ww=None, zbounds=np.array([-1.,1.])):
    """Position space 2pcf deconvolver for spin non-zero maps.

        This assumes all distances can be probed

        This can also be used to get ET and BT by using spin2 = 0

        Note:
            No pure estimator here

    """
    assert spin > 0, spin
    if spin2 is None:
        spin2 = spin
    assert spin2 >= 0, spin2
    assert len(qumap) == 2 and qumap[0].size == qumap[1].size
    assert mask.size == qumap[0].size, (mask.size, qumap[0].size)

    nside = hp.npix2nside(mask.size)
    if npts is None:
        npts = min(15000, 2 * max(lmax, lmax_mask))
    xg, wg = wigners.get_xgwg(-1., 1., npts)
    if ww is None:
        ww = wigners.wignerpos(hp.alm2cl(hph.map2alm(mask, lmax=lmax_mask)), xg, 0, 0) # mask 2pcf

    elm, blm = hph.map2alm_spin([qumap[0] * mask, qumap[1] * mask], spin,
                                lmax=min(3 * nside - 1, lmax + lmax_mask), zbounds=zbounds)
    if qumap2 is None:
        ee = hp.alm2cl(elm)
        bb = hp.alm2cl(blm)
        eb = hp.alm2cl(elm, alms2=blm)
        be = hp.alm2cl(blm, alms2=elm)
    else:
        if spin2 > 0:
            assert len(qumap2) == 2, len(qumap2)
            elm2, blm2 = hph.map2alm_spin([qumap2[0] * mask, qumap2[1] * mask], spin2,
                                lmax=min(3 * nside - 1, lmax + lmax_mask), zbounds=zbounds)
            ee = hp.alm2cl(elm, alms2=elm2)
            bb = hp.alm2cl(blm, alms2=blm2)
            eb = hp.alm2cl(elm, alms2=blm2)
            be = hp.alm2cl(blm, alms2=elm2)
            del elm2, blm2
        else:
            assert qumap2.size == mask.size, (qumap2.size, mask.size)
            tlm2 = hph.map2alm(qumap2 * mask,
                                lmax=min(3 * nside - 1, lmax + lmax_mask), zbounds=zbounds)
            ee = hp.alm2cl(elm, alms2=tlm2)
            bb = np.zeros_like(ee)
            eb = np.zeros_like(ee)
            be = hp.alm2cl(blm, alms2=tlm2)
            del tlm2
    del elm, blm

    xi_p = wigners.wignerpos(ee + bb, xg,  spin, spin2)  #xi plus  with EE + BB - i (EB - BE)
    xi_m = wigners.wignerpos(ee - bb, xg, -spin, spin2)  #xi minus with EE - BB-  i (EB + BE)
    ret_p = wigners.wignercoeff(wg * xi_p * _wwi(ww, verbose=True), xg,  spin, spin2, lmax)
    ret_m = wigners.wignercoeff(wg * xi_m * _wwi(ww, verbose=True), xg, -spin, spin2, lmax)
    ee = 0.5 * (ret_p + ret_m)
    bb = 0.5 * (ret_p - ret_m)
    if not ret_eb_be:
        return ee, bb # or TE, 0 for spin2 0
    xi_p =  wigners.wignerpos(eb - be, xg,  spin, spin2)  #xi plus  with EE + BB - i (EB - BE)
    xi_m =  wigners.wignerpos(eb + be, xg, -spin, spin2)  #xi minus with EE - BB-  i (EB + BE)
    ret_p = wigners.wignercoeff(wg * xi_p * _wwi(ww, verbose=True), xg,  spin, spin2, lmax)
    ret_m = wigners.wignercoeff(wg * xi_m * _wwi(ww, verbose=True), xg, -spin, spin2, lmax)
    eb =  0.5 * (ret_p + ret_m)
    be = -0.5 * (ret_p - ret_m)
    return ee, bb, eb, be # or TE, 0, 0, TB for spin2 0

def map2cls(tqumap, tmask, pmask, lmax, lmax_mask, tqumap2=None,
                npts=None, ww=None, zbounds=np.array([-1.,1.])):
    """Position space 2pcf deconvolver for TQU maps

        This assumes all distances can be probed

        Args:
            tqumap: list with T, Q and U Stokes maps
            tmask: mask fo the correct healpy resolution for intensity maps
            pmask: mask fo the correct healpy resolution for polarisation maps
            lmax: spectra are returned up to this multipole
            lmax_mask: the maximum multipole of the mask to consider
            tqumap2: set this to another set of maps for cross-spectra

            npts: number of point in GL integrations
            ww: set this to mask 2pcf  if already calculated
            zbounds: co-latitude cuts if relevant

        Returns:
            TT, EE, BB, TE, TB, EB, ET, BT, BE power spectra

        Note:
            No pure estimator here

    """
    assert len(tqumap) == 3
    assert tmask.size == tqumap[0].size, (tmask.size, tqumap[0].size)
    assert pmask.size == tqumap[1].size, (pmask.size, tqumap[1].size)
    assert pmask.size == tqumap[2].size, (pmask.size, tqumap[2].size)

    tnside = hp.npix2nside(tmask.size)
    pnside = hp.npix2nside(pmask.size)
    if npts is None:
        npts = min(15000, 2 * max(lmax, lmax_mask))
    xg, wg = wigners.get_xgwg(-1., 1., npts)
    if ww is None:
        ww = wigners.wignerpos(hp.alm2cl(hph.map2alm(tmask, lmax=lmax_mask)), xg, 0, 0) # mask 2pcf

    elm, blm = hph.map2alm_spin([tqumap[1] * pmask, tqumap[2] * pmask], 2,
                                lmax=min(3 * pnside - 1, lmax + lmax_mask), zbounds=zbounds)
    tlm = hph.map2alm(tqumap[0] * tmask,
                               lmax=min(3 * tnside - 1, lmax + lmax_mask), zbounds=zbounds)

    if tqumap2 is None:
        tt = hp.alm2cl(tlm)
        et = hp.alm2cl(elm, alms2=tlm)
        te = hp.alm2cl(tlm, alms2=elm)
        bt = hp.alm2cl(blm, alms2=tlm)
        tb = hp.alm2cl(tlm, alms2=blm)
        ee = hp.alm2cl(elm)
        bb = hp.alm2cl(blm)
        eb = hp.alm2cl(elm, alms2=blm)
        be = hp.alm2cl(blm, alms2=elm)
    else:
        assert len(tqumap2) == 3
        assert tmask.size == tqumap2[0].size, (tmask.size, tqumap2[0].size)
        assert pmask.size == tqumap2[1].size, (pmask.size, tqumap2[1].size)
        assert pmask.size == tqumap2[2].size, (pmask.size, tqumap2[2].size)
        elm2, blm2 = hph.map2alm_spin([tqumap2[1] * pmask, tqumap2[2] * pmask], 2,
                                lmax=min(3 * pnside - 1, lmax + lmax_mask), zbounds=zbounds)
        tlm2 = hph.map2alm(tqumap2[0] * tmask,
                               lmax=min(3 * tnside - 1, lmax + lmax_mask), zbounds=zbounds)
        tt = hp.alm2cl(tlm, alms2=tlm2)
        et = hp.alm2cl(elm, alms2=tlm2)
        te = hp.alm2cl(tlm, alms2=elm2)
        bt = hp.alm2cl(blm, alms2=tlm2)
        tb = hp.alm2cl(tlm, alms2=blm2)
        ee = hp.alm2cl(elm, alms2=elm2)
        bb = hp.alm2cl(blm, alms2=blm2)
        eb = hp.alm2cl(elm, alms2=blm2)
        be = hp.alm2cl(blm, alms2=elm2)
        del tlm2, elm2, blm2

    del tlm, elm, blm

    xi = wigners.wignerpos(tt, xg,  0, 0)
    xi_p = wigners.wignerpos(ee + bb, xg,  2, 2)  #xi plus  with EE + BB - i (EB - BE)
    xi_m = wigners.wignerpos(ee - bb, xg, -2, 2)  #xi minus with EE - BB-  i (EB + BE)
    xi_et = wigners.wignerpos(et, xg,  2, 0)
    xi_te = wigners.wignerpos(te, xg,  0, 2)

    ret = wigners.wignercoeff(wg * xi * _wwi(ww, verbose=True), xg,  0, 0, lmax)
    ret_p = wigners.wignercoeff(wg * xi_p * _wwi(ww, verbose=True), xg,  2, 2, lmax)
    ret_m = wigners.wignercoeff(wg * xi_m * _wwi(ww, verbose=True), xg, -2, 2, lmax)
    ret_et = wigners.wignercoeff(wg * xi_et * _wwi(ww, verbose=True), xg,  2, 0, lmax)
    ret_te = wigners.wignercoeff(wg * xi_te * _wwi(ww, verbose=True), xg,  0, 2, lmax)

    tt = ret
    ee = 0.5 * (ret_p + ret_m)
    bb = 0.5 * (ret_p - ret_m)
    et = ret_et
    te = ret_te

    xi_p =  wigners.wignerpos(eb - be, xg,  2, 2)  #xi plus  with EE + BB - i (EB - BE)
    xi_m =  wigners.wignerpos(eb + be, xg, -2, 2)  #xi minus with EE - BB-  i (EB + BE)
    xi_bt = wigners.wignerpos(bt, xg,  2, 0)
    xi_tb = wigners.wignerpos(tb, xg,  0, 2)

    ret_p = wigners.wignercoeff(wg * xi_p * _wwi(ww, verbose=True), xg,  2, 2, lmax)
    ret_m = wigners.wignercoeff(wg * xi_m * _wwi(ww, verbose=True), xg, -2, 2, lmax)
    ret_bt = wigners.wignercoeff(wg * xi_bt * _wwi(ww, verbose=True), xg,  2, 0, lmax)
    ret_tb = wigners.wignercoeff(wg * xi_tb * _wwi(ww, verbose=True), xg,  0, 2, lmax)

    eb =  0.5 * (ret_p + ret_m)
    be = -0.5 * (ret_p - ret_m)
    bt = ret_bt
    tb = ret_tb
    return tt, ee, bb, te, tb, eb, et, bt, be



class map2cl_binned:

    def __init__(self, mask, cl_templ, edges, lmax_mask, npts=None, zbounds=(-1., 1.)):
        """Binned coupling matrix for polspice-like approach to deconvolution of spin-0 spectrum estimate

            The number of bins is expected to be reasonable (cost is that of nbins wigner transforms)

            This bins spectra in amplitudes x template_a, for a a number of templates non overlapping in harmonic space
            This is the unity matrix when the 2pcf of the mask can be probed at all distances

            Args:
                mask      : mask to apply on maps
                cl_templ  : cl-array defining the templates for each bin in conjunction with the edges inputs
                edges     : multipoles defining the bins
                lmax_mask : highest multipole considered for the mask
                zbounds   : uses latitude bounded shts if set

            #FIXME: check modifs to apport if non-boolean mask (if at all?)

        """
        self.edges = edges
        self.cl_templ = cl_templ
        self.mask = mask

        lmax = np.max(edges)

        if npts is None:
            npts = min(15000, 2 * max(lmax, lmax_mask))

        self.npts = npts
        self.lmax = lmax
        self.lmax_mask = lmax_mask

        self.zbounds=zbounds
        M, xg_0, ww, xg, wg = self._get_mab(self.mask, lmax_mask)

        self.xg_0 = xg_0
        self.Mi = np.linalg.inv(M)
        self.wwi_wg = _wwi(ww) * wg
        self.xg = xg

        self.fsky2 = np.mean(self.mask ** 2)



    @staticmethod
    def _zro_criterion(ww, ww0):
        return ww <= 1e-2 * ww0 ** 2

    def _edges2blsbusbcs(self): #FIXME: do smthg like lav fidbp
        bls = self.edges[:-1]
        bus = self.edges[1:] - 1
        bus[-1] += 1
        return bls, bus, 0.5 * (bls + bus)

    def get_crude_errors(self):
        """Super-crude error bars based on fsky mode-counting"""
        sig = []
        bls, bus, bcs = self._edges2blsbusbcs()
        for (bl, bu) in zip(bls, bus):
            sig.append(np.sum(2 * np.arange(bl, bu + 1) + 1.))
        return np.sqrt(2./ np.array(sig) / self.fsky2)

    def get_fid_bandpowers(self):#FIXME: better scheme
        """Binned template value

            This has the merit of being on the unbinned curve at bcs

        """
        bls, bus, bcs = self._edges2blsbusbcs()
        return self.cl_templ[np.int_(bcs)]

    def _get_mab(self,  mask, lmax_mask):
        """"Matrix actual calculation

            Returns:
                nbins x nbins matrix decoupling matrix

            This finds the angle at which the mask 2cpf is zero and build the corresponding matrix

        """
        print("Calculating coupling matrix...")
        xg, wg = wigners.get_xgwg(-1., 1., self.npts)
        xg_0 = -1.
        mpcl = hp.alm2cl(hph.map2alm(mask, lmax=lmax_mask))
        ww = wigners.wignerpos(mpcl, xg, 0, 0)  # mask 2pcf
        ww0 = wigners.wignerpos(mpcl, 1., 0, 0)[0]
        zros = np.where(self._zro_criterion(ww, ww0))[0]
        if len(zros) > 0:
            xg_0 = xg[zros[-1]]
            print("zero ww at %.1f deg" % (np.arccos(xg_0) / np.pi * 180))
        xg, wg = wigners.get_xgwg(xg_0, 1., self.npts)
        ww = wigners.wignerpos(mpcl, xg, 0, 0)  # mask 2pcf
        assert np.all(ww > 0)

        Nb = len(self.edges) - 1

        xis = np.zeros((Nb, self.npts))
        xis_inv = np.zeros((Nb, self.npts))
        Nbs = np.zeros(Nb)
        bls, bus, bcs = self._edges2blsbusbcs()
        for ia, (bl, bu) in utils.enumerate_progress(list(zip(bls, bus))):
            cl = np.zeros(bu + 1, dtype=float)
            cl[bl:bu + 1] = self.cl_templ[bl:bu + 1]
            xis[ia] = wigners.wignerpos(cl, xg, 0, 0)
            cl[bl:bu + 1] = utils.cli(self.cl_templ[bl:bu + 1])
            xis_inv[ia] = wigners.wignerpos(cl, xg, 0, 0)
            Nbs[ia] = np.sum((2 * np.arange(bl, bu + 1) + 1)) / (4. * np.pi)
        Mab = np.zeros((Nb, Nb))
        for ia in range(Nb):
            for ib in range(Nb):
                Mab[ia, ib] = np.sum(xis[ib] * xis_inv[ia] * wg) / Nbs[ia]
        return Mab * (2 * np.pi), xg_0, ww, xg, wg


    def _map2pcl(self, tmap, tmap2=None):
        """Undeconvolved spectra amplitudes

        """
        nside = hp.npix2nside(tmap.size)
        Nb = len(self.edges) - 1
        if tmap2 is None:
            pcl = hp.alm2cl(hph.map2alm(tmap * self.mask, lmax=min(3 * nside - 1, self.lmax), zbounds=self.zbounds))
        else:
            alm1 = hph.map2alm(tmap  * self.mask, lmax=min(3 * nside - 1, self.lmax), zbounds=self.zbounds)
            alm2 = hph.map2alm(tmap2 * self.mask, lmax=min(3 * nside - 1, self.lmax), zbounds=self.zbounds)
            pcl = hp.alm2cl(alm1, alms2=alm2)
            del alm1, alm2
        cl = wigners.wignercoeff(wigners.wignerpos(pcl, self.xg, 0, 0) * self.wwi_wg, self.xg, 0, 0, lmax=self.lmax)
        ret = np.zeros(Nb)
        for ia, (bl, bu) in enumerate(zip(self.edges[:-1], self.edges[1:] - 1)):
            bu += (ia == Nb)
            l2p = 2 * np.arange(bl, bu + 1) + 1.
            ret[ia] = np.sum(cl[bl:bu + 1] * utils.cli(self.cl_templ[bl:bu + 1]) * l2p) / np.sum(l2p)
        return ret

    def map2cl(self, tmap, tmap2=None):
        """Position space 2pcf deconvolver for spin-0 maps.

            Args:
                tmap : healpy map to get the spectrum of
                tmap2: second map, in case of a cross-spectrum


        """
        return np.dot(self.Mi, self._map2pcl(tmap, tmap2=tmap2))


class map2cl_spin_binned:

    def __init__(self, mask, spin, clg_templ, clc_templ, edges, lmax_mask, npts=None, zbounds=(-1., 1.)):
        """Binned coupling matrix for polspice-like approach to deconvolution of spin-0 spectrum estimate

            The number of bins is expected to be reasonable (cost is that of nbins wigner transforms)

            This bins spectra in amplitudes x template_a, for a a number of templates non overlapping in harmonic space
            This is the unity matrix when the 2pcf of the mask can be probed at all distances

            Args:
                mask      : mask to apply on maps
                spin      : spin of the field (>0)
                clg_templ  : cl-array defining the gradient mode templates for each bin in conjunction with the edges
                clc_templ  : cl-array defining the curl mode templates for each bin in conjunction with the edges
                edges     : multipoles defining the bins
                lmax_mask : highest multipole considered for the mask
                zbounds   : uses latitude bounded shts if set

            Note:
                No pure estimator here


            #FIXME: check modifs to apport if non-boolean mask (if at all?)

            #FIXME: problems if cc equals gg, in which case xi_m = 0.

        """
        assert spin > 0, spin
        assert np.all( (clg_templ - clc_templ) > 0), 'fix input template issues'
        assert np.all( (clg_templ + clc_templ) > 0), 'fix input template issues'

        self.spin = spin
        self.edges = np.array(edges)
        self.clg_templ = clg_templ
        self.clc_templ = clc_templ

        self.mask = mask

        lmax = np.max(edges)

        if npts is None:
            npts = min(15000, 2 * max(lmax, lmax_mask))

        self.npts = npts
        self.lmax = lmax
        self.lmax_mask = lmax_mask

        self.zbounds=zbounds
        Mp, Mm, xg_0, ww, xg, wg = self._get_mab(self.mask, lmax_mask)

        self.xg_0 = xg_0
        self.Mpi = np.linalg.inv(Mp)
        self.Mmi = np.linalg.inv(Mm)

        self.wwi_wg = _wwi(ww) * wg
        self.xg = xg

        self.fsky2 = np.mean(self.mask ** 2)



    @staticmethod
    def _zro_criterion(ww, ww0):
        return ww <= 1e-2 * ww0 ** 2

    def _edges2blsbusbcs(self): #FIXME: better lavs
        bls = self.edges[:-1]
        bus = self.edges[1:] - 1
        bus[-1] += 1
        #fidee, fidbb = self.get_fid_bandpowers()
        #bcs_ee = spl(np.log10(self.clg_templ[2:np.max(bus) + 1]), 1. * np.arange(2, np.max(bus) + 1), s=0, k=2)(np.log10(fidee))
        #bcs_bb = spl(np.log10(self.clg_templ[2:np.max(bus) + 1]), 1. * np.arange(2, np.max(bus) + 1), s=0, k=2)(np.log10(fidbb))

        return bls, bus, 0.5 * (bls + bus), 0.5 * (bls + bus)

    def get_crude_errors(self):
        """Super-crude error bars based on fsky mode-counting"""
        sig = []
        bls, bus, bcse, bcsb = self._edges2blsbusbcs()
        for (bl, bu) in zip(bls, bus):
            sig.append(np.sum(2 * np.arange(bl, bu + 1) + 1.))
        return np.sqrt(2./ np.array(sig) / self.fsky2)

    def get_fid_bandpowers(self):#FIXME: better scheme
        """Binned template value

            This has the merit of being on the unbinned curve at bcs

        """
        #bls = self.edges[:-1]
        #bus = self.edges[1:] - 1
        #bus[-1] += 1
        #bps_g = np.zeros(len(bus))
        #bps_c = np.zeros(len(bus))

        #for ib, (bl, bu) in enumerate(zip(bls, bus)):
        #    w = 2 * np.arange(bl, bu + 1) + 1
        #    bps_g[ib] = np.sum(w * self.clg_templ[bl:bu + 1]) / np.sum(w)
        #    bps_c[ib] = np.sum(w * self.clc_templ[bl:bu + 1]) / np.sum(w)
        bls, bus, bcse, bcsb = self._edges2blsbusbcs()
        return self.clg_templ[np.int_(bcse)], self.clc_templ[np.int_(bcsb)]

    def _get_mab(self,  mask, lmax_mask):
        """"Matrix actual calculation

            Returns:
                two nbins x nbins matrix decoupling matrix for gg + cc and  gg - cc amplitude

            This finds the angle at which the mask 2cpf is zero and build the corresponding matrix

        """
        print("Calculating coupling matrix...")
        xg, wg = wigners.get_xgwg(-1., 1., self.npts)
        xg_0 = -1.
        mpcl = hp.alm2cl(hph.map2alm(mask, lmax=lmax_mask))
        ww = wigners.wignerpos(mpcl, xg, 0, 0)  # mask 2pcf
        ww0 = wigners.wignerpos(mpcl, 1., 0, 0)[0]
        zros = np.where(self._zro_criterion(ww, ww0))[0]
        if len(zros) > 0:
            xg_0 = xg[zros[-1]]
            print("zero ww at %.1f deg" % (np.arccos(xg_0) / np.pi * 180))
        xg, wg = wigners.get_xgwg(xg_0, 1., self.npts)
        ww = wigners.wignerpos(mpcl, xg, 0, 0)  # mask 2pcf
        assert np.all(ww > 0)

        Nb = len(self.edges) - 1

        xisp = np.zeros((Nb, self.npts))
        xisp_inv = np.zeros((Nb, self.npts))
        Nbs = np.zeros(Nb)
        bls, bus, bcse, bcsb = self._edges2blsbusbcs()
        Mab_p = np.zeros((Nb, Nb))
        Mab_m = np.zeros((Nb, Nb))

        for sgn in [1, -1]: # xi_{+} and xi_{-}
            for ia, (bl, bu) in utils.enumerate_progress(list(zip(bls, bus))):
                cl = np.zeros(bu + 1, dtype=float)
                cl[bl:bu + 1] = self.clg_templ[bl:bu + 1] + sgn * self.clc_templ[bl:bu + 1]
                xisp[ia] = wigners.wignerpos(cl, xg, self.spin, sgn * self.spin)
                cl[bl:bu + 1] = utils.cli(self.clg_templ[bl:bu + 1] + sgn *  self.clc_templ[bl:bu + 1])
                xisp_inv[ia] = wigners.wignerpos(cl, xg, self.spin, sgn * self.spin)
                Nbs[ia] = np.sum((2 * np.arange(bl, bu + 1) + 1)) / (4. * np.pi)
            Mab = Mab_p if sgn > 0 else Mab_m
            for ia in range(Nb):
                for ib in range(Nb):
                    Mab[ia, ib] = np.sum(xisp[ib] * xisp_inv[ia] * wg) / Nbs[ia]
        return Mab_p * (2 * np.pi), Mab_m * (2 * np.pi), xg_0, ww, xg, wg


    def _map2pcl(self, qumap1, qumap2=None):
        """Undeconvolved spectra amplitudes

                Returns gg + cc and gg - cc pseudo amplitudes

        """
        nside = hp.npix2nside(qumap1[0].size)
        Nb = len(self.edges) - 1
        s = self.spin
        lmax =min(3 * nside - 1, self.lmax)
        if qumap2 is None:
            elm, blm = hph.map2alm_spin([qumap1[0] * self.mask, qumap1[1] * self.mask], s, lmax=lmax, zbounds=self.zbounds)
            pcle = hp.alm2cl(elm)
            pclb = hp.alm2cl(blm)
            del elm, blm
        else:
            elm1, blm1 = hph.map2alm_spin([qumap1[0] * self.mask, qumap1[1] * self.mask], s, lmax=lmax, zbounds=self.zbounds)
            elm2, blm2 = hph.map2alm_spin([qumap2[0] * self.mask, qumap2[1] * self.mask], s, lmax=lmax, zbounds=self.zbounds)
            pcle = hp.alm2cl(elm1, alms2=elm2)
            pclb = hp.alm2cl(blm1, alms2=blm2)
            del elm1, elm2, blm1, blm2

        clp = wigners.wignercoeff(wigners.wignerpos(pcle + pclb, self.xg, s, +s) * self.wwi_wg, self.xg, s, +s, lmax=self.lmax)
        clm = wigners.wignercoeff(wigners.wignerpos(pcle - pclb, self.xg, s, -s) * self.wwi_wg, self.xg, s, -s, lmax=self.lmax)

        retp = np.zeros(Nb)
        retm = np.zeros(Nb)
        for ia, (bl, bu) in enumerate(zip(self.edges[:-1], self.edges[1:] - 1)):
            bu += (ia == Nb)
            l2p = 2 * np.arange(bl, bu + 1) + 1.
            retp[ia] = np.sum(clp[bl:bu + 1] * utils.cli(self.clg_templ[bl:bu + 1] + self.clc_templ[bl:bu + 1]) * l2p) / np.sum(l2p)
            retm[ia] = np.sum(clm[bl:bu + 1] * utils.cli(self.clg_templ[bl:bu + 1] - self.clc_templ[bl:bu + 1]) * l2p) / np.sum(l2p)
        return retp, retm

    def map2cl(self, qumap, qumap2=None):
        """Position space 2pcf deconvolver for spin-weight maps.

            Args:
                qumap : real and imag. part of the spin-weight map to take spectrum of.
                qumap2: second map, in case of a cross-spectrum

            Returns:
                Binned gradient and curl mode template amplitudes

        """
        Ap, Am = self._map2pcl(qumap, qumap2=qumap2)
        Ap = np.dot(self.Mpi, Ap)
        Am = np.dot(self.Mmi, Am)

        eef, bbf = self.get_fid_bandpowers()
        Ag = 0.5 * (Ap + Am) + 0.5 * bbf * utils.cli(eef) * (Ap - Am)
        Ac = 0.5 * (Ap + Am) + 0.5 * eef * utils.cli(bbf) * (Ap - Am)
        return Ag, Ac
