import numpy as np
from lenscarf.remapping import deflection
from lenscarf.remapping import d2ang
from lenscarf.utils_scarf import Geom
from lenscarf import cachers
import healpy as hp
import ducc0



def blm_gauss(fwhm, lmax, spin:int):
    """Computes spherical harmonic coefficients of a circular Gaussian beam
    pointing towards the North Pole

    See an example of usage
    `in the documentation <https://healpy.readthedocs.io/en/latest/blm_gauss_plot.html>`_

    Parameters
    ----------
    fwhm : float, scalar
        desired FWHM of the beam, in radians
    lmax : int, scalar
        maximum l multipole moment to compute
    spin : bool, scalar
        if True, E and B coefficients will also be computed

    Returns
    -------
    blm : array with dtype numpy.complex128
          lmax will be as specified
          mmax is equal to spin
    """
    fwhm = float(fwhm)
    lmax = int(lmax)
    mmax = spin
    ncomp = 2 if spin > 0 else 1
    nval = hp.Alm.getsize(lmax, mmax)

    if mmax > lmax:
        raise ValueError("lmax value too small")

    blm = np.zeros((ncomp, nval), dtype=np.complex128)
    sigmasq = fwhm * fwhm / (8 * np.log(2.0))

    if spin == 0:
        for l in range(0, lmax + 1):
            blm[0, hp.Alm.getidx(lmax, l, spin)] = np.sqrt((2 * l + 1) / (4.0 * np.pi)) * np.exp(-0.5 * sigmasq * l * l)

    if spin > 0:
        for l in range(spin, lmax + 1):
            blm[0, hp.Alm.getidx(lmax, l, spin)] = np.sqrt((2 * l + 1) / (32 * np.pi)) * np.exp(-0.5 * sigmasq * l * l)
        blm[1] = 1j * blm[0]

    return blm

class ducc_deflection(deflection):
    """spin-0 lensing interpolator based on Martin Reinecke totalconvolve package

    """
    def __init__(self, *args, epsilon=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result
        self.ofactor = 1.5  # upsampling grid factor
        print(" DUCC totalconvolve deflection instantiated", self.epsilon, self.ofactor)

        # TODO: _fwd_angles should be done better
        # TODO: no need to have the same constructor, all inputs are irrelevant

    def _get_ptg(self):
        # TODO improve this and fwd angles
        return self._build_angles() # -gamma in third argument

    def _fwd_angles(self, fortran=False):
        assert 0, 'why are you here ?'

    def _fwd_polrot(self):
        return self._build_angles()[:, -1]

    def _build_angles(self):
        """Builds deflected positions and angles


        """
        fn = 'ptg'
        if not self.cacher.is_cached(fn):
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            thp_phip_mgamma= np.empty( (3, npix), dtype=float) # (-1) gamma in last arguement
            startpix = 0
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if pixs.size > 0:
                    t_red = red[pixs]
                    i_imd = imd[pixs]
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.get_theta(ir) * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(t_red, i_imd, thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip_mgamma[0, sli] = thtp_
                    thp_phip_mgamma[1, sli] = phip_
                    assert 0 < self.geom.theta[ir] < np.pi, 'Fix this'
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_mgamma[2, sli] -= np.arctan2(i_imd, t_red )
                    thp_phip_mgamma[2, sli] += np.arctan2(i_imd, d * np.sin(d) * cot + t_red  * np.cos(d))
                    startpix += len(pixs)
            thp_phip_mgamma = thp_phip_mgamma.transpose()
            self.cacher.cache(fn, thp_phip_mgamma)
            assert startpix == npix, (startpix, npix)
            return thp_phip_mgamma
        return self.cacher.load(fn)

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return ducc_deflection(self.pbgeom, self._resamin, dlm[0], mmax_dlm, self._fft_tr, self.sht_tr, cacher, dlm[1],
                          verbose=self.verbose, epsilon=self.epsilon)


    def gclm2lenmap(self, gclm:np.ndarray or list, mmax:int or None, spin, backwards:bool, nomagn=False, polrot=True):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        if abs(spin) == 0:
            lmax_unl = hp.Alm.getlmax(gclm.size, mmax)
            blm_T = hp.sphtfunc.blm_gauss(0, lmax=lmax_unl, pol=False)
            ptg = self._get_ptg()
            inter_I = ducc0.totalconvolve.Interpolator(np.atleast_2d(gclm), blm_T, separate=False, lmax=lmax_unl, kmax=0,
                                                           epsilon=self.epsilon, ofactor=self.ofactor, nthreads=self.sht_tr)
            return inter_I.interpol(ptg).squeeze()
        else:
            lmax_unl = hp.Alm.getlmax(gclm[0].size, mmax)
            blm_GC = blm_gauss(0, lmax_unl, spin)
            inter_QU = ducc0.totalconvolve.Interpolator(
                gclm, blm_GC, separate=False, lmax=lmax_unl, kmax=abs(spin), epsilon=self.epsilon, nthreads=self.sht_tr)
            ptg = self._get_ptg()
            assert polrot
            Q = -np.sqrt(2) * inter_QU.interpol(ptg).squeeze()
            ptg[:, 2] += np.pi / (2 * spin)
            U = -np.sqrt(2) * inter_QU.interpol(ptg).squeeze()
            ptg[:, 2] -= np.pi / (2 * spin) #otherwise modifies the cached gamma
            return np.array([Q, U])

    def lensgclm(self, gclm:np.ndarray or list, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, nomagn=False, polrot=True):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

        """
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards, nomagn=nomagn)
            if spin == 0:
                return self.geom.map2alm(m, lmax_out, mmax_out, self.sht_tr)
            else:
                return self.geom.map2alm_spin(m, spin, lmax_out, mmax_out, self.sht_tr, polrot=polrot)
        else:
            lmax_unl = hp.Alm.getlmax(gclm[0].size if abs(spin) > 0 else gclm.size, mmax)
            inter = ducc0.totalconvolve.Interpolator(lmax_out, spin, 1, epsilon=self.epsilon,
                                                     ofactor=self.ofactor, nthreads=self.sht_tr)
            if spin == 0:
                xptg = self._get_ptg()
                I = self.geom.alm2map(gclm, lmax_unl, mmax, self.sht_tr)
                for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                    I[int(ofs):int(ofs + nph)] *= w
                inter.deinterpol(xptg, np.atleast_2d(I))
            else:
                Red, Imd = self.geom.alm2map_spin(gclm, spin, lmax_unl, mmax, self.sht_tr)
                for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                    Red[int(ofs):int(ofs + nph)] *= w
                    Imd[int(ofs):int(ofs + nph)] *= w
                assert polrot
                #if polrot: now done with psi angle
                #    # This can be done by setting psi to 2 pi - gamma
                #    gamma = self._bwd_polrot if backwards else self._fwd_polrot
                #    d = (Red + 1j * Imd) * np.exp( 1j * spin * gamma())
                #   Red = d.real
                #   Imd = d.imag
                xptg = self._get_ptg()
                inter.deinterpol(xptg, -np.sqrt(2) * np.atleast_2d(Red))
                xptg[:, 2] += np.pi / (spin * 2)
                inter.deinterpol(xptg, -np.sqrt(2) * np.atleast_2d(Imd))
                xptg[:, 2] -= np.pi / (spin * 2) # to avoid modifying the object which will be reused
            blm = blm_gauss(0, lmax_out, spin)
            return inter.getSlm(blm).squeeze()