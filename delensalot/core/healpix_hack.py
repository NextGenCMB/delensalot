import ctypes
import numpy as np
#/global/homes/j/jcarron/cori/cfitsio/libcfitsio.so
#PATH2GOMP = '/usr/local/lib/libgomp.1.dylib'
#/usr/lib64/gcc/x86_64-suse-linux/4.8/libgomp.so
#PATH2HEALPIX = '/Users/jcarron/PycharmProjects/healpix-code-r974-trunk/lib/libhealpix.dylib'
#/global/homes/j/jcarron/cori/healpix-code-r974-trunk/lib/libhealpix.so

PATH2CFITSIO = '/usr/common/software/cfitsio/3.47/lib/libcfitsio.so'
PATH2GOMP = '/usr/lib64/gcc/x86_64-suse-linux/7/libgomp.so'
PATH2HEALPIX = '/global/homes/j/jcarron/cori/healpix-code-r974-trunk/lib/libhealpix.so'


def get_libhealpix():
    """ Loads healpix f90 shared library. For some reason I need to load gomp and cfitsio into the globals first. """
    ctypes.CDLL(PATH2GOMP, ctypes.RTLD_GLOBAL)
    ctypes.CDLL(PATH2CFITSIO, ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(PATH2HEALPIX)


try:
    libhealpix = get_libhealpix()
    HACKOK = True
except:
    print('could not setup healpix hack. reverting to standard healpy')
    HACKOK = False

if HACKOK:
    def map2alm(tmap, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        """ Hacked from healpix-code-r974-trunk f90 shared library compilation 24 Oct 2018.
        This wraps the double precision version of (see alm_map_template.F90) as calculated with libsharp.
        !=======================================================================
        subroutine map2alm_sc_d(nsmax, nlmax, nmmax, map, alm, zbounds, w8)
        !=======================================================================
        Set zbounds for calculation on a fraction of the rings.
        """
        nside = int(np.sqrt(tmap.size // 12))
        assert 12 * nside ** 2 == tmap.size
        if lmax is None : lmax = 3 * nside -1
        if mmax is None : mmax = lmax

        _map2alm = getattr(libhealpix, 'sharp_hp_map2alm_x_d_')
        _map2alm.restype = None
        _map2alm.argtypes = [ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous', shape=(12 * nside ** 2,)),
                np.ctypeslib.ndpointer(np.complex128, ndim=3, flags='aligned, f_contiguous, writeable',
                                       shape=(1, lmax+1, mmax + 1)),
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous', shape=(2,)),
                np.ctypeslib.ndpointer(np.float64, ndim=2, flags='aligned, f_contiguous', shape=(2*nside - 1, 2))]
        _nsmax = ctypes.byref(ctypes.c_int(nside))
        _nlmax = ctypes.byref(ctypes.c_int(lmax))
        _nmmax = ctypes.byref(ctypes.c_int(mmax))

        _tmap = np.require(tmap, np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _alm = np.require(np.zeros((1, lmax + 1, lmax + 1), dtype=np.complex128), np.complex128, ['ALIGNED', 'F_CONTIGUOUS', 'WRITEABLE'])
        _w8 = np.require(np.ones( (2 * nside -1, 2), dtype=np.float64), np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _zbounds =  np.require(zbounds,np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _map2alm(_nsmax, _nlmax, _nmmax, _tmap, _alm, _zbounds, _w8)
        alm = np.zeros((lmax + 2) * (lmax + 1) // 2 , dtype=complex)
        for m in range(lmax + 1):
            alm[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] = _alm[0, m:,m]
        return alm

    def alm2map(alm, nside, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        """ Hacked from healpix-code-r974-trunk f90 shared library compilation 24 Oct 2018.
        This wraps the double precision version of (see alm_map_template.F90) as calculated with libsharp.
        !=======================================================================
        subroutine alm2map_sc_d(nsmax, nlmax, nmmax, map, alm, zbounds, w8)
        !=======================================================================
        Set zbounds for calculation on a fraction of the rings.
        """
        if lmax is None : lmax = int(np.floor(np.sqrt(2 * alm.size) - 1))
        if mmax is None : mmax = lmax
        assert (lmax + 2) * (lmax + 1) // 2 == alm.size

        _alm2map = getattr(libhealpix, 'sharp_hp_alm2map_x_d_')
        _alm2map.restype = None
        _alm2map.argtypes = [ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                np.ctypeslib.ndpointer(np.complex128, ndim=3, flags='aligned, f_contiguous', shape=(1, lmax + 1, mmax + 1)),
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous, writeable', shape=(12 * nside ** 2,)),
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous', shape=(2,)),]
        _nsmax = ctypes.byref(ctypes.c_int(nside))
        _nlmax = ctypes.byref(ctypes.c_int(lmax))
        _nmmax = ctypes.byref(ctypes.c_int(mmax))

        _tmap = np.require(np.zeros(12 * nside ** 2, dtype=np.float64, order='F'), np.float64, ['ALIGNED', 'F_CONTIGUOUS', 'WRITEABLE'])
        alm_sharp = np.zeros((1, lmax + 1, mmax + 1), dtype=np.complex128, order='F')
        for m in range(lmax + 1):
            alm_sharp[0, m:, m] = alm[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)]
        _alm_sharp = np.require(alm_sharp, np.complex128, ['ALIGNED', 'F_CONTIGUOUS'])
        _zbounds =  np.require(zbounds,np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _alm2map(_nsmax, _nlmax, _nmmax, _alm_sharp, _tmap, _zbounds)
        return _tmap

    def alm2map_spin(alms, nside, spin, lmax, mmax=None, zbounds=np.array([-1., 1.])):
        """ Hacked from healpix-code-r974-trunk f90 shared library compilation 24 Oct 2018.
        This wraps the double precision version of (see alm_map_template.F90) as calculated with libsharp.
        !=======================================================================
        subroutine alm2map_spin_d(nsmax, nlmax, nmmax, spin, alm, map, zbounds)
        !=======================================================================
        Set zbounds for calculation on a fraction of the rings.
        """
        assert spin > 0, spin
        assert (lmax + 2) * (lmax + 1) // 2 == alms[0].size
        assert (lmax + 2) * (lmax + 1) // 2 == alms[1].size
        if mmax is None: mmax = lmax
        _alm2map_spin = getattr(libhealpix, 'sharp_hp_alm2map_spin_x_d_')
        _alm2map_spin.restype = None
        _alm2map_spin.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            np.ctypeslib.ndpointer(np.complex128, ndim=3, flags='aligned, f_contiguous', shape=(2, lmax + 1, mmax + 1)),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='aligned, f_contiguous, writeable', shape=(12 * nside ** 2, 2)),
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous', shape=(2,)), ]

        maps = np.zeros((12 * nside ** 2, 2), dtype=np.float64, order='F')
        _nside = ctypes.byref(ctypes.c_int(nside))
        _nlmax = ctypes.byref(ctypes.c_int(lmax))
        _nmmax = ctypes.byref(ctypes.c_int(mmax))
        _spin = ctypes.byref(ctypes.c_int(spin))

        alm_sharp = np.zeros((2, lmax + 1, mmax + 1), dtype=np.complex128, order='F')
        for m in range(lmax + 1):
            alm_sharp[0, m:, m] = alms[0][((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)]
            alm_sharp[1, m:, m] = alms[1][((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)]
        _maps = np.require(maps, np.float64, ['ALIGNED', 'F_CONTIGUOUS', 'WRITEABLE'])
        _alm = np.require(alm_sharp, np.complex128, ['ALIGNED', 'F_CONTIGUOUS'])
        _zbounds = np.require(zbounds, np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _alm2map_spin(_nside, _nlmax, _nmmax, _spin, _alm, _maps, _zbounds)
        return _maps[:, 0], _maps[:, 1]


    def map2alm_spin(maps, spin, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        """ Hacked from healpix-code-r974-trunk f90 shared library compilation 24 Oct 2018.
        This wraps the double precision version of (see alm_map_template.F90) as calculated with libsharp
        !=======================================================================
        subroutine map2alm_spin_d(nsmax, nlmax, nmmax, spin, map, alm, zbounds, w8ring)
        !=======================================================================
        Set zbounds for calculation on a fraction of the rings.
        """
        assert spin > 0, spin
        assert len(maps) == 2
        nside = int(np.sqrt(maps[0].size // 12))
        assert 12 * nside ** 2 == maps[0].size
        assert 12 * nside ** 2 == maps[1].size
        if lmax is None: lmax = 3 * nside -1
        if mmax is None: mmax = lmax
        _map2alm_spin = getattr(libhealpix, 'sharp_hp_map2alm_spin_x_d_')
        _map2alm_spin.restype = None
        _map2alm_spin.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='aligned, F_contiguous',  shape=(12 * nside ** 2, 2)),
            np.ctypeslib.ndpointer(np.complex128, ndim=3, flags='aligned, f_contiguous, writeable', shape=(2, lmax + 1, mmax + 1)),
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='aligned, f_contiguous', shape=(2,)),
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='aligned, f_contiguous', shape=(2 * nside - 1, 2))]

        alm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.complex128, order='F')
        w8 = np.ones((2 * nside - 1, 2), dtype=np.float64, order='F')

        _nside = ctypes.byref(ctypes.c_int(nside))
        _nlmax = ctypes.byref(ctypes.c_int(lmax))
        _nmmax = ctypes.byref(ctypes.c_int(mmax))
        _spin = ctypes.byref(ctypes.c_int(spin))

        _maps = np.require(np.array(maps).transpose(), np.float64, ['ALIGNED', 'F_CONTIGUOUS', 'WRITEABLE'])
        _alm = np.require(alm, np.complex128, ['ALIGNED', 'F_CONTIGUOUS', 'WRITEABLE'])
        _zbounds = np.require(zbounds, np.float64, ['ALIGNED', 'F_CONTIGUOUS'])
        _w8 = np.require(w8, np.float64, ['ALIGNED', 'F_CONTIGUOUS'])

        _map2alm_spin(_nside, _nlmax, _nmmax, _spin, _maps, _alm, _zbounds, _w8)
        ret = np.zeros((2, (lmax + 1) * (lmax + 2) // 2), dtype=complex)
        for m in range(lmax + 1):
            ret[:, ((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] = _alm[:, m:, m]
        return ret

else:
    import healpy as hp
    def map2alm(tmap, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        return hp.map2alm(tmap, lmax=lmax, mmax=mmax, iter=0)
    def alm2map(alm, nside, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        return hp.alm2map(alm, nside, lmax=lmax, mmax=mmax, verbose=False)
    def map2alm_spin(maps, spin, lmax=None, mmax=None, zbounds=np.array([-1., 1.])):
        return hp.map2alm_spin(maps, spin, lmax=lmax, mmax=mmax)
    def alm2map_spin(alms, nside, spin, lmax, mmax=None, zbounds=np.array([-1., 1.])):
        return hp.alm2map_spin(alms, nside, spin, lmax, mmax=mmax)

def _test(lmax, nside, spin):
    rtol=1e-10
    from lenspyx.utils import timer
    import healpy as hp
    glm = hp.synalm(np.ones(lmax + 1), new=True)
    clm = 2 * hp.synalm(np.ones(lmax + 1), new=True)
    print("*** alm2map_spin")
    times = timer(True)
    ds_hp, dms_hp = hp.alm2map_spin([glm, clm], nside, spin, lmax, mmax=None)
    times.add('hp      alm2map_spin')
    ds_ha, dms_ha = alm2map_spin([glm, clm], nside, spin, lmax, mmax=None)
    times.add('hp hack alm2map_spin')
    print(np.allclose(ds_hp, ds_ha, rtol=rtol))
    print(np.allclose(dms_hp, dms_ha, rtol=rtol))
    del ds_ha, dms_ha
    print("*** map2alm_spin")
    times.reset()
    glm_hp, clm_hp = hp.map2alm_spin([ds_hp, dms_hp], spin, lmax=lmax)
    times.add('hp      map2alm_spin')
    glm_ha, clm_ha = map2alm_spin([ds_hp, dms_hp], spin, lmax=lmax)
    times.add('hp hack map2alm_spin')
    print(np.allclose(glm_hp, glm_ha, rtol=rtol))
    print(np.allclose(clm_hp, clm_ha, rtol=rtol))
    del clm_ha, clm_hp, glm_hp
    print("*** alm2map")
    times.reset()
    ds_hp = hp.alm2map(glm_ha, nside)
    times.add('hp      alm2map')
    ds_ha = alm2map(glm_ha, nside)
    times.add('hp hack alm2map')
    print(np.allclose(ds_hp, ds_ha, rtol=rtol))
    print("*** map2alm")
    times.reset()
    glm_hp = hp.map2alm(ds_hp, lmax=lmax, iter=0)
    times.add('hp      map2alm')
    glm_ha = map2alm(ds_hp, lmax=lmax)
    times.add('hp hack map2alm')
    print(np.allclose(glm_hp, glm_ha, rtol=rtol))
    print(times)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test healpy hack')
    parser.add_argument('-nside', dest='nside', type=int, default=512, help='healpy reso.')
    parser.add_argument('-lmax', dest='lmax', type=int, default=512)
    parser.add_argument('-s', dest='s', type=int, default=0, help='spin')
    args = parser.parse_args()
    try:
        get_libhealpix()
        HACKOK = True
    except:
        print('could not setup healpix hack. reverting to standard healpy')
        HACKOK = False

    if HACKOK:
        _test(args.lmax, args.nside, args.s)
    else:
        print("Could not load healpix-hack")
