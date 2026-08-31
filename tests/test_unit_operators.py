"""
Unit tests: adjointness of MAP operators.

For each operator B we verify the dot-product identity
    <Bx, y> = <x, B†y>
using random Gaussian TEB alm vectors.

Tested operators (all in delensalot.core.MAP.operator):
    - Multiply          — scalar/complex multiplication
    - Beam              — harmonic transfer function (diagonal, self-adjoint)
    - InverseNoiseVariance  — isotropic harmonic weighting (self-adjoint)
    - Birefringence     — pixel-space polarization rotation
    - Lensing           — curved-sky remapping
    - Secondary         — compound Lensing ∘ Birefringence chain

Expected relative errors:
    - Algebraic ops (Multiply, Beam, InverseNoiseVariance): ~ machine precision (< 1e-12)
    - Birefringence (SHT round-trip):                       < 1e-8
    - Lensing (nonlinear remapping, epsilon=1e-12):         < 1e-6
    - Secondary (combined chain):                           < 1e-6

Run with:
    python -m unittest tests/test_unit_operators.py -v
"""

import unittest
import tempfile
import os
import numpy as np

from lenspyx.lensing import get_geom
from lenspyx.utils_hp import synalm

from delensalot.core.MAP import operator
from delensalot.utility.utils_hp import Alm, gauss_beam


# ── helpers ──────────────────────────────────────────────────────────────────

LMAX    = 1000   # large enough for SHT accuracy, small enough for speed
MMAX    = LMAX
LM_MAX  = (LMAX, MMAX)
SHT_TR  = 4      # number of SHT threads
RNG     = np.random.default_rng(42)


def rand_alms(lmax=LMAX, mmax=MMAX, n=3):
    """Return n random complex alm arrays of size Alm.getsize(lmax, mmax)."""
    size = Alm.getsize(lmax, mmax)
    return np.array([
        RNG.standard_normal(size) + 1j * RNG.standard_normal(size)
        for _ in range(n)
    ], dtype=complex)


def inner(x, y):
    """Hermitian inner product <x, y> = Re sum_{i,lm} x_i^{lm*} y_i^{lm}

    where m>0 modes are counted twice (standard alm convention).
    This is correct for testing <Bx, y> = <x, B†y> with complex operators.

    Note: hp.alm2cl(x,y) gives the real part of the *power spectrum* which
    includes a 1/(2l+1) normalisation — do NOT use it here.
    """
    result = 0.0
    for xi, yi in zip(x, y):
        # all m counted once
        dot = np.dot(xi.conj(), yi)
        # m>0 modes: add again (healpy ordering: m=0 block is first lmax+1 entries)
        lmax = Alm.getlmax(len(xi), None)
        dot += np.dot(xi[lmax + 1:].conj(), yi[lmax + 1:])
        result += dot.real
    return result


def rel_error(lhs, rhs):
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)


# Global results collector for the summary
_results = []

def record(name, err, tol, passed):
    _results.append((name, err, tol, passed))


def make_geomlib(lmax=LMAX):
    return get_geom(('thingauss', {'lmax': lmax + 300, 'smax': 3}))


def make_beam_desc(lmax=LMAX, beam_fwhm_amin=1.0):
    bl = gauss_beam(beam_fwhm_amin / 180 / 60 * np.pi, lmax=lmax)
    return {
        'data_key': 'p',
        'lm_max': (lmax, lmax),
        'transferfunction': {'t': bl, 'e': bl, 'b': bl},
    }


def make_lensing_op(tmpdir, lmax=LMAX, lm_max_in=LM_MAX, lm_max_out=LM_MAX):
    desc = {
        'libdir':      tmpdir,
        'data_key':    'p',
        'LM_max':      lm_max_in,
        'lm_max_in':   lm_max_in,
        'lm_max_out':  lm_max_out,
        'perturbative': False,
        'component':   ['p'],
        'sht_tr':      SHT_TR,
    }
    op = operator.Lensing(desc)
    # realistic lensing potential spectrum: Cl_phi ~ 1e-7 / (l(l+1))^2
    # giving RMS deflection ~1 amin — physically sensible and numerically stable
    ell = np.arange(lmax + 1, dtype=float)
    cl_phi = np.zeros(lmax + 1); cl_phi[1:] = 1e-7 / (ell[1:] * (ell[1:] + 1)) ** 2
    plm = synalm(cl_phi, lmax, lmax)
    op.set_field([plm])
    return op


def make_birefringence_op(tmpdir, lmax=LMAX):
    desc = {
        'libdir':      tmpdir,
        'LM_max':      LM_MAX,
        'lm_max':      LM_MAX,
        'component':   ['f'],
        'perturbative': False,
        'sht_tr':      SHT_TR,
    }
    op = operator.Birefringence(desc)
    # small birefringence angle field
    betam = synalm(1e-4 * np.ones(lmax + 1), lmax, lmax)
    op.set_field(betam)
    return op


# ── test cases ────────────────────────────────────────────────────────────────

class TestMultiplyAdjoint(unittest.TestCase):
    """Multiply: B†=B* (complex conjugate of scalar factor)."""

    def _check(self, factor, tol=1e-14):
        op = operator.Multiply({'factor': factor})
        x, y = rand_alms(), rand_alms()
        Bx  = op.act(x.copy())
        BTy = op.adjoint(y.copy())
        err = rel_error(inner(Bx, y), inner(x, BTy))
        record(f'Multiply adjoint (factor={factor})', err, tol, err < tol)
        self.assertLess(err, tol, f"factor={factor}: rel_error={err:.2e}")

    def test_real_factor(self):
        self._check(3.7)

    def test_complex_factor(self):
        self._check(2.0 + 1.5j)

    def test_unit_factor(self):
        self._check(1.0)


class TestBeamAdjoint(unittest.TestCase):
    """Beam: diagonal in harmonic space, self-adjoint for real transfer functions."""

    def test_adjoint(self):
        op = operator.Beam(make_beam_desc())
        errs = []
        for _ in range(3):
            x, y = rand_alms(), rand_alms()
            Bx  = op.act(x.copy(), adjoint=False)
            BTy = op.act(y.copy(), adjoint=True)
            err = rel_error(inner(Bx, y), inner(x, BTy))
            errs.append(err)
            self.assertLess(err, 1e-12, f"rel_error={err:.2e}")
        record('Beam adjoint', max(errs), 1e-12, max(errs) < 1e-12)

    def test_self_adjoint(self):
        """For real transfer function: Bx == B†x."""
        op = operator.Beam(make_beam_desc())
        x = rand_alms()
        Bx  = op.act(x.copy(), adjoint=False)
        BTx = op.act(x.copy(), adjoint=True)
        diff = np.max(np.abs(Bx - BTx))
        self.assertLess(diff, 1e-14, f"max|Bx - B†x|={diff:.2e}")


class TestInverseNoiseVarianceAdjoint(unittest.TestCase):
    """InverseNoiseVariance has two distinct code paths:

    ALM path (full sky, isotropic):
        input dtype complex → diagonal harmonic filter → output alm
        operator is square and self-adjoint: <N⁻¹x, y>_alm = <x, N⁻¹y>_alm

    MAP path (masked sky, anisotropic):
        input dtype float (pixel maps) → multiply by niv map → adjoint_synthesis → output alm
        operator maps (3, npix) → (3, nalm), so it is NOT square.
        The correct adjoint pairing is:
            <N⁻¹ x_map, y_alm>_alm = <x_map, (N⁻¹)† y_alm>_pixel
        where the pixel inner product is a flat dot product (apply_weights=False).
        (N⁻¹)† y_alm is: synthesis(y_alm) then multiply by niv — i.e. the transpose
        of [multiply niv → adjoint_synthesis].
    """

    def _make_op(self, tmpdir, geomlib=None):
        nlev = {'T': 2.0, 'P': np.sqrt(2) * 2.0}
        bl   = gauss_beam(1.0 / 180 / 60 * np.pi, lmax=LMAX)
        geomlib = geomlib or make_geomlib()
        return operator.InverseNoiseVariance(
            nlev             = nlev,
            lm_max           = LM_MAX,
            niv_desc         = {'t': np.array([1.0]), 'e': np.array([1.0]), 'b': np.array([1.0])},
            geom_lib         = geomlib,
            geominfo         = ('thingauss', {'lmax': LMAX + 300, 'smax': 3}),
            transferfunction = [bl, bl, bl],
            libdir           = tmpdir,
            sht_tr           = SHT_TR,
            filtering_type   = 'isotropic',
            data_key         = 'p',
        ), geomlib

    # ── ALM path (full sky, isotropic) ───────────────────────────────────────

    def test_alm_path_adjoint(self):
        """ALM path: self-adjoint diagonal harmonic filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op, _ = self._make_op(tmpdir)
            errs = []
            for _ in range(3):
                x, y = rand_alms(), rand_alms()
                Bx  = op.act(x.copy(), adjoint=False)
                BTy = op.act(y.copy(), adjoint=True)
                err = rel_error(inner(Bx, y), inner(x, BTy))
                errs.append(err)
                self.assertLess(err, 1e-12, f"alm path rel_error={err:.2e}")
            record('InvNoiseVar adjoint (alm path)', max(errs), 1e-12, max(errs) < 1e-12)

    def test_alm_path_self_adjoint(self):
        """ALM path: for real transfer function N⁻¹ = (N⁻¹)†."""
        with tempfile.TemporaryDirectory() as tmpdir:
            op, _ = self._make_op(tmpdir)
            x = rand_alms()
            Bx  = op.act(x.copy(), adjoint=False)
            BTx = op.act(x.copy(), adjoint=True)
            diff = np.max(np.abs(Bx - BTx))
            self.assertLess(diff, 1e-14, f"max|Bx - B†x|={diff:.2e}")

    # ── MAP path (masked sky, anisotropic) ───────────────────────────────────

    def test_map_path_adjoint(self):
        """MAP path: (3,npix) float → (3,nalm) complex.

        Forward:  x_map  →  niv * x_map  →  adjoint_synthesis  →  y_alm
        Adjoint:  y_alm  →  synthesis    →  niv * result        →  x_map

        Inner product pairing (apply_weights=False):
            <N⁻¹ x, y>_alm  ==  <x, (N⁻¹)† y>_pixel
        where pixel inner product is flat np.dot.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            op, geomlib = self._make_op(tmpdir)
            npix = geomlib.npix()

            for t in range(3):
                # x lives in pixel space (float), y lives in alm space (complex)
                x_map = np.array([
                    RNG.standard_normal(npix),  # T map
                    RNG.standard_normal(npix),  # Q map
                    RNG.standard_normal(npix),  # U map
                ], dtype=np.float64)
                y_alm = rand_alms()

                # Forward: map → alm
                Bx_alm = op.act(x_map.copy(), adjoint=False)

                # Adjoint: alm → map
                # (N⁻¹)†: synthesis then multiply by niv
                # T: spin-0 synthesis
                t_map = geomlib.synthesis(
                    np.atleast_2d(y_alm[0]), 0, LMAX, LMAX, SHT_TR)[0]
                t_map *= op.niv[0]
                # Q,U: spin-2 synthesis
                qu_map = geomlib.synthesis(
                    y_alm[1:], 2, LMAX, LMAX, SHT_TR)
                qu_map *= op.niv[1]
                BTy_map = np.array([t_map, qu_map[0], qu_map[1]])

                # <N⁻¹ x, y>_alm  (alm inner product)
                lhs = inner(Bx_alm, y_alm)
                # <x, (N⁻¹)† y>_pixel  (flat dot product, apply_weights=False)
                rhs = float(np.sum(x_map * BTy_map))

                err = rel_error(lhs, rhs)
                self.assertLess(err, 1e-12,
                    f"map path test {t}: rel_error={err:.2e}")
            record('InvNoiseVar adjoint (map path)', err, 1e-12, err < 1e-12)


class TestBirefringenceAdjoint(unittest.TestCase):
    """Birefringence F: pixel-space rotation.
    F†: rotation by -2β (i.e. F†∘F = I).
    """

    def test_adjoint(self, ntests=3, tol=1e-10):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = make_birefringence_op(tmpdir)
            errs = []
            for _ in range(ntests):
                x, y = rand_alms(), rand_alms()
                Bx  = op.act(x.copy(), adjoint=False)
                BTy = op.act(y.copy(), adjoint=True)
                err = rel_error(inner(Bx, y), inner(x, BTy))
                errs.append(err)
                self.assertLess(err, tol, f"test {_}: rel_error={err:.2e}")
            record('Birefringence adjoint', max(errs), tol, max(errs) < tol)

    def test_adjoint_implies_near_isometry(self, tol=1e-5):
        """If F is adjoint-consistent and the SHT is accurate, then
        <Fx, Fx> ≈ <x, F†Fx> — i.e. F approximately preserves norm.
        This is a weaker sanity check than exact unitarity.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            op = make_birefringence_op(tmpdir)
            x  = rand_alms()
            Fx   = op.act(x.copy(), adjoint=False)
            FTFx = op.act(Fx.copy(), adjoint=True)
            # <Fx, Fx> vs <x, F†Fx>  — if adjoint is correct these match
            lhs = inner(Fx, Fx)
            rhs = inner(x, FTFx)
            err = rel_error(lhs, rhs)
            record('Birefringence near-isometry', err, tol, err < tol)
            self.assertLess(err, tol,
                f"<Fx,Fx> != <x,F†Fx>: rel_error={err:.2e}")


class TestLensingAdjoint(unittest.TestCase):
    """Lensing D: curved-sky remapping.
    D† is the adjoint (backwards) remap.
    Tolerance is looser due to finite pixelization and interpolation.
    """

    def test_adjoint(self, ntests=3, tol=1e-10):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = make_lensing_op(tmpdir)
            errs = []
            for _ in range(ntests):
                x, y = rand_alms(), rand_alms()
                Bx  = op.act(x.copy(), adjoint=False, out_sht_mode='STANDARD')
                BTy = op.act(y.copy(), adjoint=True, backwards=True, out_sht_mode='STANDARD')
                err = rel_error(inner(Bx, y), inner(x, BTy))
                errs.append(err)
                self.assertLess(err, tol, f"test {_}: rel_error={err:.2e}")
            record('Lensing adjoint', max(errs), tol, max(errs) < tol)


class TestSecondaryAdjoint(unittest.TestCase):
    """Secondary: compound chain of Lensing ∘ Birefringence.
    Tests both orderings (lensing-first, birefringence-first).
    Adjoint reverses the order: (F∘D)† = D†∘F†.
    """

    def _make_secondary(self, tmpdir, order='lensing_first'):
        lens_op = make_lensing_op(tmpdir)
        bire_op = make_birefringence_op(tmpdir)
        if order == 'lensing_first':
            ops = [lens_op, bire_op]   # apply lensing then birefringence
        else:
            ops = [bire_op, lens_op]
        return operator.Secondary(ops)

    def _test_secondary(self, order, ntests=3, tol=1e-5):
        with tempfile.TemporaryDirectory() as tmpdir:
            op = self._make_secondary(tmpdir, order=order)
            for _ in range(ntests):
                x, y = rand_alms(), rand_alms()
                Bx  = op.act(x.copy(), adjoint=False, out_sht_mode='STANDARD')
                BTy = op.act(y.copy(), adjoint=True,  out_sht_mode='STANDARD')
                err = rel_error(inner(Bx, y), inner(x, BTy))
                self.assertLess(err, tol,
                    f"order={order} test {_}: rel_error={err:.2e}")

    def test_lensing_first(self):
        self._test_secondary('lensing_first')

    def test_birefringence_first(self):
        self._test_secondary('birefringence_first')


class TestGradientChainAdjoint(unittest.TestCase):
    """Test that the Secondary operator used in the gradient chain respects adjointness.

    The quadratic gradient in LensingGradientSub and BirefringenceGradientSub
    uses Secondary.act() with adjoint=True to form the gradient. The key
    property needed is:
        <Secondary x, y> = <x, Secondary† y>

    The full Compound(SpinRaise, Secondary) operator that maps alm->map is
    tested indirectly via the Secondary adjoint tests above, since SpinRaise
    is a fixed harmonic filter with no adjoint path (it modifies alms in-place
    and its adjoint is not implemented — see operator.py).
    """

    def test_secondary_adjoint_lensing_first(self, ntests=3, tol=1e-10):
        """Lensing ∘ Birefringence: adjoint reverses to Birefringence† ∘ Lensing†."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lens_op = make_lensing_op(tmpdir)
            bire_op = make_birefringence_op(tmpdir)
            sec_op  = operator.Secondary([lens_op, bire_op])
            errs = []
            for _ in range(ntests):
                x, y = rand_alms(), rand_alms()
                Bx  = sec_op.act(x.copy(), adjoint=False, out_sht_mode='STANDARD')
                BTy = sec_op.act(y.copy(), adjoint=True,  out_sht_mode='STANDARD')
                err = rel_error(inner(Bx, y), inner(x, BTy))
                errs.append(err)
                self.assertLess(err, tol, f"test {_}: rel_error={err:.2e}")
            record('Gradient chain: Lensing∘Birefringence adjoint', max(errs), tol, max(errs) < tol)

    def test_secondary_adjoint_birefringence_first(self, ntests=3, tol=1e-10):
        """Birefringence ∘ Lensing: adjoint reverses to Lensing† ∘ Birefringence†."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lens_op = make_lensing_op(tmpdir)
            bire_op = make_birefringence_op(tmpdir)
            sec_op  = operator.Secondary([bire_op, lens_op])
            errs = []
            for _ in range(ntests):
                x, y = rand_alms(), rand_alms()
                Bx  = sec_op.act(x.copy(), adjoint=False, out_sht_mode='STANDARD')
                BTy = sec_op.act(y.copy(), adjoint=True,  out_sht_mode='STANDARD')
                err = rel_error(inner(Bx, y), inner(x, BTy))
                errs.append(err)
                self.assertLess(err, tol, f"test {_}: rel_error={err:.2e}")
            record('Gradient chain: Birefringence∘Lensing adjoint', max(errs), tol, max(errs) < tol)


class TestSummary(unittest.TestCase):
    """Prints a summary table of all adjoint test results.
    Must run last — depends on _results populated by other tests.
    """

    def test_zzz_summary(self):
        """zzz prefix ensures this runs last in alphabetical test ordering."""
        GREEN = "[32m"
        RED   = "[31m"
        RESET = "[0m"
        BOLD  = "[1m"

        print(f"{BOLD}{'='*64}")
        print(f"  Adjoint test summary  (lmax={LMAX})")
        print(f"{'='*64}{RESET}")
        print(f"  {'Operator':<42} {'Rel. error':>12}  {'Tol':>10}  Status")
        print(f"  {'-'*62}")

        all_passed = True
        for name, err, tol, passed in _results:
            status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
            if not passed:
                all_passed = False
            print(f"  {name:<42} {err:>12.2e}  {tol:>10.2e}  {status}")

        print(f"  {'-'*62}")
        overall = f"{GREEN}All passed{RESET}" if all_passed else f"{RED}Some FAILED{RESET}"
        print(f"  Overall: {overall}")
        print(f"{BOLD}{'='*64}{RESET}")


if __name__ == '__main__':
    unittest.main()