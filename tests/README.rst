Tests
=====

This directory contains the automated tests for delensalot.
They are run automatically on every push and pull request via GitHub Actions
(see ``.github/workflows/``).


CI tests
--------

``test_integration_mwe.py``
    **Minimal working example — full pipeline smoke test.**
    Exercises sim generation, QE reconstruction, and MAP reconstruction
    end-to-end on a single simulation. Tuned for CI speed: full-sky
    polarization, ``lmax=2000``, ``nside=512``, high noise (3 μK-arcmin),
    2 MAP iterations, ``cg_tol=1e-3``. Passes if all steps complete without
    error and outputs are finite.

    Triggered by: ``.github/workflows/smoke_test.yaml``

    Run locally with::

        python -m unittest tests/test_integration_mwe.py -v


Old / development tests
-----------------------

Legacy and development tests have been moved to ``tests/old/``.
They are not run in CI and may require additional setup or be outdated.