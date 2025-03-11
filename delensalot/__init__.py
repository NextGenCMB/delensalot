from pathlib import Path
import os
from os.path import join as opj

os.environ['USE_PLANCKLENS_MPI'] = ""

package_root = Path(__file__).resolve().parent.parent

if 'SCRATCH' not in os.environ:
    if "site-packages" in str(package_root):
        # If installed in site-packages (because of pip install . , e.g.), use a fallback location
        scratch_path = Path.home() / "delensalot_temp"
    else:
        # Otherwise, use the package directory
        scratch_path = package_root / "delensalot_temp"

    os.environ["SCRATCH"] = str(scratch_path.resolve())
    print(f"INFO: Setting SCRATCH to {os.environ['SCRATCH']}")