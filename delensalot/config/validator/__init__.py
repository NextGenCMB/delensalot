""" validator.__init__.py: The validator module collects functions which are used when validating the delensalot configuration file.
The init contains a helper attribute and is accessed when two configuration files are compared against one another.
"""

safelist = [
    'version',
    'key',
    'jobs',
    'simidxs',
    'simidxs_mf',
    'tasks',
    'iterations',
    'cl_analysis',
    'blt_pert',
    'itmax',
    'cg_tol',
    'mfvar',
    'dlm_mod',
    'spectrum_calculator',
    'binning',
    'outdir_plot_root',
    'outdir_plot_rel',
    'OMP_NUM_THREADS',
    'rhits_normalised',
    'masks_fn',
    'secondaries'
]

DEFAULT_NotAValue = -123456789