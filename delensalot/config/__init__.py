"""config.__init__.py: The config module handles the communication to the user, the configuration files, and the mapping between a configuration file to a valid delensalot model
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
    'secondaries',
    'libdir_suffix',
    'mfvar', #not really safe, but ok for now as long as you know what you do
    'operator_info',
]

DEFAULT_NotAValue = -123456789