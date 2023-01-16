.. _Configuration Files:

========================
configuration files
========================

All analysis details can be put into a structured, human-readable metamodel.
D.lensalot interprets this metamodel and executes the chosen Jobs with the chosen configuration.

Some `example`_ models are provided for different experiments and may be found in the subdirectories therein.


.. _example: https://github.com/NextGenCMB/D.lensalot/tree/main/lenscarf/lerepi/config



Structure:
--------------------

A complete Dlensalot model includes,

==================== ===========
        Type         Description
-------------------- -----------
    Job              Which jobs to run
    Analysis         Analysis specific settings
    Data             Which data to use
    Noisemodel       A noisemodel to define the Wiener-filter
    Qerec            Quadratic estimator lensing reconstruction specific settings
    Itrec            Iterative delensing lensing reconstruction specific settings
    Mapdelensing     Delensing specific settings
    Config           General configuration
==================== ===========



* There is a predefined list of Jobs from which the user can choose, which are chosen via ``<boolean choice>`` = ``True`` or ``False``,
    * build_OBD = <boolean choice>,
    * QE_lensrec = <boolean choice>,
    * MAP_lensrec = <boolean choice>,
    * map_delensing = <boolean choice>,
    * inspect_result = <boolean choice>,


The following is an example Dlensalot model for CMBS-4 configurations, for which iterative lensing reconstruction and map delensing is chosen as Job.

.. code-block:: python

    import numpy as np
    from lenscarf.lerepi.core.metamodel.dlensalot_v2 import *
    from MSC import pospace
    import os
    from os.path import join as opj

    dlensalot_model = DLENSALOT_Model(
        job = DLENSALOT_Job(
            build_OBD = False,
            QE_lensrec = False,
            MAP_lensrec = True,
            map_delensing = True,
            inspect_result = False,
            OMP_NUM_THREADS = 16
        ),
        analysis = DLENSALOT_Analysis(
            TEMP_suffix = 'QErun',
            K = 'p_p',
            V = '',
            ITMAX = 0,
            simidxs_mf = np.arange(0,200),
            zbounds =  ('nmr_relative', 100),
            zbounds_len = ('extend', 5.),   
            pbounds = [1.97, 5.71],
            LENSRES = 1.7,
            Lmin = 2, 
            lmax_filt = 4000,
            lmax_unl = 4000,
            mmax_unl = 4000,
            lmax_ivf = 3000,
            mmax_ivf = 3000,
            lmin_ivf = 10,
            mmin_ivf = 10,
            STANDARD_TRANSFERFUNCTION = True
        ),
        data = DLENSALOT_Data(
            IMIN = 0,
            IMAX = 499,
            package_ = 'lenscarf',
            module_ = 'lerepi.config.cmbs4.data.data_08b',
            class_ = 'caterinaILC_May12',
            class_parameters = {
                'fg': '00'
            },
            data_type = 'map',
            data_field = "qu",
            beam = 2.3,
            lmax_transf = 4000,
            nside = 2048
        ),
        noisemodel = DLENSALOT_Noisemodel(
            typ = 'OBD',
            BMARG_LIBDIR = opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/'),
            BMARG_LCUT = 200,
            BMARG_RESCALE = (0.42/0.350500)**2,
            ninvjob_geometry = 'healpix_geometry',
            lmin_tlm = 30,
            lmin_elm = 30,
            lmin_blm = 30,
            CENTRALNLEV_UKAMIN = 0.42,
            nlev_t = 0.42/np.sqrt(2),
            nlev_p = 0.42,
            nlev_dep = 10000.,
            inf = 1e4,
            mask = ('nlev', np.inf),
            rhits_normalised = (opj(os.environ['CFS'], 'cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits'), np.inf),
            tpl = 'template_dense'
        ),
        qerec = DLENSALOT_Qerec(
            ivfs = 'sepTP', # Change only if other than sepTP for QE is desired
            qlms = 'sepTP',
            cg_tol = 1e-3,
            tasks = ["calc_phi", "calc_meanfield", "calc_blt"],
            ninvjob_qe_geometry = 'healpix_geometry_qe',
            lmax_qlm = 4000,
            mmax_qlm = 4000,
            QE_LENSING_CL_ANALYSIS = False, # Change only if a full, Planck-like QE lensing power spectrum analysis is desired
            chain = DLENSALOT_Chaindescriptor(
                p0 = 0,
                p1 = ["diag_cl"],
                p2 = None,
                p3 = 2048,
                p4 = np.inf,
                p5 = None,
                p6 = 'tr_cg',
                p7 = 'cache_mem'
            )
        ),
        itrec = DLENSALOT_Itrec(
            filter = 'opfilt_ee_wl.alm_filter_ninv_wl',
            cg_tol = 1e-4,
            tasks = ["calc_phi", "calc_meanfield", "calc_btemplate"], #["calc_phi", "calc_meanfield", "calc_btemplate"],
            lenjob_geometry = 'thin_gauss',
            lenjob_pbgeometry = 'pbdGeometry',
            iterator_typ = 'constmf', # Either pertmf or const_mf
            mfvar = '',
            soltn_cond = lambda it: True,
            stepper = DLENSALOT_Stepper(
                typ = 'harmonicbump',
                xa = 400,
                xb = 1500
            ),
            btemplate_perturbative_lensremap = True
        ),
        madel = DLENSALOT_Mapdelensing(
            iterations = [],
            edges = ['cmbs4', 'ioreco'], # overwritten when binning=unbinned
            masks = ("nlevels", [1.2, 2, 10, 50]),
            lmax = 2048, # automatically set to 200 when binning=unbinned
            Cl_fid = 'ffp10',
            binning = 'binned',
            spectrum_calculator = pospace,
            btemplate_perturbative_lensremap = True,
            data_from_CFS = False
        ),
        config = DLENSALOT_Config(
            outdir_plot_root = opj(os.environ['HOME'], 'plots'),
            outdir_plot_rel = "cmbs4/08b/"
        )
    )
