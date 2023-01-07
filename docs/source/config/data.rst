====================================
data and simulations access
====================================

This section describes how to use simulation modules used for lensing reconstruction.
At the end, you will understand the structure of the data file, adapt it to your liking, and create your very own to read in your data.


D.lensalot uses a single T,E, and B map to perform lensing reconstruction.
These can be simulated maps, or real data.
They can come both, in spherical harmonics, or pixel space.
We assume the data to be in `HealPix`_-format.

.. _HealPix: https://healpy.readthedocs.io/en/latest/


Each class of a simulation module defines a simulation set. 
For each simulation set, there are class parameters, configuring the simulation set.

Each class must contain the following two :code:`functions` to work for lensing reconstruction: 

 * :code:`hashdict()`, return a dictionary describing how to identify the (intermediate) data stored on disk,
 * :code:`<get_sim_map>()`, returns the simulation map

All other functions are optional; vanilla D.lensalot will not use them.


minmal working example
-----------------------

The `minimal working example simulation module`_ generates, very naively, t,e, and b spherical harmonics from a FFP10 power spectrum, and at :code:`nside=2048` at runtime,
and adds a Gaussian noise to it.

.. _`minimal working example simulation module`: https://github.com/NextGenCMB/D.lensalot/blob/main/lenscarf/lenscarf/lerepi/config/examples/mwe/data_mwe/sims_mwe.py

The noise level can be set via the class parameter :code:`nlev`.


..  literalinclude:: /_static/sims_mwe.py
    :language: python
    :emphasize-lines: 6-14,
    :linenos:

To use the minimal working example (mwe) in a configuration file, the :code:`DLENSALOT_Data`-object has to be set accordingly,

.. code-block:: python
    
    DLENSALOT_Data(
        IMIN = 0,
        IMAX = 1,
        package_ = 'dlensalot',
        module_ = 'lerepi.config.examples.mwe.data_mwe.sims_mwe',
        class_ = 'mwe',
        class_parameters = {
            'nlev': '0.25'
        },
        data_type = 'alm',
        data_field = "eb",
        beam = 1,
        lmax_transf = 4000,
        nside = 2048
    )


A configuration file for the minimal working example can be found `here`_.

.. _`here`: https://github.com/NextGenCMB/D.lensalot/blob/main/lenscarf/lenscarf/lerepi/config/examples/mwe/conf_mwe.py


example FFP10
--------------

The `FFP10 simulation module`_ is more sophisticated.

D.lensalot comes with a `FFP10 simulation module`_, again generating simulations on the fly.
They are stored in a :code:`$temp` directory after generating them.

..  literalinclude:: /_static/sims_ffp10.py
    :language: python
    :linenos:


A :code:`DLENSALOT_Data`-object using this simulation module, generating 100 simulations, and reducing lmax to 2048, would look as follows,

.. code-block:: python
    
    DLENSALOT_Data(
        IMIN = 0,
        IMAX = 99,
        package_ = 'dlensalot',
        module_ = 'sims.sims_ffp10',
        class_ = 'cmb_len_ffp10',
        class_parameters = {
            'lmax_thingauss': 2048
        },
        data_type = 'alm',
        data_field = "eb",
    )


.. _`FFP10 simulation module`: https://github.com/NextGenCMB/D.lensalot/blob/main/lenscarf/sims/sims_ffp10.py



build your own
----------------


Your own simulation module can, in principle, lay anywhere.
To access it, set the :code:`root_path` parameter accordingly.
This will overwrite :code:`package_` and :code:`module_`. (?)
A complete list of :code:`DLENSALOT_Data` parameters are shown in the :ref:`Configuration Files` section.
