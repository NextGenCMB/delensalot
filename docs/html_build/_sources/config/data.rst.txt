====================================
data and simulations access
====================================



This section describes how to use simulation modules used for lensing reconstruction.
At the end, you will understand the structure of the data file, adapt it to your liking, and create your very own to read in your data.

A complete list of :code:`DLENSALOT_Data` parameters are shown in the :ref:`Configuration Files` section.

delensalot uses a single T,E, and B map to perform lensing reconstruction.
They can come both, in spherical harmonics, or pixel space.
We assume the data to be in `HealPix`_-format.

.. _HealPix: https://healpy.readthedocs.io/en/latest/


Each class of a simulation module defines a simulation set. 
For each simulation set, class parameters configure the simulation set and delensalot provides ways to handle this.

Each class must contain the following two :code:`functions` to work for lensing reconstruction: 

 * :code:`hashdict()`, return a dictionary that describes the (intermediate) data stored on disk,
 * :code:`<get_sim_map>()`, returns the simulation map

All other functions are optional; vanilla delensalot will not use them.


minmal working example
-----------------------

The `minimal working example generate simulation`_ generates T,E, and B spherical harmonics from a FFP10 power spectrum using the delensalot-internal `simulation library`_, is lensed using `lenspyx`, and Gaussian noise is added. The full simulated polarization maps inclusive of the transfer function comes from `plancklens`_.

.. _`minimal working example generate simulation`: https://github.com/NextGenCMB/delensalot/blob/main/first_steps/notebooks/conf_mwe_simgen.py
.. _`simulation library`: https://github.com/NextGenCMB/delensalot/blob/main/delensalot/sims/generic.py
.. _`plancklens`: https://github.com/carronj/plancklens/blob/master/plancklens/sims/maps.py

The noise level can be set via the parameter :code:`nlev`.

..  literalinclude:: ../_static/conf_mwe_simgen.py
    :language: python
    :emphasize-lines: 34-35
    :linenos:

build your own
----------------

To use your own data or simulations, you will have to write a class that provides the function :code:`get_sim_pmap()` and point delensalot to it.
Your own simulation module can, in principle, be put anywhere.