====================================
data and simulations access
====================================

D.lensalot uses a single T,E, and B map to perform lensing reconstruction.
These could be plain simulated and combined frequency maps, real data, or component separated maps.

The instructions to read the data are provided by a python 'data file', here an example of a `FFP10 data file`_,
and its structure is as follows.
A :code:`class` which contains the paths to the maps.
The class must contain the following two :code:`functions` to work for lensing reconstruction: 

 * :code:`hashdict()`, return a dictionary describing how to identify the (intermediate) data stored on disk,
 * :code:`<get_sim_map>()`, returns the simulation

..
    Depending on the analysis, :code:`<get_sim_map>()` can be one of the following,
    :code:`get_sim_pmap()`, :code:`get_sim_tmap()`, :code:`get_sim_tebmap()`

.. _`FFP10 data file`: https://github.com/NextGenCMB/D.lensalot/blob/main/lenscarf/sims/sims_ffp10.py


For 