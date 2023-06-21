"""Module to allow general simulations that include cmb+foregrounds+other at the signal level.
"""

from plancklens.sims import maps

import healpy as hp


class cmb_maps(maps.cmb_maps):

    """
    Class to handle multiple objects that give a field at the CMB signal level.
    """

    def __init__(self, **kwargs):
        """
        Initializes the cmb_maps object. Note, for now you have to initialize it with a sims_cmb_len object.
        """
        super(cmb_maps, self).__init__(**kwargs)
        self.components = []
        self.lmax = self.sims_cmb_len.lmax


    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map
        """
        tmap = self.get_sim_tlm(idx)
        hp.almxfl(tmap,self.cl_transf,inplace=True)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_tlm(self, idx):
        """
        Returns the temperature alm of the sum of the components.

        Args:
            idx: simulation index
        
        Returns:    
            alm of the sum of the components.
        """
        return sum([self.check(c.get_sim_tlm(idx)) for c in self.components])

    def check(self, alms):
        """
        Just a simple check to make sure the alms have the same lmax as the one initialized the object.

        Args:
            alms: alms to check
        
        Returns:
            alms
        """
        assert hp.Alm.getlmax(alms.size) == self.lmax, "The alms you are trying to add have a different lmax than the one you initialized the object with!"
        return alms
    
    def __add__(self, other):
        """
        Adds a component to the cmb_maps object.
        """
        return self._update(other)

    def _update(self, other):
        """
        Adds a component to the cmb_maps object.
        """
        assert callable(other.get_sim_tlm), "The object you are trying to add has to have a get_sim_tlm method!"
        self.components.append(other)
        return self