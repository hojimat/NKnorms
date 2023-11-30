from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np
from numpy.typing import NDArray
import nkpack as nk

class Landscape:
    """
    This class handles the problem space (task environment),
    ans serves as a performance lookup center.
    It relates to an Organization class in a one-to-one manner.

    """

    def __init__(self, p:int, n:int, k:int, c:int, s:int, rho:float, normalize:bool, precompute:bool):
        # user input
        self.p = p # population / number of agents
        self.n = n # number of tasks per agent
        self.k = k # number of internally coupled bits
        self.c = c # number of externally coupled bits
        self.s = s # number of externally coupled species (peers)
        self.rho = rho # correlation coefficient among individual landscapes
        self.normalize = normalize # normalize or not by the global maximum; CPU-heavy 
        self.precompute = precompute # normalize or not by the global maximum; CPU-heavy 
        # generated
        self.globalmax : float = 1.0
        self.interaction_matrix : NDArray[np.int8] = None
        self.landscape : NDArray[np.float32] = None
        self.performance_lookup: NDArray[np.float32] = None # Px2^(n*p) lookup table of bstring perfs


    def generate(self):
        """Generates the problem space (task environment)"""
        
        self._generate_interaction_matrix()
        self._generate_landscape()

        if self.normalize and self.precompute:
            self._calculate_all_performances()
        elif self.normalize and not self.precompute:
            self._calculate_global_maximum()
        elif not self.normalize and self.precompute:
            raise nk.InvalidParameterError('It is inefficient to precompute if normalization is not needed.')
      
      
    def phi(self, bstring: NDArray[np.int8]) -> NDArray[np.float32]:
        """
        Calculates individual performances of all agents for a given bitstring.
        If precompute=True, it just looks up the performances in the lookup_table,
        else it calculates the performance of the given bitstring.

        Args:
            bstring: an input bitstring

        Returns:
            P-sized vector with performances of a bitstring for P agents.
        """

        if len(bstring) != self.n * self.p:
            raise nk.InvalidBitstringError("Please enter the full bitstring.")

        if self.precompute:
            return self.performance_lookup[nk.bin2dec(bstring),:] / self.globalmax

        return nk.calculate_performances(bstring, self.interaction_matrix, self.landscape, self.n, self.p) / self.globalmax


    def _generate_interaction_matrix(self):
        """generates the NKCS interaction matrix"""
        inmat = np.zeros((self.n * self.p, self.n * self.p), dtype=np.int8)

        if self.s > (self.p - 1):
            raise nk.InvalidParameterError("The value of S cannot exceed P-1.")

        # internal coupling
        # the idea is to randomly draw only once
        # and have the same internal interaction for
        # all agents. This allows rho=1 to work.
        internal = nk.interaction_matrix(self.n, self.k, "random")
        for i in range(self.p):
            inmat[i*self.n : (i+1)*self.n, i*self.n : (i+1)*self.n] = internal

        # external coupling
        external = nk.random_binary_matrix(self.n, self.c)
        peers_list = nk.generate_couples(self.p, self.s)
        for i, peers in zip(range(self.p), peers_list):
            for peer in peers:
                inmat[i*self.n : (i+1)*self.n, peer*self.n : (peer+1)*self.n] = external

        # save the interaction matrix
        self.interaction_matrix = inmat

    def _generate_landscape(self):
        """generates the landscape given by the interaction matrix."""
        self.landscape = nk.generate_landscape(self.p, self.n, self.k, self.c, self.s, self.rho)

    def _calculate_global_maximum(self):
        """
        !!! WARNING: The most processing-heavy part of the project !!!

        Calculates global maximum for normalization. Set normalize=False to skip.
        
        """

        self.globalmax = nk.get_globalmax(self.interaction_matrix, self.landscape, self.n, self.p)

    def _calculate_all_performances(self):
        """
        !!! WARNING: The most processing-heavy part of the project !!!

        Pre-computes all performances before simulation starts, and saves it
        into a giant lookup table. Set precompute=False to skip this step.
        
        """

        self.performance_lookup, self.globalmax = nk.calculate_all_performances(self.interaction_matrix, self.landscape, self.n, self.p)